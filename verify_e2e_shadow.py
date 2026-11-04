"""End-to-end check of 影子跟读, through a real browser.

What this feature claims is small and completely checkable: the microphone opens
BEFORE the original starts, and when it plays back the original is in the left
ear while your own take is in the right one. So that is all this asserts — and
it asserts it from the produced signal, not from the DOM.

The measurement is a passive tap on the page's output bus: a ChannelSplitter
feeds two AnalyserNodes, and this reads the RMS of each channel while the
replay runs. That is the whole point. "The button exists and has a listener"
proved nothing for the A/B replay in verify_e2e_score.py, and it proves nothing
here either: a panner constructed but never connected, a buffer started at the
wrong offset, or a route left panned all look identical from the DOM.

The fake microphone is the source sentence again, but delayed by three seconds.
That delay is what makes the two channels tell themselves apart: at the start
of the replay the left ear is already speaking while the right ear still holds
the pre-roll, so "left loud, right silent" and "right loud too" are two facts
that can only both be true when the split actually works.

Scoring is deliberately NOT part of this mode, so the check also asserts that
no /api/score request was made. A shadow take that quietly gets scored is a
regression, not a bonus.

Usage:  .venv/bin/python verify_e2e_shadow.py [sentence-index]
"""

from __future__ import annotations

import glob
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("MYNA_E2E_PORT", "8798"))
BASE = f"http://127.0.0.1:{PORT}/index.html"

# How long the fake device stays silent before the sentence starts. Long enough
# that a slow getUserMedia cannot eat it, so the right channel is provably quiet
# while the left one is speaking.
LEAD = 3.0


def _target() -> int:
    for a in sys.argv[1:]:
        if a.isdigit():
            return int(a)
    return 12


TARGET = _target()


def wait_for_port(port: int, timeout: float = 20.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.25):
                return True
        except OSError:
            time.sleep(0.15)
    return False


def fake_mic_wav(index: int) -> str | None:
    """The sentence, preceded by LEAD seconds of silence, as a WAV.

    The recording begins a beat before the original does (that is the whole
    design: the mic must be live before the first syllable), so the take opens
    with pre-roll. A lead of zero would make both ears start speaking on the
    same frame and the channel test would have nothing to separate.
    """
    en = json.load(open(ROOT / "player" / "energy.json", encoding="utf-8"))
    src = sorted(p for p in glob.glob(str(ROOT / "audio2" / "native_*"))
                 if os.path.isfile(p))
    if not src:
        return None
    c = en["captions"][index]
    s0, s1 = c["speech_start"], c["speech_end"]
    out = f"/tmp/myna_shadow_mic_{index}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-ss", str(s0), "-t", str(s1 - s0), "-i", src[0],
         # -12dB on purpose. The fake device otherwise hands the page a
         # full-scale copy of the source, which is louder than any real
         # microphone and would hide the volume matching entirely: a take that
         # is already as loud as the original never needs any gain. A laptop
         # mic at speaking distance is roughly this quiet.
         "-af", f"volume=-12dB,adelay={int(LEAD*1000)}|{int(LEAD*1000)},"
                "apad=pad_dur=8",
         "-ac", "1", "-ar", "48000", out], check=True)
    return out


def sample_levels(page, seconds: float, every: int = 200) -> list[dict]:
    """RMS per channel, repeatedly, while the replay runs.

    Each sample is the loudest of three reads 15ms apart. Speech at an 85ms
    window still dips to near zero between syllables, and a single read would
    report "the right ear is silent" for a channel that is plainly speaking.
    """
    out = []
    end = time.time() + seconds
    while time.time() < end:
        best = {"l": 0.0, "r": 0.0}
        for _ in range(3):
            lv = page.evaluate("() => window.__shadow.level()")
            if lv:
                best = {"l": max(best["l"], lv["l"]), "r": max(best["r"], lv["r"])}
            page.wait_for_timeout(15)
        if best["l"] or best["r"]:
            out.append(best)
        page.wait_for_timeout(every)
    return out


def main() -> int:
    wav = fake_mic_wav(TARGET)
    if not wav:
        print("no source audio cached; run rebuild.py first")
        return 1

    en = json.load(open(ROOT / "player" / "energy.json", encoding="utf-8"))
    cap = en["captions"][TARGET]
    need = cap["speech_end"] - cap["speech_start"]

    srv = subprocess.Popen(
        [sys.executable, "serve.py"], cwd=ROOT,
        env={**os.environ, "MYNA_PORT": str(PORT)},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not wait_for_port(PORT):
        srv.kill()
        print("server did not start")
        print(srv.stdout.read() if srv.stdout else "")
        return 1

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={wav}",
                "--autoplay-policy=no-user-gesture-required",
            ])
            ctx = browser.new_context(permissions=["microphone"])
            page = ctx.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
            page.on("console", lambda m: errors.append("console: " + m.text)
                    if m.type in ("error", "warning") else None)

            # Anything posted to the scorer here is a bug: this mode has no
            # verdict, and a take that gets scored anyway would show a number
            # nobody asked for.
            posts: list[str] = []
            page.route("**/api/score",
                       lambda r: (posts.append(r.request.url), r.continue_()))

            page.goto(BASE)
            try:
                page.wait_for_function(
                    "() => document.querySelectorAll('#list .row').length > 0",
                    timeout=20000)
            except Exception:
                print("manifest never loaded")
                print("page errors:", errors or "none")
                browser.close()
                return 1

            page.click(f"#list .row:nth-child({TARGET + 1})")
            page.wait_for_timeout(300)
            # 选中只是选中。这里以前会自动 play(),于是"跳到第 12 句看一眼"等于
            # "从头放一遍",而想听的人本来就有 ▶ 和 ↻ 重听本句。
            # 判据取播放按钮的文案:它在 play/pause 事件里翻转,是页面自己承认的
            # 播放状态,不是我从 DOM 猜的。
            select_paused = "暂停" not in page.evaluate(
                "() => document.getElementById('play').textContent")
            print(f"sentence      : {TARGET} ({need:.2f}s of speech)  "
                  f"点行后 paused={select_paused}")

            page.click("#shadowBtn")
            try:
                page.wait_for_function(
                    "() => window.__shadow && window.__shadow.state() "
                    "=== 'recording'", timeout=20000)
            except Exception:
                print("never reached the recording state")
                print("page errors:", errors or "none")
                browser.close()
                return 1

            # ---- the ordering the design depends on ----------------------
            # tPlay is written only once the original's own clock has moved, so
            # it is absent for the first ~250ms (arm delay + measurement). Wait
            # for it rather than sampling a fixed 120ms and calling the
            # absence a silence bug.
            try:
                page.wait_for_function(
                    "() => { const c = window.__shadow.clock();"
                    " return !!(c && c.tPlay); }", timeout=6000)
            except Exception:
                print("\nFAIL: the original never started — 影子跟读 recorded "
                      "silence")
                print("page errors:", errors or "none")
                browser.close()
                return 1
            clock = page.evaluate("() => window.__shadow.clock()")
            ctx_state = page.evaluate("() => window.__shadow.ctxState()")
            print(f"ctx state     : {ctx_state}")
            d_ms = clock["tPlay"] - clock["tRec"]
            print(f"mic vs play   : mic opened {d_ms:.0f}ms before the original")
            ok = True
            if not select_paused:
                print("\nFAIL: 点一条字幕就自动播放了 — 选中只该定位,不该开播")
                ok = False
            if d_ms <= 0:
                print("\nFAIL: the original started BEFORE the microphone — the "
                      "head of every sentence would be cut off")
                ok = False
            elif d_ms > 1500:
                print("\nFAIL: the microphone took {:.0f}ms longer than the arm "
                      "delay; the take would open with dead air".format(d_ms))
                ok = False

            # ---- wait for the dual-track replay --------------------------
            try:
                page.wait_for_function(
                    "() => window.__shadow.state() === 'playing'",
                    timeout=int((need + LEAD + 12) * 1000))
            except Exception:
                print("\nFAIL: the take never reached playback")
                print("page errors:", errors or "none")
                browser.close()
                return 1

            take = page.evaluate("() => window.__shadow.take()")
            pan = page.evaluate("() => window.__shadow.pan()")
            route = page.evaluate("() => window.__shadow.route()")
            print(f"take          : {take:.2f}s decoded" if take else "take: none")
            print(f"pan / route   : {pan} / {route}")
            if pan is None or abs(pan - (-1)) > 1e-6:
                print("\nFAIL: the original is not panned left during playback")
                ok = False
            if route != "panned":
                print("\nFAIL: the original is still on the direct route — no "
                      "separation is happening")
                ok = False
            if take is None or take < need:
                print(f"\nFAIL: the take is shorter than the sentence "
                      f"({take}s vs {need:.2f}s) — it was cut off")
                ok = False

            # ---- the button must still be a button ----------------------
            # 第一次实现给回放态加了类名 `play`,而 `.play` 已经被进度条的填充块
            # 占用(position:absolute;top:0;bottom:0;opacity:.3):按钮被拉成整屏
            # 高,在页面左边留下一道半透明蓝框。在 DOM 里它仍然是一个
            # textContent 正确的 button,只有几何能发现这件事。
            geo = page.evaluate("""() => {
              const b = document.getElementById('shadowBtn');
              const r = b.getBoundingClientRect();
              const s = getComputedStyle(b);
              return {x: Math.round(r.x), y: Math.round(r.y),
                      w: Math.round(r.width), h: Math.round(r.height),
                      pos: s.position, op: parseFloat(s.opacity)};
            }""")
            print(f"button box    : {geo['w']}x{geo['h']} at "
                  f"({geo['x']},{geo['y']}) position={geo['pos']} "
                  f"opacity={geo['op']}")
            if geo["pos"] == "absolute" or geo["h"] > 60 or geo["w"] > 320:
                print("\nFAIL: 影子跟读 button is no longer laid out as a button "
                      "— it inherited some other element's positioning. This "
                      "shows up as a coloured band across the page.")
                ok = False
            if geo["op"] < 0.95:
                print("\nFAIL: the button is translucent during playback — it "
                      "picked up the scrub fill's opacity")
                ok = False

            # ---- the actual waveform judgement ---------------------------
            # Short window on purpose: the replay ends on its own a beat after
            # the sentence does, and a sampling loop that outlives it would make
            # the "stop" click below start a *new* take instead of stopping the
            # old one. Long enough, though, to cover the take's pre-roll (the
            # right ear is silent for the first ~3s by construction) and then
            # hear it start.
            levels = sample_levels(page, 4.6)
            if not levels:
                print("\nFAIL: no channel levels at all — the meter is dead")
                browser.close()
                return 1
            ls = [x["l"] for x in levels]
            rs = [x["r"] for x in levels]
            peak_l, peak_r = max(ls), max(rs)
            # A moment with real sound on the left and near-nothing on the
            # right: that is the pre-roll of the take, and it can only exist if
            # the two sources are on separate channels.
            left_only = [i for i, x in enumerate(levels)
                         if x["l"] > peak_l * 0.4 and x["r"] < peak_l * 0.15]
            print(f"left channel  : peak {peak_l:.4f}, "
                  f"{sum(1 for v in ls if v > peak_l * 0.2)} of {len(ls)} samples voiced")
            print(f"right channel : peak {peak_r:.4f}, "
                  f"{sum(1 for v in rs if v > peak_r * 0.2)} of {len(rs)} samples voiced")
            # The profile, not just the summary: when this check fails the only
            # question that matters is WHICH samples were loud on WHICH side.
            print("profile l/r   : " + " ".join(
                f"{x['l']:.2f}/{x['r']:.2f}" for x in levels))
            if peak_l < 0.004:
                print("\nFAIL: the left channel is silent — the original is not "
                      "reaching either ear")
                ok = False
            if peak_r < 0.004:
                print("\nFAIL: the right channel is silent — the take never "
                      "reached the right ear")
                ok = False
            if not left_only:
                print("\nFAIL: both channels always move together — the two "
                      "sources are not separated, they are just mixed")
                ok = False
            else:
                i = left_only[0]
                print(f"separation    : sample {i} left={levels[i]['l']:.4f} "
                      f"right={levels[i]['r']:.4f} (left-only)")

            # ---- the two ears must be at comparable loudness -------------
            # 录音关掉了 AGC,而原声是做过后期的人声,两边天然差十几 dB ——
            # 用户听到的是"我的声音很小",而这跟发音没关系。
            #
            # 判据不能看峰值:take 里只装着句子的某一段,峰值和整句无关(实测
            # 放过一次,右耳峰值反而是左耳的 1.6 倍,而 RMS 已经对齐)。要对齐
            # 的是"语音电平"本身,所以直接问:放大之后,两侧的语音 RMS 相等吗。
            #
            # 口径:页面在解码时就把增益和软限幅烘进了 buffer,所以 levels()['take']
            # 已经是右耳真正听到的电平,不能再乘一次 gain。
            #
            # 峰值单独立一条断言,因为它证的是另一件事:放大到一般响之后没有削波。
            # 旧实现反过来 —— 用 0.98/peak 压住增益,峰值永远合规,代价是右耳
            # 永远差 2-3dB,而且差多少取决于素材的波峰因数。两件事都断言,才不
            # 会用一个指标买到另一个指标的违规。
            gain = page.evaluate("() => window.__shadow.gain()")
            lvls = page.evaluate("() => window.__shadow.levels()")
            pk = page.evaluate("() => window.__shadow.peak()")
            gdb = 20 * math.log10(gain) if gain and gain > 0 else -99
            err = 20 * math.log10(lvls["take"] / lvls["orig"]) if lvls["orig"] else -99
            print(f"take gain     : {gain:.2f}x ({gdb:+.1f}dB)  "
                  f"raw {lvls['raw']:.4f} -> {lvls['take']:.4f}")
            print(f"take peak     : {pk:.3f} (1.0 = 满刻度)")
            print(f"speech rms    : orig {lvls['orig']:.4f}  take {lvls['take']:.4f}"
                  f"  ({err:+.1f}dB off)")
            if gain is None or gain <= 1.05:
                print("\nFAIL: a quiet take was not amplified — the right ear "
                      "would stay far below the original")
                ok = False
            if abs(err) > 1.5:
                print(f"\nFAIL: after matching, the take is still {abs(err):.1f}dB "
                      f"off the original — the gain did not survive to playback")
                ok = False
            if pk is None or pk > 0.99:
                print(f"\nFAIL: the matched take peaks at {pk:.3f} — matching the "
                      f"level by clipping is not matching the level")
                ok = False

            # ---- stopping must restore the normal route -----------------
            # A second press means "stop", never "start another take". If the
            # replay ended by itself the click below would do the latter, and
            # the failure would read as "stop does not work".
            still = page.evaluate("() => window.__shadow.state()")
            if still != "playing":
                print(f"\nFAIL: the replay ended before it could be stopped "
                      f"(state={still!r}) — it should outlast the sentence")
                ok = False
                browser.close()
                print("\n" + ("PASS" if ok else "FAIL"))
                return 0 if ok else 1
            page.click("#shadowBtn")
            page.wait_for_timeout(400)
            after = page.evaluate(
                "() => ({s: window.__shadow.state(), "
                "r: window.__shadow.route(), "
                "l: window.__shadow.level()})")
            print(f"stopped       : state={after['s']!r} route={after['r']!r} "
                  f"level={after['l']['l']:.4f}/{after['l']['r']:.4f}")
            if after["s"] != "idle":
                print("\nFAIL: a second press did not stop the replay")
                ok = False
            if after["r"] != "direct":
                print("\nFAIL: the original was left on the panned route — the "
                      "next normal playback would only use one ear")
                ok = False

            # ---- and it can be replayed ---------------------------------
            page.click("#shadowReplay")
            page.wait_for_timeout(600)
            again = page.evaluate(
                "() => ({s: window.__shadow.state(), "
                "l: window.__shadow.level()})")
            print(f"replay        : state={again['s']!r} "
                  f"left={again['l']['l']:.4f}")
            if again["s"] != "playing" or again["l"]["l"] < 0.004:
                print("\nFAIL: 左右耳重听 did not start audible playback")
                ok = False
            page.click("#shadowBtn")
            page.wait_for_timeout(300)

            # ---- 下一句 must continue the mode, not just move on --------
            # 练一长段时不该每句都回列表里找行:上一句是影子跟读练的,按一下
            # 下一句就该直接在下一句开录。这里断言的是"模式被延续",不是
            # "位置变了" —— 只换位置的那种实现照样能让指示器前进。
            # 上一步已经把回放停掉、状态回到 idle,但"练的是影子跟读"这件事
            # 还该记得,这正是它和录制状态分开记的理由。
            nxt = page.evaluate("() => window.__shadow.index()") + 1
            page.click("#nextBtn")
            try:
                page.wait_for_function(
                    f"() => window.__shadow.state() === 'recording' "
                    f"&& window.__shadow.index() === {nxt}", timeout=15000)
                print(f"next sentence : 继续影子跟读,第 {nxt} 句开始录音")
            except Exception:
                print(f"\nFAIL: 下一句没有续上影子跟读(期望第 {nxt} 句开录,"
                      f"实际 {page.evaluate('() => [window.__shadow.state(), window.__shadow.index()]')})")
                ok = False
            page.click("#shadowBtn")     # 收掉这次录音,别让它去解码回放
            page.wait_for_timeout(300)

            # ---- the message must not have been queued for scoring ------
            if posts:
                print(f"\nFAIL: the shadow take was posted to the scorer "
                      f"({len(posts)} request(s)) — this mode has no verdict")
                ok = False

            browser.close()
            real = [e for e in errors if "favicon" not in e.lower()]
            if real:
                print(f"\nFAIL: page errors: {real}")
                ok = False
            print("\n" + ("PASS" if ok else "FAIL"))
            return 0 if ok else 1
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    sys.exit(main())
