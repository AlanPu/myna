"""End-to-end check of the shadowing score, through a real browser.

Everything else in this project tests against files. This one drives the actual
page: it grants the microphone, records the sentence with the browser's own
MediaRecorder, posts it to /api/score and reads the verdict back off the DOM.

The microphone is Chromium's fake device fed from a real WAV slice of the
source recording, so the captured audio genuinely contains the sentence's
syllables. That matters: with the default beep-tone fake device there is no
speech to align and the scorer correctly answers "cannot align", which would
make this check pass without ever exercising the per-word verdicts.

Usage:  .venv/bin/python verify_e2e_score.py [sentence-index]
"""

from __future__ import annotations

import base64
import glob
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("MYNA_E2E_PORT", "8799"))
BASE = f"http://127.0.0.1:{PORT}/index.html"


def _target() -> int:
    # check.py forwards its own arguments, and one of them may be this check's
    # name. Only a bare integer names a sentence.
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
    """Slice the source sentence out as a WAV for the fake capture device.

    The slice is delayed to imitate capture latency. The page plays the whole
    sentence first and only then opens the microphone, so the fake device is
    already running when the recorder starts; without the delay the first
    syllable would arrive mid-take and the take would be scored against a
    window it does not line up with. 400ms is a realistic figure for
    getUserMedia plus MediaRecorder startup, and it sits inside the onset
    window the scorer is built to absorb.

    The slice is padded at the end so the device is not left with silence
    before the recording is stopped.
    """
    en = json.load(open(ROOT / "player" / "energy.json", encoding="utf-8"))
    # The glob also matches the audio2 directory itself, and handing a
    # directory to ffmpeg fails in a way that looks like a broken fixture
    # rather than a bad path. Keep only real files.
    src = sorted(p for p in glob.glob(str(ROOT / "audio2" / "native_*"))
                 if os.path.isfile(p))
    if not src:
        return None
    c = en["captions"][index]
    s0, s1 = c["speech_start"], c["speech_end"]
    lead, tail = 0.4, 6.0
    out = f"/tmp/myna_fake_mic_{index}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-ss", str(s0), "-t", str(s1 - s0), "-i", src[0],
         # -af belongs AFTER the input: ffmpeg rejects an output filter
         # placed before -i.
         "-af", f"adelay={int(lead*1000)}|{int(lead*1000)},apad=pad_dur={tail}",
         "-ac", "1", "-ar", "48000", out], check=True)
    return out


def main() -> int:
    wav = fake_mic_wav(TARGET)
    if not wav:
        print("no source audio cached; run rebuild.py first")
        return 1

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

            page.goto(BASE)
            # `M` is a top-level `let`, so it is deliberately not on `window`;
            # readiness is observed from the DOM the player actually builds.
            try:
                page.wait_for_function(
                    "() => document.querySelectorAll('#list .row').length > 0",
                    timeout=20000)
            except Exception:
                print("manifest never loaded")
                print("page errors:", errors or "none")
                browser.close()
                return 1

            # Keep the bytes the browser produced and a copy of the server's
            # answer. Every bug this check has caught — recording the wrong
            # sentence, truncating the take, the caption sliding off the scored
            # line — was invisible from the page alone and only showed up here,
            # so the diagnostics stay.
            taken: dict = {}

            def grab(route):
                # Copy the request bytes, then let it through untouched.
                # route.fetch() was tried here and it serialises the page's own
                # requests, which delayed the recording past the wait below.
                body = route.request.post_data or ""
                try:
                    taken["json"] = json.loads(body)
                except Exception:
                    taken["raw_len"] = len(body)
                route.continue_()

            page.route("**/api/score", grab)

            # The page keeps a copy of the last server response, so it can be
            # inspected here. It is installed inside the page rather than via
            # route.fetch(), which serialises the page's own requests and pushes
            # the recording past its window.
            page.evaluate("""() => {
              const f = window.fetch;
              window.__lastScore = null;
              window.fetch = async (...a) => {
                const r = await f(...a);
                try { if (String(a[0]).includes('/api/score'))
                        window.__lastScore = await r.clone().json(); } catch (e) {}
                return r;
              };
            }""")

            # Select the sentence the way a user does: click its row. Setting
            # loopIndex directly left `cur` at -1, so the page fell back to
            # inferring the sentence from a playhead parked at 0 and recorded
            # against sentence 0 instead of TARGET.
            page.click(f"#list .row:nth-child({TARGET + 1})")
            page.wait_for_timeout(300)
            sel = page.evaluate("() => [cur, loopIndex, micIndex]")
            print(f"selected      : cur={sel[0]} loopIndex={sel[1]}")
            if sel[0] != TARGET:
                print("\nFAIL: clicking the row did not select the target "
                      f"sentence (got {sel[0]}, want {TARGET})")
                browser.close()
                return 1

            page.click("#micBtn")
            # The button plays the original first, then opens the mic and
            # records. Wait for the recording state, let it run, then stop.
            try:
                page.wait_for_function(
                    "() => document.getElementById('micBtn')"
                    ".classList.contains('on')",
                    timeout=45000)
            except Exception:
                # Printing what the page logged is the difference between a
                # five-second fix and guessing: the flow fails inside the
                # browser and the reason is only visible in its console.
                print("never reached the recording state")
                print("page errors:", errors or "none")
                browser.close()
                return 1
            # Record long enough for the whole take to land. Waiting on a wall
            # clock is not enough: the microphone takes a few hundred ms to open,
            # so a 3s wait yields a 2.5s recording and truncates the sentence —
            # which is what made this check fail with "cannot align" while every
            # server-side measurement of the same audio scored fine.
            en = json.load(open(ROOT / "player" / "energy.json", encoding="utf-8"))
            cap = en["captions"][TARGET]
            need = cap["speech_end"] - cap["speech_start"]
            page.wait_for_timeout(int(need * 1000) + 2500)
            page.click("#micBtn")

            page.wait_for_function(
                "() => !document.getElementById('scored').hidden", timeout=60000)
            page.wait_for_timeout(400)

            big = page.inner_text("#sBig")
            why = " ".join(page.inner_text("#sWhy").split())
            metrics = " ".join(page.inner_text("#sMetrics").split())
            note = page.inner_text("#sNote").strip()
            words = page.eval_on_selector_all(".cue .w", "els => els.length")
            tally = page.evaluate("""() => {
              const o = {};
              document.querySelectorAll('.cue .w').forEach(e => {
                const c = [...e.classList]
                  .filter(x => ['ok', 'missed', 'late'].includes(x))[0] || 'plain';
                o[c] = (o[c] || 0) + 1;
              });
              return o;
            }""")

            print(f"sentence      : {TARGET}")
            print(f"score         : {big}")
            print(f"metrics       : {metrics}")
            print(f"verdict       : {why}")
            if note:
                print(f"note          : {note}")
            print(f"word spans    : {words} -> {tally}")
            print(f"page errors   : {errors or 'none'}")

            # Write the take to disk. When this check fails, the difference
            # between what the browser sent and what the scorer expected is the
            # only thing that matters, and re-deriving it from a live browser
            # run is slow and flaky.
            if "json" in taken:
                blob = base64.b64decode(taken["json"].get("audio", ""))
                take_path = f"/tmp/myna_e2e_take_{TARGET}.webm"
                with open(take_path, "wb") as f:
                    f.write(blob)
                dur = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", take_path],
                    capture_output=True, text=True).stdout.strip()
                print(f"take          : {take_path} ({len(blob)} bytes, "
                      f"{dur or '?'}s, t0={taken['json'].get('t0')})")

            # The server's own answer vs. the same bytes re-scored here. If they
            # disagree the browser is not the problem; if they agree, the page
            # is discarding a good response.
            if "json" in taken:
                tj = taken["json"]
                print(f"request       : index={tj.get('index')} t0={tj.get('t0')} "
                      f"audio_b64={len(tj.get('audio',''))}")
            if "resp" in taken:
                rs = taken["resp"]
                print(f"server        : score={rs.get('score')} "
                      f"note={rs.get('note')!r} ok={rs.get('ok')}")
            last = page.evaluate("() => window.__lastScore")
            if last is not None:
                print(f"last response : score={last.get('score')} "
                      f"note={last.get('note')!r} ok={last.get('ok')} "
                      f"user_peaks={last.get('user_peaks')} "
                      f"ref_peaks={last.get('ref_peaks')} "
                      f"onset={last.get('onset_ms')}")
            if "offline" in taken:
                print(f"offline       : {taken['offline']}")

            browser.close()

            ok = True
            if big.strip() in ("", "--"):
                print("\nFAIL: no score rendered")
                ok = False
            if words == 0:
                print("\nFAIL: no caption words rendered")
                ok = False
            if tally.get("plain", 0) == words:
                print("\nFAIL: per-word verdicts never reached the caption spans")
                ok = False
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