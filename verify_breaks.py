"""检查句子显示区里的换气斜杠:位置有依据,画在词与词之间,点了能改。

斜杠是给人「照着停」用的,所以**宁少不要错**。三类错误值得一条检查:

  1. 断在短语中间("is a / personality trait")。跟着它读,读者会在录音没停
     的地方停 —— 这比不标更糟。
  2. 斜杠没有依据。页面现在的唯一来源是 energy.json 里实测的停顿
     (pauses:[起, 止, 锚词下标]),由 wordmarks.pauses() 在能量包络上找出来。
     这里不调用页面的 computeBreaks,而是自己从同一份 json 重新判断"这根
     斜杠有没有资格存在",所以规则偷偷变松不会蒙过去。
  3. 锚错了词。停顿的位置对了、挂到了隔壁的词上,屏幕上就是"停错地方"。
     这一条单独用逐词时间(可选 sidecar)重算一遍锚点来对。

另外几条是渲染不变量:
  * 把斜杠 token 去掉、按空白切开,必须原样还原这一句的每个词。
  * 斜杠必须落在**规则选中的那个词之后**。只比根数不够:整屏斜杠齐齐晚
    一个词的版本根数完全一致 —— 第一版就是这么错出去的。
  * 录音态(.cue.live)斜杠颜色必须和平时不同。
  * 点某个词要在它后面加/去一根斜杠,而且要留住(刷新后还在)。

最后是非空断言:整条素材里要有相当比例的句子真的带上了斜杠。一个把渲染
删掉的实现能让前面几条全过(没有斜杠就没有违反)。
"""

from __future__ import annotations

import difflib
import glob
import json
import os
import re
import sys

from playwright.sync_api import sync_playwright

URL = f"http://127.0.0.1:{os.environ.get('MYNA_PORT', '8777')}/index.html"

# 规则里写死的过滤条件,这里当作不变量复核。这份常量是有意抄一遍的:
# 检查就该用"我以为的规则"去 judge 页面,共用一份的话,规则改松了检查
# 也跟着松,等于没查。
MIN_PAUSE = 0.15        # 与页面 BREAK_MIN 一致,单位秒
SOFT_PAUSE = 0.25       # 短于此值才用功能词过滤
SPACING = 2             # 两根斜杠之间至少隔几个词
MAX_PER_SENT = 4
EDGE = 2                # 斜杠两侧各至少几个词
LEAD_STOP = set("the a an of to in on at for with from by as and but or that "
                "which who is are was were be been am my your his her its our "
                "their this these those me him us them".split())

SAMPLE = 12             # 逐句核对渲染的句数


def bare(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


def eligible(seg) -> dict[int, tuple[float, bool, str]]:
    """锚词位置 → (停顿时长, 是否够格, 不够格的原因)。

    键是 j = 停顿前最后一个词的下标(0 起),和页面 BREAKS 的约定一致 ——
    "在第 j 个词之后插一根斜杠"。同一位置有多处停顿时取最长的。
    """
    words = seg["text"].split()
    out: dict[int, tuple[float, bool, str]] = {}
    for p in seg.get("pauses") or []:
        if len(p) < 3:
            continue
        t0, t1, j = p
        dur = t1 - t0
        if not isinstance(j, int) or j < 0 or j >= len(words) - 1:
            continue
        # 判据逐条写在这里,而不是只判"够不够格" —— 报错时说得出是被哪一条
        # 拦下的,否则调规则的时候只能靠猜。
        ok, why = True, ""
        if dur < MIN_PAUSE - 1e-9:
            ok, why = False, f"只停了 {dur:.2f}s"
        elif j + 1 < EDGE and not re.search(r"[,;:—–][\"'”’)\]]*$", words[j]):
            # 左边不够两个词,只有带标点的句首话语标记("So, / let's …")破例
            ok, why = False, f"左边只有 {j + 1} 个词"
        elif len(words) - 1 - j < EDGE:
            ok, why = False, f"右边只有 {len(words) - 1 - j} 个词"
        prev = out.get(j)
        if prev is None or dur > prev[0]:
            out[j] = (dur, ok, why)
        elif ok and not prev[1]:
            out[j] = (dur, ok, why)
    return out


def spans_of(video_id: str):
    """逐词时间 sidecar(可选)。拿不到就跳过锚点那一条。"""
    hits = glob.glob(f"audio2/words_{video_id}.json")
    if not hits:
        return None
    return json.loads(open(hits[0], encoding="utf-8").read()) or None


def anchors(caps, spans) -> dict[int, int]:
    """独立重算"停顿挂到哪个词":词尾 ≤ 停顿起点的最后一个词。

    这是页面锚点规则的**另一份实现**(wordmarks.ends_for 用插值补没对上的
    词,这里直接按整片对齐取词尾),不是调用它 —— 同一个 bug 抄两遍就查不
    出来了。
    """
    flat = [w for c in caps for w in c["text"].split()]
    a = [bare(w) for w in flat]
    b = [bare(x.get("w", "")) for x in spans]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    idx = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for d in range(i2 - i1):
                idx[i1 + d] = j1 + d

    out, off = {}, 0
    for i, c in enumerate(caps):
        words = c["text"].split()
        n = len(words)
        aligned = [off + p in idx for p in range(n)]
        out[i] = (None if not all(aligned) else
                  [spans[idx[off + p]]["e"] for p in range(n)], n)
        off += n
    return out


def main() -> int:
    man = json.loads(open("player/player_captions.json", encoding="utf-8").read())
    en = json.loads(open("player/energy.json", encoding="utf-8").read())
    caps = en["captions"]
    problems: list[str] = []

    spans = spans_of(man["videoId"])
    truth = anchors(caps, spans) if spans else {}

    with sync_playwright() as pw:
        b = pw.chromium.launch()
        p = b.new_page()
        errs: list[str] = []
        p.on("pageerror", lambda e: errs.append(str(e)))
        p.goto(URL, wait_until="networkidle")
        p.wait_for_function("() => document.querySelectorAll('.row').length > 0",
                            timeout=20000)
        p.wait_for_timeout(800)

        # ---- 1. 每一条斜杠都要有依据,并且守住过滤条件 --------------------
        page_breaks = p.evaluate("""() => {
            const out = [];
            for (let i = 0; i < M.captions.length; i++) {
                const w = M.captions[i].text.split(/\\s+/).filter(Boolean);
                out.push({i, n: w.length, b: breaksFor(i), w});
            }
            return out;
        }""")

        n_sent = len(page_breaks)
        with_break = 0
        total_break = 0
        anchor_checked = anchor_bad = 0
        examples: list[tuple[int, str]] = []

        for row in page_breaks:
            i, words, bs = row["i"], row["w"], row["b"]
            if bs:
                with_break += 1
            total_break += len(bs)
            ok = eligible(caps[i]) if i < len(caps) else {}
            for j in bs:
                if j not in ok:
                    problems.append(
                        f"句 {i}: 词 {j} 之后的斜杠没有依据(那里没有实测停顿)")
                    continue
                dur, good, why = ok[j]
                if not good:
                    problems.append(f"句 {i}: 词 {j} 之后的斜杠不该有({why})")
                    continue
                if dur < SOFT_PAUSE - 1e-9 and bare(words[j + 1]) in LEAD_STOP:
                    problems.append(f"句 {i}: 短停顿断在功能词前 "
                                    f"'{words[j + 1]}'")
            for a, c in zip(bs, bs[1:]):
                if c - a < SPACING:
                    problems.append(f"句 {i}: 两个斜杠只隔 {c - a} 个词")
            if len(bs) > MAX_PER_SENT:
                problems.append(f"句 {i}: {len(bs)} 根斜杠,超过上限 "
                                f"{MAX_PER_SENT}")

            # 锚词:停顿位置对了但挂到隔壁的词上,读者照样停错地方。
            if i in truth and truth[i][0]:
                ends, n = truth[i]
                for pause in (caps[i].get("pauses") or []):
                    t0, j = pause[0], pause[2]
                    want = max(0, min(n - 2, sum(1 for e in ends if e <= t0) - 1))
                    anchor_checked += 1
                    if want != j:
                        anchor_bad += 1
                        problems.append(
                            f"句 {i}: 停顿 {t0:.2f}s 锚在词 {j}"
                            f"({words[j]!r}),词尾看应在 {want}"
                            f"({words[want]!r})")
            if bs and len(examples) < 6:
                line = ""
                for k, w in enumerate(words):
                    line += w + (" /" if k in bs else "") + " "
                examples.append((i, line.strip()))

        # ---- 2. 渲染:斜杠插进去不能改变词表,也不能插错位置 ---------------
        idx = [round(i * (n_sent - 1) / (SAMPLE - 1)) for i in range(SAMPLE)]
        rendered = p.evaluate("""(idx) => idx.map(i => {
            seekOrig(M.captions[i].start + 0.05);
            const cue = document.getElementById('cue');
            return {i, text: cue.textContent, expect: breaksFor(i)};
        })""", idx)
        for r in rendered:
            i = r["i"]
            want = caps[i]["text"].split()
            toks = r["text"].split()
            got = [t for t in toks if t != "/"]
            if got != want:
                k = next((j for j in range(min(len(got), len(want)))
                          if got[j] != want[j]), min(len(got), len(want)))
                problems.append(
                    f"句 {i}: 去掉斜杠后词表不一致(第 {k} 个词:"
                    f"{got[k] if k < len(got) else '缺'} "
                    f"≠ {want[k] if k < len(want) else '多'})")
            # 屏幕上数出来的位置必须等于 BREAKS。只比根数是不够的 ——
            # 位置整体偏一个词时根数完全一致。
            seen = 0
            drawn = []
            for t in toks:
                if t == "/":
                    drawn.append(seen - 1)
                else:
                    seen += 1
            if drawn != list(r["expect"]):
                problems.append(f"句 {i}: 斜杠画在位置 {drawn},"
                                f"断点是 {list(r['expect'])}")

        # ---- 3. 点击微调 ---------------------------------------------------
        # 挑一句词数够的,点中间那个词。断言三件事:屏幕上多了一根、位置就在
        # 那个词之后、刷新之后还在。
        pick = next(r["i"] for r in rendered if len(r["expect"]) < MAX_PER_SENT
                    and len(caps[r["i"]]["text"].split()) >= 6)
        k = 2
        click = p.evaluate("""([i, k]) => {
            seekOrig(M.captions[i].start + 0.05);
            const cue = document.getElementById('cue');
            const before = breaksFor(i).slice();
            const el = cue.querySelectorAll('.w')[k];
            el.click();
            const after = breaksFor(i).slice();
            // 斜杠前面是 &nbsp;(防止行尾孤儿),所以不能按空格切 ——
            // textContent 里那是 \u00a0,split(' ') 切不开,"/ " 会连成 "@nbsp;/"。
            const toks = cue.textContent.split(/\\s+/);
            let seen = 0; const drawn = [];
            for (const t of toks) {
                if (t === '/') drawn.push(seen - 1); else seen += 1;
            }
            return {before, after, drawn,
                    stored: localStorage.getItem('myna.breaks.v1'),
                    dataset: cue.dataset.i};
        }""", [pick, k])
        if click["dataset"] != str(pick):
            problems.append(f"点击测试:cue 记的句号是 {click['dataset']},"
                            f"实际停在 {pick} —— 微调会落到别的句子上")
        if click["before"] == click["after"]:
            problems.append(f"点击测试:点了句 {pick} 的第 {k} 个词,"
                            f"断点没变({click['before']})")
        elif sorted(set(click["before"]) ^ set(click["after"])) != [k]:
            problems.append(f"点击测试:断点变化不是在第 {k} 个词之后:"
                            f"{click['before']} → {click['after']}")
        if k not in click["drawn"]:
            problems.append(f"点击测试:断点里有 {k},屏幕上数出来的是 "
                            f"{click['drawn']}")
        if not click["stored"]:
            problems.append("点击测试:微调没有写进 localStorage,刷新就没了")

        # 刷新后还在吗
        p.reload(wait_until="networkidle")
        p.wait_for_function("() => document.querySelectorAll('.row').length > 0",
                            timeout=20000)
        kept = p.evaluate("""(i) => breaksFor(i).slice()""", pick)
        if k not in kept:
            problems.append(f"刷新后句 {pick} 的微调丢了(断点 {kept})")
        # 再点一次要能收掉,并且回写成空表(而不是把默认值又算回来)
        undo = p.evaluate("""([i, k]) => {
            seekOrig(M.captions[i].start + 0.05);
            const cue = document.getElementById('cue');
            cue.querySelectorAll('.w')[k].click();
            return breaksFor(i).slice();
        }""", [pick, k])
        if k in undo:
            problems.append(f"再点一次没有收掉第 {k} 个词后的斜杠,断点 {undo}")

        # ---- 4. 录音态必须把斜杠提亮 --------------------------------------
        # 分两次读、中间等一等:斜杠颜色有 .12s 过渡,加完类立刻读
        # getComputedStyle 拿到的还是过渡起点的颜色。前面那轮 seekOrig 也会
        # 异步触发 seeked → tick(),而 tick 按状态重算 live 类。
        p.evaluate("() => seekOrig(M.captions[0].start + 0.05)")
        p.wait_for_timeout(400)
        base = p.evaluate("""() => {
            const cue = document.getElementById('cue');
            cue.classList.remove('live');
            const el = cue.querySelector('.brk');
            return el ? getComputedStyle(el).color : null;
        }""")
        p.evaluate("() => document.getElementById('cue').classList.add('live')")
        p.wait_for_timeout(250)
        live = p.evaluate("""() => {
            const el = document.querySelector('.cue .brk');
            return el ? getComputedStyle(el).color : null;
        }""")
        p.evaluate("() => document.getElementById('cue').classList.remove('live')")
        colors = None if base is None else {"base": base, "live": live}

        states = p.evaluate("""() => {
            const cue = document.getElementById('cue');
            const probe = (on, off) => {
                on(); const lit = cue.classList.contains('live');
                off(); return lit;
            };
            const sOn = probe(() => setShadowState('recording'),
                              () => setShadowState('idle'));
            const mOn = probe(() => setMicState('recording'),
                              () => setMicState('idle'));
            return {on: sOn && mOn, off: cue.classList.contains('live'),
                    shadow: sOn, mic: mOn};
        }""")

        b.close()

    if errs:
        problems.append(f"页面报错: {errs[:2]}")

    if colors is None:
        problems.append("字幕区里没有 .brk 元素 —— 斜杠根本没渲染出来")
    elif colors["base"] == colors["live"]:
        problems.append(f"录音态斜杠没有提亮({colors['base']})")
    if not states["on"]:
        problems.append(f"切到录音态后 .cue 没拿到 live 类:"
                        f"影子 {states['shadow']} · 录音 {states['mic']}")
    if states["off"]:
        problems.append("录音结束 .cue 仍留在 live 态")

    # ---- 5. 非空:这套断言不能靠"没有斜杠"通过 ---------------------------
    # 下限按当前规则的实际产出(308 根 / 64%)往下留两成余量,不是卡着边界:
    # 卡边界的阈值会在换一段素材时随机报错,而不卡边界的阈值照样能抓住
    # "渲染删掉"这种归零的回归。
    share = with_break / max(1, n_sent)
    if total_break < 240:
        problems.append(f"整条素材只有 {total_break} 根斜杠,太少(下限 240)")
    if share < 0.50:
        problems.append(f"只有 {share:.0%} 的句子带斜杠,太少(下限 50%)")
    if spans and anchor_checked and anchor_bad > anchor_checked * 0.02:
        problems.append(f"{anchor_bad}/{anchor_checked} 处停顿锚错了词"
                        f"(超过 2%)")

    print(f"句子 {n_sent}  带斜杠 {with_break} ({share:.0%})  "
          f"斜杠合计 {total_break}  平均 {total_break/n_sent:.2f}/句")
    if spans:
        print(f"锚词复核 {anchor_checked} 处,与逐词时间不一致 {anchor_bad} 处")
    else:
        print("没有逐词时间 sidecar,跳过锚词复核")
    print("-" * 66)
    for i, line in examples:
        print(f"#{i:>4} {line}")
    print("-" * 66)
    if colors:
        print(f"斜杠颜色 平时 {colors['base']}  录音 {colors['live']}")
    print(f"抽查渲染 {len(rendered)} 句:词表逐句比对,斜杠位置逐根比对")
    print(f"点击微调:句 {pick} 词 {k} 加→刷新仍在→再点收掉")

    if problems:
        print(f"\nFAILED — {len(problems)} 处")
        for msg in problems[:12]:
            print(f"  · {msg}")
        if len(problems) > 12:
            print(f"  … 另有 {len(problems) - 12} 处")
        return 1
    print("\nPASS — 每根斜杠都有实测停顿支撑,锚词对得上,可点可存")
    return 0


if __name__ == "__main__":
    sys.exit(main())
