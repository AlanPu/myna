"""检查句子显示区里的换气斜杠:位置有依据,并且真的画在词与词之间。

斜杠是给人「照着停」用的,所以**宁少不要错**。两类错误值得一条检查:

  1. 断在短语中间("is a / personality trait")。跟着它读,读者会在录音没停
     的地方停 —— 这比不标更糟。
  2. 斜杠没有依据。页面声称只有两个来源:字幕里逗号类的标点,或 energy.json
     里实测的词间隔 ≥0.70s。第二条在这里用 Python 从同一份 json 重算
     (不调用页面自己的 computeBreaks),所以规则偷偷变松不会蒙过去。

另外三条是渲染不变量:
  * 把斜杠 token 去掉、按空白切开,必须原样还原这一句的每个词。斜杠插进来时
    若吞掉了空格,"you'll be" 会变成 "you'll/be" —— 词表对得上,但人眼看到
    的是两个词粘在一起。
  * 斜杠必须落在**规则选中的那个词之后**。只比根数是不够的:整屏斜杠齐齐晚
    一个词的版本,根数完全一致 —— 第一版就是这么错出去的,这条断言就是为它
    写的。
  * 录音态(.cue.live)斜杠颜色必须和平时不同:用户只在录音跟读时盯这一行,
    平时那根线压得很暗,不提亮就等于没标。

最后是非空断言:整条素材里要有相当比例的句子真的带上了斜杠。一个把渲染删掉
的实现能让前两条全过(没有斜杠就没有违反)。
"""

from __future__ import annotations

import json
import os
import sys

from playwright.sync_api import sync_playwright

URL = f"http://127.0.0.1:{os.environ.get('MYNA_PORT', '8777')}/index.html"

GAP = 0.70                     # 与页面 BREAK_GAP 一致,单位秒
PUNCT = ",;:—–"
SAMPLE = 12                    # 逐句核对渲染的句数

# 规则里写死的过滤条件,这里当作不变量复核。这份名单是有意抄一遍的:
# 检查就该用"我以为的规则"去judge页面,共用一份常量的话,规则改松了
# 检查也跟着松,等于没查。
LEAD_STOP = set("the a an of to in on at for with from by as and but or that "
                "which who is are was were be been am my your his her its our "
                "their this these those me him us them".split())
TRAIL_STOP = set("the a an of to in on at for with from by as is are was were "
                 "be been am do does did can could will would shall should may "
                 "might must have has had that this these those my your his her "
                 "its our their".split())


def bare(w: str) -> str:
    return w.lower().strip("\"'“”‘’()[].,!?;:")


def justified(seg) -> dict[int, str]:
    """这一句里**有资格**成为换气点的位置,不套过滤条件。

    键是 j = 停顿前最后一个词的下标(0 起),和页面 BREAKS 的约定一致 ——
    "在第 j 个词之后插一根斜杠"。
    """
    words = seg["text"].split()
    marks = seg.get("word_marks") or []
    out: dict[int, str] = {}
    for j in range(0, len(words) - 1):
        prev = words[j].rstrip("\"'”’)]")
        if prev and prev[-1] in PUNCT:
            out[j] = "punct"
        elif len(marks) == len(words) and marks[j + 1] - marks[j] >= GAP - 1e-9:
            out[j] = "gap"
    return out


def main() -> int:
    en = json.loads(open("player/energy.json", encoding="utf-8").read())
    caps = en["captions"]
    problems: list[str] = []

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
        kinds = {"punct": 0, "gap": 0}
        examples: list[tuple[int, str]] = []

        for row in page_breaks:
            i, words, bs = row["i"], row["w"], row["b"]
            if bs:
                with_break += 1
            total_break += len(bs)
            ok = justified(caps[i]) if i < len(caps) else {}
            for j in bs:
                if j not in ok:
                    problems.append(
                        f"句 {i}: 词 {j} 之后的斜杠没有依据(既无标点,"
                        f"实测间隔也不到 {GAP}s)")
                    continue
                kinds[ok[j]] += 1
                need = 1 if ok[j] == "punct" else 2
                before, after = j + 1, len(words) - 1 - j
                if before < need or after < need:
                    problems.append(f"句 {i}: 斜杠贴边(词 {j}/{len(words)}")
                if ok[j] == "gap":
                    if bare(words[j + 1]) in LEAD_STOP:
                        problems.append(f"句 {i}: 断在功能词前 "
                                        f"'{words[j + 1]}'")
                    if bare(words[j]) in TRAIL_STOP:
                        problems.append(f"句 {i}: 紧跟在功能词后 "
                                        f"'{words[j]}'")
            for a, c in zip(bs, bs[1:]):
                # 标点之间隔 1 个词是允许的;有一头是间隔推出来的则要 3 个词
                near = 2 if (ok.get(a) == "punct" and ok.get(c) == "punct") else 3
                if c - a < near:
                    problems.append(f"句 {i}: 两个斜杠只隔 {c - a} 个词")
            gaps = sum(1 for j in bs if ok.get(j) == "gap")
            if gaps > 2:
                problems.append(f"句 {i}: 靠间隔推出来的斜杠有 {gaps} 处(上限 2)")
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
                # 只报第一处差异,够定位
                k = next((j for j in range(min(len(got), len(want)))
                          if got[j] != want[j]), min(len(got), len(want)))
                problems.append(
                    f"句 {i}: 去掉斜杠后词表不一致(第 {k} 个词:"
                    f"{got[k] if k < len(got) else '缺'} "
                    f"≠ {want[k] if k < len(want) else '多'})")
            # 斜杠落在"第几个词之后":屏幕上数出来的位置必须等于 BREAKS。
            # 只比根数是不够的 —— 位置整体偏一个词时根数完全一致,而那正是
            # 第一版犯的错(整屏斜杠齐齐晚一个词)。
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

        # ---- 3. 录音态必须把斜杠提亮 --------------------------------------
        # 分两次读、中间等一等:斜杠颜色有 .12s 过渡,加完类立刻读
        # getComputedStyle 拿到的还是过渡起点的颜色,两次会一模一样 ——
        # 这会让"提亮"这条断言永远失败(或永远通过),与页面无关。
        #
        # 前面那轮 seekOrig 会异步触发 seeked → tick(),而 tick() 按
        # micState/shadowState 重算 live 类。测量前必须等这些事件落定,
        # 否则刚加上的 live 会被一次迟到的 tick 摘掉,读回来还是平时色 ——
        # 第一次写这条断言就是这么误报的。
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

        # 颜色只是样式,真正要证的是"录音状态 → .live"这条路。tick() 只在
        # 时间前进时跑,录音时音频是停的,所以这条是靠状态函数补的 ——
        # 把状态切换直接走一遍,断言类真的加上了、停下来又摘掉。
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
        problems.append(f"切到录音态后 .cue 没拿到 live 类(斜杠不会提亮):"
                        f"影子 {states['shadow']} · 录音 {states['mic']}")
    if states["off"]:
        problems.append("录音结束 .cue 仍留在 live 态")

    # ---- 4. 非空:这套断言不能靠"没有斜杠"通过 ---------------------------
    share = with_break / max(1, n_sent)
    if total_break < 60:
        problems.append(f"整条素材只有 {total_break} 根斜杠,太少(下限 60)")
    if share < 0.35:
        problems.append(f"只有 {share:.0%} 的句子带斜杠,太少(下限 35%)")

    print(f"句子 {n_sent}  带斜杠 {with_break} ({share:.0%})  "
          f"斜杠合计 {total_break}  "
          f"(标点 {kinds['punct']} · 实测间隔 {kinds['gap']})")
    print("-" * 66)
    for i, line in examples:
        print(f"#{i:>4} {line}")
    print("-" * 66)
    if colors:
        print(f"斜杠颜色 平时 {colors['base']}  录音 {colors['live']}")
    print(f"抽查渲染 {len(rendered)} 句:词表逐句比对,斜杠位置逐根比对")

    if problems:
        print(f"\nFAILED — {len(problems)} 处")
        for msg in problems[:12]:
            print(f"  · {msg}")
        if len(problems) > 12:
            print(f"  … 另有 {len(problems) - 12} 处")
        return 1
    print("\nPASS — 每根斜杠都有依据,词表未被改动,录音态可辨识")
    return 0


if __name__ == "__main__":
    sys.exit(main())
