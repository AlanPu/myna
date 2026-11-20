"""Derive per-word light-up times from the audio, not from syllable math.

The player had been pacing words by spreading their syllable weights evenly
across the measured speech span. That produces a metronome: in one 30-word
sentence the gaps came out at 0.299s for nearly every word, while the speaker
actually varies pace, pauses, and stretches emphasised words by a factor of
several. Evenly spread marks therefore read as a highlight running ahead of or
behind the voice, depending on where the speaker happened to pause.

This module instead detects syllable nuclei in the audio and snaps each word
to a real nucleus, so pacing follows the recording.
"""

from __future__ import annotations

import json
import math
import re
import struct
import subprocess

SR = 16000
HOP = 0.01


def decode(path, start, dur):
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-ss", str(start),
         "-t", str(dur), "-i", path, "-f", "f32le", "-ac", "1",
         "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    n = len(out) // 4
    return struct.unpack("<%df" % n, out[: n * 4])


def envelope(pcm, win=0.02):
    w = int(SR * win)
    step = int(SR * HOP)
    out = []
    for i in range(0, len(pcm) - w + 1, step):
        a = 0.0
        for v in pcm[i:i + w]:
            a += v * v
        out.append(math.sqrt(a / w))
    return out


def smooth(a, k=2):
    out = []
    for i in range(len(a)):
        lo, hi = max(0, i - k), min(len(a), i + k + 1)
        out.append(sum(a[lo:hi]) / (hi - lo))
    return out


def nuclei(env, floor_pct=0.25, min_gap=0.07, env_gate=None,
           env_gate_frames=None):
    """Syllable nuclei: local maxima that clear a prominence gate.

    env_gate, when given, drops nuclei sitting in near-silence. A tiny
    fraction of marks used to land in a pause (the nearest local maximum to
    the ideal slot was a room-noise blip), which reads on screen as the
    highlight hanging back until the word is actually spoken.

    env_gate_frames is a per-frame boolean speech mask (onsets.adaptive_gate
    returns one) applied INSTEAD of env_gate when supplied. A scalar threshold
    cannot serve a recording made in a noisy room: it is derived from
    percentiles of the whole signal, so raising the noise floor raises the
    threshold with it and the quieter real syllables fall below it. The scorer
    needs that mask for microphone recordings, which are never as clean as the
    master audio this module normally runs on.
    """
    n = len(env)
    srt = sorted(env)
    floor = srt[int(n * floor_pct)]
    peak = srt[int(n * 0.98)]
    gate = max(floor * 1.9, peak * 0.05)
    out, i = [], 1
    while i < n - 1:
        if env[i] > gate and env[i] >= env[i - 1] and env[i] > env[i + 1]:
            j = i
            while (j + 1 < n - 1 and env[j + 1] >= env[j] and env[j + 1] > gate):
                j += 1
            if env_gate_frames is not None:
                loud = bool(env_gate_frames[i])
            else:
                loud = env_gate is None or env[i] >= env_gate
            if loud and (not out or (i - out[-1]) >= int(min_gap / HOP)):
                out.append(i)
            i = j + 1
        else:
            i += 1
    return out, gate




def syllables(w):
    t = re.sub(r"[^a-z]", "", w.lower())
    if not t:
        return 1
    g = re.findall(r"[aeiouy]+", t)
    k = len(g)
    if t.endswith("e") and k > 1 and not re.search(r"(le|ee|ye|oe)$", t):
        k -= 1
    if re.search(r"[^aeiouy]le$", t):
        k += 1
    return max(1, k)




def block_levels(env, block=0.25):
    """每 0.25s 一个"当地说话音量"。

    停顿的门限必须是**相对的**:这个片子的底噪和句子之间的音量都不同,
    一把固定的 dB 门限要么在安静的段落里把气声当停顿,要么在响的段落里
    连着漏掉两个停顿。窗口取 0.25s 而不是更长,是因为说话人换气前常常先
    压低——用 1s 的中位数会把停顿自己的低能量算进"正常音量"里。
    """
    n = max(1, int(block / HOP))
    out = []
    for i in range(0, len(env), n):
        chunk = sorted(env[i:i + n])
        out.append(chunk[len(chunk) // 2] if chunk else 0.0)
    return out


def pauses(env, levels, s0, s1, frac=0.34, minlen=0.10, merge=0.09,
           block=0.25):
    """句子内部的换气停顿,返回 [(start, end), …]。

    判据是"能量掉到当地音量的 frac 以下,持续 minlen 秒"。这个信号和人耳
    听到的停顿基本一致,而 word_marks 的间隔不一致 —— 后者是词标之间的差,
    词标本身在快句子上会漂(第 4 句里 exactly 被标晚了 0.5s,于是它前面的
    真实停顿在间隔上根本看不出来)。

    merge 取 0.09 而不是更小,是为了让"弱一点的停顿"也能连成一片:换气前
    说话人常把尾音压低,中间留一个 20~40ms 的反弹(第 4 句 23.56 之后那处
    的 ratio 一度回到 0.58),合并窗口小的时候它会被切成两段各不足 minlen,
    于是整整一处停顿被漏掉。

    贴在句子两头的静音不算:那是和上一句/下一句之间的间隔,不是句内换气。
    """
    a = int(max(0, s0 / HOP))
    b = int(min(len(env) - 1, s1 / HOP))
    nblk = max(1, int(block / HOP))

    def level(i):
        k = min(len(levels) - 1, i // nblk)
        # 邻块的均值:单块容易被一个响音节带偏
        lo, hi = max(0, k - 1), min(len(levels), k + 2)
        return sum(levels[lo:hi]) / (hi - lo)

    raw, run = [], None
    for i in range(a, b + 1):
        if env[i] < frac * level(i):
            if run is None:
                run = i
        else:
            if run is not None:
                raw.append([run, i])
                run = None
    if run is not None:
        raw.append([run, b])

    merged = []
    for r in raw:
        if merged and (r[0] - merged[-1][1]) * HOP <= merge:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    out = []
    for x, y in merged:
        if (y - x) * HOP < minlen:
            continue
        # 两头贴边的是句间间隔,不是句内换气
        if x * HOP <= s0 + 0.05 or y * HOP >= s1 - 0.05:
            continue
        out.append((round(x * HOP, 3), round(y * HOP, 3)))
    return out


def bare(w):
    return re.sub(r"[^a-z']", "", w.lower())


def load_spans(video_id):
    """逐词时间(可选):audio2/words_<videoId>.json, [{w,s,e}, …]。

    这份数据不是本模块能算出来的 —— 它需要一次真正的语音识别。它只用来
    定**停顿挂到哪个词**,不参与高亮,所以缺了也能跑,只是锚点会退回到
    word_marks(实测只有 16% 的停顿能锚对,见 anchor_pauses 的注释)。
    """
    import glob
    hits = glob.glob(f"audio2/words_{video_id}.json")
    if not hits:
        return None
    data = json.loads(open(hits[0], encoding="utf-8").read())
    return data or None


def align_spans(caption_words, spans):
    """字幕词下标 → spans 下标。

    两边不是同一份文本(逐词时间是识别出来的,字幕是另一个转录,3201 对
    3206 词),只能按词形做最长公共子序列,对上 98%。对不上的词不返回,
    由 ends_for 插值补。
    """
    import difflib
    a = [bare(w) for w in caption_words]
    b = [bare(x.get("w", "")) for x in spans]
    out = {}
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for d in range(i2 - i1):
                out[i1 + d] = j1 + d
    return out


def ends_for(n, s0, s1, known, base=0):
    """每个词"说完"的时刻。known: {词下标: 时刻}(下标相对 base)。

    没对上的词按左右最近的已知点线性插值;句子两头用 speech_start /
    speech_end 当虚拟锚。返回值单调不减 —— 锚词靠的就是这个单调性。
    """
    knots = [(-1, s0)]
    for p in range(n):
        g = base + p
        if g in known:
            knots.append((p, known[g]))
    knots.append((n, s1))            # n = 最后一个词之后
    # 时钟只能往前走,反了就是识别抖动,压平
    for k in range(1, len(knots)):
        if knots[k][1] < knots[k - 1][1]:
            knots[k] = (knots[k][0], knots[k - 1][1])

    ends = []
    ki = 0
    for p in range(n):
        while ki + 1 < len(knots) and knots[ki + 1][0] < p:
            ki += 1
        pa, ta = knots[ki]
        if pa == p:
            ends.append(ta)
            continue
        pb, tb = knots[ki + 1] if ki + 1 < len(knots) else (n, s1)
        span = max(1, pb - pa)
        ends.append(ta + (tb - ta) * (p - pa) / span)
    return ends


def anchor_pauses(ps, ends, n):
    """把每处停顿挂到它前面最后一个"说完"的词上。

    为什么不用 word_marks 直接锚:word_marks 是"词亮起来"的时刻,按音节
    核分片取的,在快句子上会整体后移 —— 第 4 句里 exactly 被标到 22.50,
    而它 21.72 就说完了,于是 22.01 的停顿被挂到 "to" 上,整体错位两个词。
    实测(全片 637 处停顿)只有 16% 和逐词时间一致,37% 错两个词以上。
    """
    import bisect
    out = []
    for p0, p1 in ps:
        j = bisect.bisect_right(ends, p0) - 1
        # 句子开头就说不出话的停顿(ends 全在 p0 之后)挂到第一个词后面没意义,
        # 末尾同理:留一个空短语的斜杠不如不画。
        j = max(0, min(n - 2, j))
        out.append((p0, p1, j))
    return out


def snap_words(words, s0, s1, cand, loud=None):
    """Place each word on a detected nucleus, preserving order.

    Two earlier models failed, both measured against the waveform:

      * syllable-weighted ideal slots pushed every word late (mean lag 0.46s,
        worst 2.35s) because real delivery front- or back-loads a sentence;
      * a DP whose cost was distance from those same even slots collapsed an
        11.4s sentence into its first 4s, since the early peaks happened to sit
        nearest the ideal grid.

    What works is letting the peaks speak: sample one seed peak per word across
    the whole sentence so every region is represented, then let each word pull
    only a short distance (at most a third of its own slot) to the nearest
    better peak. Pacing therefore comes from the recording, and no word can
    migrate out of its region.
    """
    n = len(words)
    if n == 0:
        return [], []
    if not cand or s1 <= s0:
        return ([s0 + (s1 - s0) * (k / n) for k in range(n)],) * 2

    m = len(cand)
    # Partition the peaks by TIME, not by count. Seeding by index spread
    # words evenly over the peak list, so a stretch of speech packed with
    # peaks consumed several words while a slow stretch got none, leaving
    # marks stranded in silence (11% of them). Boundaries halfway between
    # consecutive peaks divide the timeline by where the speaker actually is,
    # and each word then takes the strongest peak in its own slice.
    marks = []
    bounds = []
    for j in range(n + 1):
        pos = j * (m - 1) / n if m > 1 else 0
        lo_i = int(math.floor(pos))
        frac = pos - lo_i
        if lo_i + 1 < m:
            t = cand[lo_i] * (1 - frac) + cand[lo_i + 1] * frac
        else:
            t = cand[min(lo_i, m - 1)]
        bounds.append(t)
    bounds[0] = min(bounds[0], cand[0])
    bounds[n] = max(bounds[n], cand[-1])

    for k in range(n):
        lo, hi = bounds[k], bounds[k + 1]
        if hi <= lo:                       # degenerate slice
            lo, hi = cand[0], cand[-1]
        best, bl = None, lo
        for c in cand:
            if c < lo or c > hi:
                continue
            # energy of the envelope at this peak decides within the slice
            j = min(len(loud) - 1, max(0, int(c / HOP)))
            if loud[j] > bl:
                bl, best = loud[j], c
        marks.append(best if best is not None else
                    min(max(lo, cand[0]), cand[-1]))
    for k in range(1, n):
        marks[k] = max(marks[k], marks[k - 1] + 0.02)
    # No word may light up before the sentence's measured speech begins. The
    # first slice reaches back to cand[0], which can sit a few tens of
    # milliseconds ahead of s0; over an 18-minute build that produced six
    # marks flagged as "early" by verify_sync.
    if marks[0] < s0:
        marks[0] = s0
    marks[-1] = max(marks[-1], s1 - 0.10)
    # the clamp above can invert the first pair; restore monotonicity
    for k in range(1, n):
        marks[k] = max(marks[k], marks[k - 1] + 0.02)
    return marks, bounds






def build(audio, captions, out_path, start=0.0, log=True, spans=None):
    if not captions:
        raise SystemExit("no captions")
    span = captions[-1]["end"] + 0.5
    pcm = decode(audio, start, span)
    env = smooth(envelope(pcm))
    levels = block_levels(env)
    # reject nuclei in real silence: a mark there would visibly lag
    srt = sorted(env)
    env_gate = max(srt[int(len(env) * 0.12)] * 2.4, srt[int(len(env) * 0.98)] * 0.045)
    idx, gate = nuclei(env, env_gate=env_gate)
    times = [i * HOP for i in idx]
    if log:
        print(f"  nuclei {len(idx)}  gate {20*math.log10(gate+1e-12):.1f} dB")

    # 逐词时间整片对齐一次:瓶颈在 SequenceMatcher,按句对齐会把同一段
    # 文本反复比 258 次,而且句子之间对不上时插值链条会断。
    flat = [w for c in captions
            for w in re.split(r"\s+", c["text"].strip()) if w]
    pair = align_spans(flat, spans) if spans else {}
    known = {i: spans[j]["e"] for i, j in pair.items()} if spans else {}
    if log:
        print(f"  word spans {len(known)}/{len(flat)} aligned"
              if spans else "  no word spans: anchoring pauses on word_marks")

    out, snapped, total_w, ref_n = [], 0, 0, 0
    off = 0                      # 当前句第一个词在 flat 里的下标
    for c in captions:
        words = [w for w in re.split(r"\s+", c["text"].strip()) if w]
        s0, s1 = c.get("speech_start"), c.get("speech_end")
        if not words or s0 is None or s1 is None or not (s1 > s0):
            out.append({**c, "word_marks": None, "ref_nuclei": None,
                        "pauses": []})
            off += len(words)
            continue
        # The padding below serves two callers with conflicting needs, so the two
        # windows are computed separately.
        #
        # snap_words wants a little slack past speech_end: a tail nucleus is a
        # legitimate place for the final word's mark to land.
        #
        # ref_nuclei is the reference the scorer aligns a recording against, and
        # it must NOT contain the next sentence. 120 of the 229 adjacent pairs
        # in this build overlap in the audio, by up to 4.08s, and the +0.06 pad
        # walked straight into them: sentence 5 exported peaks out to 32.93s
        # while sentence 6 began at 32.00s, so its reference described two
        # sentences and every reading of it was scored against audio the
        # learner was never asked to produce.
        pad = 0.06
        inside = [t for t in times if s0 - pad <= t <= s1 + pad]
        ref = [t for t in inside if t <= s1]
        marks, _ = snap_words(words, s0, s1, inside, loud=env)
        # 停顿位置来自波形(和"高亮卡住"无关),锚点来自逐词时间 —— 两者
        # 都写进能量文件,页面只负责画,不再自己猜。
        ends = (ends_for(len(words), s0, s1, known, base=off) if spans
                else [round(m, 3) for m in marks])
        ps = anchor_pauses(pauses(env, levels, s0, s1), ends, len(words))
        out.append({**c, "word_marks": [round(m, 3) for m in marks],
                    "ref_nuclei": [round(t, 3) for t in ref],
                    "nuclei": len(inside),
                    "pauses": [[round(a, 3), round(b, 3), j] for a, b, j in ps]})
        off += len(words)
        total_w += len(words)
        snapped += 1
        ref_n += len(ref)
        c["_m"] = marks

    # ref_nuclei is the reference the scorer aligns a recording against. It is
    # emitted here, in the module that already computed the peak times, rather
    # than re-derived per sentence in the scorer: re-running nuclei() on a
    # per-sentence window detects a different set of peaks than the whole-file
    # pass did (the gate is relative to the window's own floor and peak), and
    # the scorer would then compare the user's audio against a reference that
    # never existed in the build.
    json.dump({"captions": out, "hop": HOP,
               "method": "syllable nuclei snapping",
               # 斜杠的锚词是靠什么定的:words = 逐词时间 sidecar,
               # marks = 退回 word_marks(实测只有 16% 对,会差一两个词)
               "pause_anchor": "words" if spans else "marks",
               "nuclei": len(idx),
               "ref_nuclei_total": ref_n},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    if log:
        print(f"  {snapped}/{len(captions)} sentences snapped, "
              f"{total_w} words")
        nps = sum(len(x.get("pauses") or []) for x in out)
        withp = sum(1 for x in out if x.get("pauses"))
        print(f"  pauses {nps} in {withp}/{len(captions)} sentences "
              f"({nps/len(captions):.2f} per sentence)")
        # show how much the pacing actually varies now
        c = next((x for x in out if x.get("word_marks")), None)
        if c and len(c["word_marks"]) > 4:
            g = [round(c["word_marks"][k + 1] - c["word_marks"][k], 3)
                 for k in range(len(c["word_marks"]) - 1)]
            print(f"  example gaps: min {min(g)} max {max(g)} "
                  f"(spread {max(g)/max(min(g),1e-6):.1f}x)")
    return out


if __name__ == "__main__":
    import glob
    man = json.loads(open("player/player_captions.json", encoding="utf-8").read())
    # The cached download is named by video id, not a fixed "native" path.
    hits = glob.glob(f"audio2/native_{man['videoId']}.*")
    if not hits:
        raise SystemExit(f"找不到 audio2/native_{man['videoId']}.*")
    en = json.loads(open("player/energy.json", encoding="utf-8").read())
    build(hits[0], en["captions"], "player/energy.json",
          spans=load_spans(man["videoId"]))
    print("updated player/energy.json with snapped word marks")
