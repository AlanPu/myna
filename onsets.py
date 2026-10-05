"""Locate word onsets from the audio, instead of trusting one global threshold.

The previous pass derived light-up times from an Otsu threshold on the RMS
envelope (-17 dBFS here). That threshold sits high: this recording has a
loud noise floor (median frame -18.9 dB, peak -5.6 dB), so the quiet onsets of
function words get classified as silence and every mark lands late. Measured
against the captions themselves, 27% of sentences started >0.15s after their
caption start.

The fix is a per-frame adaptive gate: a frame is "speech" when it rises above
its own local noise floor, where the floor is tracked from a trailing window.
That follows the recording's drift instead of assuming a fixed level, and it
catches soft onsets that a global cutoff throws away.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess

SR = 16000
HOP = 0.02


def decode(path, start, dur):
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-ss", str(start),
           "-t", str(dur), "-i", path, "-f", "f32le", "-ac", "1",
           "-ar", str(SR), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(out) // 4
    return struct.unpack(f"<{n}f", out[: n * 4])


def envelope(pcm, win=0.025):
    w = int(SR * win)
    step = int(SR * HOP)
    out = []
    for i in range(0, len(pcm) - w + 1, step):
        a = 0.0
        for v in pcm[i:i + w]:
            a += v * v
        out.append(math.sqrt(a / w))
    return out


def smooth(a, k=3):
    out = []
    for i in range(len(a)):
        lo, hi = max(0, i - k), min(len(a), i + k + 1)
        out.append(sum(a[lo:hi]) / (hi - lo))
    return out


def adaptive_gate(env):
    """Return a per-frame boolean: is this frame above its local floor?

    A purely trailing local floor fails when the previous sentence is still
    ringing: the real onset then looks like background and is rejected, which
    pushes the mark late (sentence 11 was 0.80s behind). Pairing the local
    test with a global one fixes that, because the global floor stays low even
    when the trailing window is loud.
    """
    n = len(env)
    srt = sorted(env)
    floor_global = srt[int(n * 0.20)]
    peak = srt[int(n * 0.98)]
    win = int(0.75 / HOP)
    gate = [False] * n
    for i in range(n):
        lo = max(0, i - win)
        seg = sorted(env[lo:i]) or [env[i]]
        local_floor = seg[int(len(seg) * 0.25)]
        local_test = env[i] > max(local_floor * 1.6, peak * 0.035)
        # global test: never require more than this absolute share of peak
        global_test = env[i] > max(floor_global * 2.0, peak * 0.075)
        gate[i] = local_test and global_test
    return gate, floor_global, peak


def runs_of(flag, i0, i1, min_len=0.06, merge_gap=0.10):
    """Contiguous True runs between frame indices, as (start, end) seconds."""
    runs, s = [], None
    for i in range(i0, i1):
        if flag[i]:
            if s is None:
                s = i
        elif s is not None:
            runs.append([s * HOP, i * HOP])
            s = None
    if s is not None:
        runs.append([s * HOP, i1 * HOP])

    # drop specks, then bridge the tiny gaps that separate words
    runs = [r for r in runs if r[1] - r[0] >= min_len]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged]


def syllable(w):
    t = re.sub(r"[^a-z]", "", w.lower())
    if not t:
        return 1
    g = re.findall(r"[aeiouy]+", t)
    n = len(g)
    if t.endswith("e") and n > 1 and not re.search(r"(le|ee|ye|oe)$", t):
        n -= 1
    if re.search(r"[^aeiouy]le$", t):
        n += 1
    return max(1, n)


import re  # noqa: E402  (used by syllable above)


def measure(audio, captions, out_path, start=0.0, log=True):
    if not captions:
        raise SystemExit("no captions")
    span = captions[-1]["end"] + 0.5
    pcm = decode(audio, start, span)
    env = smooth(envelope(pcm))
    gate, floor, peak = adaptive_gate(env)
    n = len(env)
    voiced = sum(gate) / n
    if log:
        print(f"  noise floor {20*math.log10(floor+1e-12):.1f} dB   "
              f"peak {20*math.log10(peak+1e-12):.1f} dB   "
              f"voiced {100*voiced:.0f}%")

    # Pass 1: each sentence starts at its own caption start (rolling cues put
    # it there), so pass 2 can use the next sentence's start as this one's
    # hard end boundary.
    starts = [c["start"] for c in captions]
    for c, s in zip(captions, starts):
        c.setdefault("_s0", s)
    for c, s in zip(captions, starts):
        c["_s0"] = s

    out = []
    for idx, c in enumerate(captions):
        i0 = max(0, int(c["start"] / HOP))
        # A sentence's audio must stop where the next one begins. Rolling cue
        # windows overlap, so searching all the way to this caption's own end
        # swallowed the following sentence: sentence 0 was given speech_end
        # 6.30s while its last word actually finished at ~4.4s, which left its
        # final words ("weren't born like that?") unhighlightable.
        limit = c["end"]
        if idx + 1 < len(captions):
            nxt = captions[idx + 1]
            nxt_start = nxt.get("_s0", nxt["start"])
            limit = min(limit, nxt_start)
        i1 = min(n, int(limit / HOP) + 1)
        if i1 <= i0:
            out.append({**c, "speech_start": c["start"], "speech_end": c["end"]})
            continue
        # Look back a little, but never claim the sentence began before its
        # own caption. YouTube's rolling cues put the caption start at the
        # real onset; a backward search that crosses into the previous
        # sentence (whose tail is still ringing) is what made marks read late.
        rs = runs_of(gate, max(0, i0 - int(0.30 / HOP)), i1)
        if not rs:
            out.append({**c, "speech_start": c["start"], "speech_end": c["end"]})
            continue
        # the caption start is authoritative within a small tolerance
        speech_start = min(c["start"], rs[0][0] + 0.02)
        speech_start = max(speech_start, c["start"] - 0.25)
        # The end is the last frame that still carries voice, not the end of
        # the last "solid" run. runs_of drops short fragments and bridges
        # 0.10s gaps, which clipped a genuine tail: sentence 0 spoke
        # "…born like that?" with level dipping to -30dB around 3.8s and rising
        # again, and the run-based end reported 3.56s instead of ~4.4s.
        tail = None
        for i in range(i1 - 1, i0 - 1, -1):
            if gate[i]:
                tail = i
                break
        speech_end = min((tail + 1) * HOP if tail is not None else limit, limit)
        if speech_end <= speech_start:
            speech_end = min(limit, speech_start + 0.4)
        # word marks: allocate syllable time over the measured speech span
        words = re.split(r"\s+", c["text"].strip())
        words = [w for w in words if w]
        wgt = [syllable(w) for w in words]
        tot = sum(wgt) or len(wgt)
        marks, acc = [], 0
        for k, w in enumerate(wgt):
            marks.append(speech_start + (speech_end - speech_start) * acc / tot)
            acc += w
        out.append({
            **c,
            "speech_start": round(speech_start, 3),
            "speech_end": round(speech_end, 3),
            "word_marks": [round(m, 3) for m in marks],
        })

    json.dump({"captions": out, "hop": HOP,
               "voiced": round(voiced, 3),
               "method": "trailing local floor gate"},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False)

    if log:
        d = [c["speech_start"] - c["start"] for c in out
             if c.get("speech_start") is not None]
        t = [c["end"] - c["speech_end"] for c in out
             if c.get("speech_end") is not None]
        late = sum(1 for x in d if x > 0.15)
        print(f"  onset vs caption: mean {sum(d)/len(d):+.3f}s  "
              f"late>0.15s: {late}/{len(d)}")
        print(f"  tail beyond caption: mean {sum(t)/len(t):.3f}s")
    return out


if __name__ == "__main__":
    man = json.loads(open("player/player_captions.json", encoding="utf-8").read())
    measure("audio2/native.webm", man["captions"], "player/energy.json")
    print("wrote player/energy.json")
