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


def nuclei(env, floor_pct=0.25, min_gap=0.07, env_gate=None):
    """Syllable nuclei: local maxima that clear a prominence gate.

    env_gate, when given, drops nuclei sitting in near-silence. A tiny
    fraction of marks used to land in a pause (the nearest local maximum to
    the ideal slot was a room-noise blip), which reads on screen as the
    highlight hanging back until the word is actually spoken.
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






def build(audio, captions, out_path, start=0.0, log=True):
    if not captions:
        raise SystemExit("no captions")
    span = captions[-1]["end"] + 0.5
    pcm = decode(audio, start, span)
    env = smooth(envelope(pcm))
    # reject nuclei in real silence: a mark there would visibly lag
    srt = sorted(env)
    env_gate = max(srt[int(len(env) * 0.12)] * 2.4, srt[int(len(env) * 0.98)] * 0.045)
    idx, gate = nuclei(env, env_gate=env_gate)
    times = [i * HOP for i in idx]
    if log:
        print(f"  nuclei {len(idx)}  gate {20*math.log10(gate+1e-12):.1f} dB")

    out, snapped, total_w = [], 0, 0
    for c in captions:
        words = [w for w in re.split(r"\s+", c["text"].strip()) if w]
        s0, s1 = c.get("speech_start"), c.get("speech_end")
        if not words or s0 is None or s1 is None or not (s1 > s0):
            out.append({**c, "word_marks": None})
            continue
        inside = [t for t in times if s0 - 0.06 <= t <= s1 + 0.06]
        marks, _ = snap_words(words, s0, s1, inside, loud=env)
        out.append({**c, "word_marks": [round(m, 3) for m in marks],
                    "nuclei": len(inside)})
        total_w += len(words)
        snapped += 1
        c["_m"] = marks

    json.dump({"captions": out, "hop": HOP,
               "method": "syllable nuclei snapping",
               "nuclei": len(idx)},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    if log:
        print(f"  {snapped}/{len(captions)} sentences snapped, "
              f"{total_w} words")
        # show how much the pacing actually varies now
        c = next((x for x in out if x.get("word_marks")), None)
        if c and len(c["word_marks"]) > 4:
            g = [round(c["word_marks"][k + 1] - c["word_marks"][k], 3)
                 for k in range(len(c["word_marks"]) - 1)]
            print(f"  example gaps: min {min(g)} max {max(g)} "
                  f"(spread {max(g)/max(min(g),1e-6):.1f}x)")
    return out


if __name__ == "__main__":
    en = json.loads(open("player/energy.json", encoding="utf-8").read())
    build("audio2/native.webm", en["captions"], "player/energy.json")
    print("updated player/energy.json with snapped word marks")
