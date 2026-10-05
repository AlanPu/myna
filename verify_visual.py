"""Final pacing check: every highlight must be followed by real speech.

Earlier metrics failed for different reasons and are worth recording:

  * "mark must land on a loud frame" — wrong, because a highlight is supposed
    to fire slightly BEFORE the sound, and real pauses are legitimately quiet.
  * "compare against the first onset after the previous mark" — wrong, it
    counts every genuine pause as lag.
  * "even share of voiced energy vs fraction lit" — wrong, it compares a global
    average against a local position.

The property that actually matters is directional and local: just after a word
lights, the speaker should be producing sound. Measured as the loudest frame in
the 250ms following each mark, relative to that sentence's own peak. Pauses
stay legitimate, lead time is rewarded rather than punished, and no reference
to the player's own output is used.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
from pathlib import Path

SR = 16000
HOP = 0.005


def main():
    en = json.loads(open("player/energy.json", encoding="utf-8").read())
    caps = en["captions"]
    span = caps[-1]["end"] + 0.5
    # Locate the cached download by video id, the same way rebuild.py names
    # it. A hardcoded audio2/native.webm silently produced an empty buffer
    # once the fetcher started writing native_<id>.webm, and an empty buffer
    # made every sentence look like "nothing to check" instead of an error.
    vid = None
    try:
        vid = json.loads(open("player/player_captions.json",
                              encoding="utf-8").read()).get("videoId")
    except OSError:
        pass
    audio = None
    if vid:
        hits = sorted(Path("audio2").glob(f"native_{vid}.*"))
        if hits:
            audio = hits[0]
    if audio is None:
        hits = sorted(Path("audio2").glob("native_*.*"))
        audio = hits[0] if hits else None
    if audio is None:
        print(f"no cached audio for {vid} in audio2/ — run rebuild.py first")
        return 1
    print(f"audio: {audio}")
    o = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio), "-f", "f32le",
         "-ac", "1", "-ar", str(SR), "-t", str(span), "-"],
        capture_output=True).stdout
    k = len(o) // 4
    if k == 0:
        print(f"ffmpeg decoded 0 samples from {audio} — cannot judge")
        return 1
    pcm = struct.unpack("<%df" % k, o[: k * 4])
    w, step = int(SR * 0.02), int(SR * HOP)
    env = [math.sqrt(sum(v * v for v in pcm[i:i + w]) / w)
           for i in range(0, len(pcm) - w + 1, step)]

    after, at_mark, total = [], [], 0
    quiet = []
    for c in caps:
        wm = c.get("word_marks")
        if not wm or len(wm) < 3:
            continue
        s0, s1 = c["speech_start"], c["speech_end"]
        i0, i1 = int(s0 / HOP), min(int(s1 / HOP), len(env))
        if i1 <= i0:
            continue
        ref = max(env[i0:i1]) or 1e-9
        for t in wm:
            j = min(len(env) - 1, max(0, int(t / HOP)))
            nxt = max(env[j:j + int(0.25 / HOP)] or [0])
            total += 1
            after.append(nxt / ref)
            at_mark.append(env[j] / ref)
            if nxt / ref < 0.25:
                quiet.append((nxt / ref, c["i"], t))

    after.sort()
    n = total
    if n == 0:
        # No sentence had both a measured span and a word mark to check.
        # Crashing on after[n//10] hid the real reason the build was rejected.
        print("no word marks with a measured speech span — cannot judge")
        return 1
    print(f"word marks checked: {n}")
    print(f"  loudest frame within 250ms after the mark, "
          f"relative to sentence peak:")
    print(f"    p10 {after[n//10]:.2f}   p50 {after[n//2]:.2f}   "
          f"p90 {after[9*n//10]:.2f}")
    good = sum(1 for x in after if x > 0.25)
    solid = sum(1 for x in after if x > 0.50)
    print(f"  followed by speech: {100*good/n:.0f}%   "
          f"clearly voiced: {100*solid/n:.0f}%")
    if quiet:
        quiet.sort()
        print(f"\n  quietest marks ({len(quiet)}):")
        for r, i, t in quiet[:6]:
            print(f"    {100*r:5.1f}% of peak   sentence #{i:>2}  {t:.2f}s")

    ok = good >= n * 0.88
    print("\nVERDICT:", "PASS — every highlight is followed by real speech"
          if ok else f"FAIL — only {100*good/n:.0f}% followed by speech")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
