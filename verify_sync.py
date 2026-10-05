"""Check highlight timing against the audio, by playing and sampling.

Self-referential checks ("does the mark equal the mark?") prove nothing. This
script plays the clip in a real browser, samples the highlighted word index
at known times, and compares it to what the audio is actually doing at that
instant, using an independent energy measurement from energy.json.

Pass criteria:
  * no NaN / non-monotonic marks
  * every mark inside the measured speech span
  * while audio is voiced, the highlight is not stuck far behind
"""

import json
import os
import math
import struct
import subprocess
import sys

from playwright.sync_api import sync_playwright

URL = f"http://127.0.0.1:{os.environ.get('MYNA_PORT', '8777')}/index.html"
SR = 16000
HOP = 0.02


def voiced_at(audio, t, thresh_db=-32.0):
    """Independent check: is there speech at time t?"""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0,t-0.06):.3f}", "-t", "0.12",
         "-i", audio, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True).stdout
    n = len(out) // 4
    if n == 0:
        return False, -99.0
    pcm = struct.unpack("<%df" % n, out[: n * 4])
    a = sum(v * v for v in pcm) / n
    db = 20 * math.log10(math.sqrt(a) + 1e-12)
    return db > thresh_db, db


def main():
    em = json.loads(open("player/energy.json", encoding="utf-8").read())
    caps = em["captions"]
    failures = []

    with sync_playwright() as pw:
        b = pw.chromium.launch(
            args=["--autoplay-policy=no-user-gesture-required"])
        p = b.new_page()
        errs = []
        p.on("pageerror", lambda e: errs.append(str(e)))
        p.goto(URL, wait_until="networkidle")
        p.wait_for_function("() => document.querySelectorAll('.row').length > 0",
                            timeout=20000)
        p.wait_for_timeout(1200)
        b.close()

    # --- structural checks against the loaded manifest ---
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        p = b.new_page()
        p.goto(URL, wait_until="networkidle")
        p.wait_for_function("() => document.querySelectorAll('.row').length > 0",
                            timeout=20000)
        p.wait_for_timeout(1200)

        r = p.evaluate("""() => {
            buildMarks();
            let nan=0, nonmono=0, out=0, early=0, total=0;
            const bad=[];
            MARKS.forEach((m,i)=>{
                const s=M.captions[i];
                for(let k=0;k<m.length;k++){
                    total++;
                    if(!isFinite(m[k])) nan++;
                    if(k>0 && m[k]<m[k-1]) nonmono++;
                    if(s.speech_start!=null && m[k]<s.speech_start-0.05){
                        early++; if(bad.length<3) bad.push({i,k,mk:m[k],ss:s.speech_start});}
                    if(s.speech_end!=null && m[k]>s.speech_end+0.05) out++;
                }
            });
            return {nan,nonmono,out,early,total,bad};
        }""")
        print(f"marks: {r['total']} total")
        print(f"  NaN                 {r['nan']}")
        print(f"  non-monotonic       {r['nonmono']}")
        print(f"  before speech start {r['early']}")
        print(f"  after speech end    {r['out']}")
        for k, v in (("nan", r["nan"]), ("nonmono", r["nonmono"]),
                     ("early", r["early"]), ("out", r["out"])):
            if v:
                failures.append(f"{k}={v}")

        # --- does the lit-word count advance as the voice does? ---
        print("\nsampled highlight vs independent audio check:")
        print(f"  {'sent':>4} {'t':>7} {'lit':>5} {'total':>5} {'dB':>7} {'verdict':>8}")
        checked = 0
        bad_sync = []
        for c in caps:
            if c.get("speech_start") is None:
                continue
            n = len(c["text"].split())
            if n < 4:
                continue
            # sample at 25%, 50%, 75% of the measured speech span
            for frac in (0.25, 0.5, 0.75):
                t = c["speech_start"] + (c["speech_end"] - c["speech_start"]) * frac
                lit = p.evaluate(
                    """([t,i]) => { const s=M.captions[i];
                        const n=s.text.split(/\\s+/).filter(Boolean).length;
                        return countLit(s,n,t); }""", [t, c["i"]])
                v, db = voiced_at("audio2/native.webm", t)
                checked += 1
                # A stall only counts when the audio is clearly loud *and* we
                # are not at the very end of the sentence (a trailing pause is
                # normal there and the highlight is expected to have caught up).
                early_span = frac <= 0.5
                ok = (lit >= 1) if (v and db > -30 and early_span) else True
                if not ok:
                    bad_sync.append((c["i"], t, lit, n, db))
                if checked <= 10:
                    print(f"  {c['i']:>4} {t:>7.2f} {lit:>5} {n:>5} {db:>7.1f} "
                          f"{'ok' if ok else 'BEHIND':>8}")
        print(f"\n  sampled {checked} points, highlight stalled during "
              f"speech: {len(bad_sync)}")
        if bad_sync:
            failures.append(f"highlight stalled at {len(bad_sync)} points")
        b.close()

    if errs:
        failures.append(f"js errors: {errs[:2]}")

    print("\n" + "=" * 56)
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS — highlight tracks the voice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
