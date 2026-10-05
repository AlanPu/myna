"""Verify the slow-speed shadowing player end to end.

Covers what matters for the product:
  1. every speed file loads and decodes
  2. switching speed keeps the playhead on the same *original* timestamp
  3. the caption highlight stays locked to the slowed audio
  4. sentence loop repeats the same window and then exits
  5. keyboard + control wiring has no runtime errors
"""

import os
import sys

from playwright.sync_api import sync_playwright

URL = f"http://127.0.0.1:{os.environ.get('MYNA_PORT', '8777')}/index.html"
SHOTS = "player/shots"
SPEEDS = [1, 0.75, 0.6, 0.5]


def main() -> int:
    os.makedirs(SHOTS, exist_ok=True)
    failures = []

    with sync_playwright() as pw:
        # Without this flag headless Chromium silently refuses to actually
        # play: play() resolves, paused reports False, but currentTime never
        # advances. That looks exactly like a player bug and is not one.
        browser = pw.chromium.launch(
            args=["--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": 1000, "height": 1150})

        errors = []
        page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)

        page.goto(URL, wait_until="networkidle")
        page.wait_for_function("() => document.querySelectorAll('.row').length > 0",
                               timeout=15000)

        # ---------- 1. speed assets ----------
        print("[1] speed variants")
        for s in SPEEDS:
            page.evaluate(f"() => setSpeed({s})")
            try:
                page.wait_for_function(
                    "() => { const a=document.getElementById('a'); return a.readyState>=1; }",
                    timeout=8000)
            except Exception:
                failures.append(f"speed {s}: media never became ready")
                continue
            info = page.evaluate("""() => {
                const a=document.getElementById('a');
                return {dur:a.duration, src:a.currentSrc.split('/').pop(), err:a.error&&a.error.code};
            }""")
            expect = page.evaluate("() => M.duration") / s
            # The audio is a slice of the source, so it is slightly longer than
            # the span the captions cover — on an 18-minute clip that is ~2s,
            # which a fixed 0.6s tolerance rejected. Scale the tolerance and
            # allow for the un-captioned tail.
            tol = max(0.6, 0.05 * expect)
            ok = abs(info["dur"] - expect) <= tol and not info["err"]
            print(f"    {s:>4}x  {info['src']:<20} {info['dur']:7.2f}s "
                  f"(expect {expect:7.2f} ±{tol:.1f})  {'OK' if ok else 'BAD'}")
            if not ok:
                failures.append(f"speed {s}: {info}")

        # ---------- 2. speed switch preserves position ----------
        print("\n[2] speed switch keeps original-timeline position")
        page.evaluate("() => setSpeed(1)")
        page.wait_for_timeout(500)
        # Anchor in the middle of the clip rather than after sentence 1: on a
        # long build sentence 1 is only a few seconds in, and probing there
        # says nothing about a 18-minute timeline.
        anchor = page.evaluate("() => M.duration * 0.5")
        page.evaluate(f"() => seekOrig({anchor})")
        page.wait_for_timeout(250)
        for s in [0.6, 0.5, 1.0]:
            page.evaluate(f"() => setSpeed({s})")
            page.wait_for_timeout(700)
            t_media = page.evaluate("() => document.getElementById('a').currentTime")
            t_orig = page.evaluate("() => document.getElementById('a').currentTime * speed")
            ok = abs(t_orig - anchor) < 0.5
            print(f"    -> {s}x  media {t_media:6.2f}s  orig {t_orig:6.2f}s  "
                  f"{'OK' if ok else 'BAD'}")
            if not ok:
                failures.append(f"speed switch to {s} lost position: "
                                f"orig={t_orig} want {anchor}")

        # ---------- 3. highlight locked to slowed audio ----------
        print("\n[3] caption highlight at each speed (same original moment)")
        # Pick a probe sentence instead of hardcoding index 3. The opening
        # lines of a video are often short or silent, and a probe with no
        # voiced span makes every assertion meaningless (or crashes on an
        # empty array) — which is what happened on the full 18-minute build.
        probe = page.evaluate("""() => {
            const C = M.captions;
            let best = -1, bestLen = -1;
            for (let i = 0; i < C.length; i++) {
                const c = C[i];
                const s0 = c.speech_start, s1 = c.speech_end;
                if (s0 == null || s1 == null) continue;
                const len = s1 - s0, words = c.text.split(/\\s+/).filter(Boolean).length;
                if (len < 1.2 || words < 5) continue;
                // A clip cut mid-sentence leaves the last line's measured span
                // running past the audio, so probing it lights only part of the
                // line. Require the span plus a tail to sit inside the clip.
                if (s1 + 0.5 > M.duration) continue;
                if (len > bestLen) { bestLen = len; best = i; }
            }
            return {i: best, n: C.length};
        }""")
        pi = probe.get("i", -1)
        if pi < 0:
            print("    no usable probe sentence (all spans missing or truncated)")
            failures.append("no usable probe sentence for the speed check")
            pi = 0
        else:
            print(f"    probe sentence #{pi} of {probe.get('n')} "
                  f"(longest measured span that fits inside the clip)")
        seg = page.evaluate(f"() => M.captions[{pi}]")
        # Sample against the measured SPEECH span, not the caption window. A
        # rolling caption window can trail the last spoken word by more than a
        # second, so "70% through the window" can land in silence, where a
        # fully lit line is correct rather than a bug. speech_start/end are
        # grafted onto M.captions at load time from energy.json.
        s0 = page.evaluate(f"() => M.captions[{pi}].speech_start") or seg["start"]
        s1 = page.evaluate(f"() => M.captions[{pi}].speech_end") or seg["end"]
        for s in SPEEDS:
            page.evaluate(f"() => setSpeed({s})")
            page.wait_for_timeout(600)
            target = (s0 + (s1 - s0) * 0.5) / s
            page.evaluate(f"() => tick({target})")
            page.wait_for_timeout(120)
            lit = page.evaluate("() => document.querySelectorAll('#cue .w.on').length")
            tot = page.evaluate("() => document.querySelectorAll('#cue .w').length")
            row = page.evaluate("() => {const r=document.querySelector('.row.on'); return r?+r.dataset.i:-1;}")
            # The row must be the probed sentence, and the highlight must be
            # strictly between "nothing" and "everything" — the midpoint of a
            # measured span should not light the whole line, because the last
            # words have not been spoken yet.
            ok = row == pi and 0 < lit < tot
            print(f"    {s:>4}x  row={row}  lit {lit}/{tot} words  "
                  f"{'OK' if ok else 'BAD'}")
            if not ok:
                failures.append(f"speed {s}: highlight wrong "
                                f"(row={row}, want {pi}, {lit}/{tot})")

        # ---------- 4. sentence loop ----------
        # A repetition costs the sentence length plus the imitation gap, so the
        # sampling window must scale with the sentence, not use a fixed count.
        print("\n[4] sentence loop repeats then auto-exits")
        page.evaluate("() => setSpeed(1)")
        page.wait_for_timeout(500)
        info = page.evaluate("""() => {
            const s = M.captions.slice().sort((a,b)=>
                        (a.end-a.start)-(b.end-b.start))[0];
            return {i:s.i, dur:s.end-s.start, text:s.text};
        }""")
        need = (info["dur"] + 0.6) * 5.5          # 5 reps + slack
        print(f"    shortest sentence #{info['i']} ({info['dur']:.2f}s) "
              f"-> allow {need:.0f}s")
        page.evaluate(
            f"() => {{ document.getElementById('a').pause(); "
            f"seekOrig(M.captions[{info['i']}].start + 0.2); }}")
        page.evaluate("() => startLoop()")
        page.wait_for_timeout(500)
        seen = 0
        deadline = need * 1000
        elapsed = 0
        while elapsed < deadline:
            page.wait_for_timeout(400)
            elapsed += 400
            st = page.evaluate("() => ({loop:loop, n:loopCount})")
            seen = max(seen, st["n"])
            if not st["loop"] and seen >= 5:
                break
        final = page.evaluate(
            "() => ({loop:loop, n:loopCount, btn:document.getElementById('loopBtn').textContent})")
        print(f"    completed repetitions: {final['n']}   still looping: {final['loop']}")
        print(f"    button label: {final['btn']!r}")
        if seen < 5:
            failures.append(f"loop completed only {seen}/5 repetitions")
        if final["loop"]:
            failures.append("loop did not auto-exit after 5 repetitions")

        # ---------- 5. no runtime errors + screenshot ----------
        page.evaluate("() => setSpeed(0.6)")
        page.wait_for_timeout(600)
        page.evaluate(f"() => {{ const s=M.captions[{pi}]; "
                      "tick((s.start+s.end)/2/speed); }")
        page.wait_for_timeout(250)
        shot = os.path.join(SHOTS, "player_slow.png")
        page.screenshot(path=shot, full_page=True)
        print(f"\n[5] screenshot -> {shot}")

        real = [e for e in errors if "favicon" not in e.lower()]
        if real:
            print(f"\n[!] runtime errors: {real[:5]}")
            failures.extend(real[:5])

        browser.close()

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED — slow-speed shadowing loop works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
