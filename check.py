"""Run every pacing check against the built player, automatically.

Both bugs that produced a visibly lagging highlight were invisible to the
structural self-checks the project started with, because those only compared
the player against its own data. The checks that caught them are here, and
they are run automatically after every rebuild so a regression cannot survive
a video swap.

What each check proves
  verify_switch   walking the clock forward: at the instant the display leaves
                  a sentence, every one of its words had already lit. This is
                  the property the user reported as "the highlight is a few
                  words behind when the sentence ends" — words were being
                  skipped entirely, not merely lit late.
  verify_visual   every highlight is followed by real speech within 250ms,
                  judged against the waveform alone. Catches marks that drift
                  away from the voice.
  verify_sync     no NaN, no non-monotonic marks, no mark outside its speech
                  span, and sampled playback points where the highlight is
                  moving while the speaker is talking.
  verify_slow     the whole shadowing loop still works at 1x/0.75x/0.6x/0.5x.
  verify_score    the shadowing metric separates a correct reading from injected
                  defects: rushing, dragging, a dropped phrase, a noisy room,
                  a late start, and capture latency that must NOT be charged to
                  the speaker.
  verify_e2e_score a real browser records a take with MediaRecorder, posts it,
                  and the per-word verdicts reach the caption spans. The
                  earlier checks could all pass with this path broken.
  verify_e2e_shadow a real browser records a 影子跟读 take — microphone open
                  BEFORE the original starts — and the replay really puts the
                  original in the left ear and the take in the right one,
                  judged by the RMS of each output channel.

Usage:
    .venv/bin/python check.py            # all checks
    .venv/bin/python check.py switch     # one check by name
    .venv/bin/python check.py --no-server   # reuse a server already running
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PLAYER = Path("player")
ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("MYNA_PORT", "8777"))
URL = f"http://127.0.0.1:{PORT}/index.html"

# name -> (script, needs browser?)
# The two score checks start their own server: the e2e one drives a fake
# microphone through a port nothing else uses, and the metric one needs no
# server at all. Both are cheap, so they run with everything else rather than
# being left to be remembered.
CHECKS = [
    ("switch", "verify_switch.py", True),
    ("visual", "verify_visual.py", False),
    ("sync", "verify_sync.py", True),
    ("slow", "verify_slow.py", True),
    ("score", "verify_score.py", False),
    ("e2e-score", "verify_e2e_score.py", False),
    ("e2e-shadow", "verify_e2e_shadow.py", False),
]


def up() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def free_port() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


@contextlib.contextmanager
def server(own: bool):
    """Serve player/ so the page can fetch its json; reuse one if present."""
    if up():
        print(f"  using the server already on :{PORT}")
        yield
        return
    if not free_port():
        raise SystemExit(f"port {PORT} is busy but is not serving the player; "
                         f"stop whatever is on it, or set MYNA_PORT")
    py = sys.executable
    # serve.py rather than http.server: it serves the same files and also
    # exposes /api/build, so a check and a live session behave identically.
    proc = subprocess.Popen(
        [py, "serve.py"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "MYNA_PORT": str(PORT)})
    try:
        for _ in range(40):
            if up():
                break
            time.sleep(0.25)
        else:
            raise SystemExit("player server did not come up")
        yield
    finally:
        if own:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)


def missing_build() -> list[str]:
    """Which per-video build outputs are absent.

    player_captions.json and energy.json describe the ONE video that was last
    built and are gitignored, so a fresh clone has neither. Every check below
    reads them, and without this guard each one dies on its own
    FileNotFoundError — six tracebacks that all say the same thing less
    clearly than one line does.
    """
    return [str(p) for p in (PLAYER / "player_captions.json",
                             PLAYER / "energy.json") if not p.exists()]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reuse = "--no-server" in sys.argv
    only = {a.lower() for a in args}
    todo = [c for c in CHECKS if not only or c[0] in only]
    if not todo:
        raise SystemExit(f"unknown check(s): {', '.join(only)}; "
                         f"choose from {', '.join(c[0] for c in CHECKS)}")

    absent = missing_build()
    if absent:
        raise SystemExit(
            "nothing to check: " + ", ".join(absent) + " missing.\n"
            "These are written by the build, describe the video that was last\n"
            "built, and are not kept in git. Build one first:\n"
            '    .venv/bin/python rebuild.py "<youtube-url>" 0 180 "title"')

    env = dict(os.environ)
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH",
                   str(Path.home() / ".cache/dsh-tools/ms-playwright"))

    results = []
    with contextlib.ExitStack() as stack:
        if any(c[2] for c in todo):
            stack.enter_context(server(own=not reuse))
        for name, script, _ in todo:
            print(f"\n{'='*62}\n  {name}  ({script})\n{'='*62}")
            r = subprocess.run([sys.executable, script], env=env)
            results.append((name, r.returncode == 0))

    print(f"\n{'='*62}\n  summary\n{'='*62}")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
        print("Do not ship this build: the highlight will read as lagging.")
        return 1
    print("\nall checks passed — the highlight tracks the voice, "
          "and the score tracks the voice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
