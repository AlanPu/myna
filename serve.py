"""Serve the player, and rebuild it when the user pastes a different video.

The page used to be a static file served by `python -m http.server`, which can
only hand over what is already on disk. Swapping videos needs the whole
pipeline — captions, ffmpeg slices, atempo variants, speech measurement and
the audio checks — so the player now talks to this server instead.

Endpoints
  GET  /                     static files from player/
  POST /api/build            {url, start?, length?} -> rebuild, then reload
  GET  /api/build            the in-progress build's log
  GET  /api/now              the currently loaded video, for the default field

The build runs in a child process and streams its log, because a full rebuild
downloads audio and renders four speed variants; that takes minutes and the
browser has to be able to show progress rather than freeze.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _Limited:
    """File wrapper that stops copyfileobj after `remain` bytes."""

    def __init__(self, f, remain: int):
        self.f, self.remain = f, remain

    def read(self, n=-1):
        if self.remain <= 0:
            return b""
        if n is None or n < 0:
            n = self.remain
        d = self.f.read(min(n, self.remain))
        self.remain -= len(d)
        return d

    def close(self):
        self.f.close()

ROOT = Path(__file__).resolve().parent
PLAYER = ROOT / "player"
PORT = int(os.environ.get("MYNA_PORT", "8777"))

PROXY = os.environ.get("MYNA_PROXY", "http://127.0.0.1:7890")

# One build at a time. A second request while one is running gets the same log
# rather than racing it: two ffmpeg passes writing the same file would corrupt
# the player, and the user only ever wants the video they just pasted.
_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_log: list[str] = []
_state = {"status": "idle", "video": None, "title": None, "rc": None}


def log(line: str) -> None:
    _log.append(line)
    print(line, flush=True)


def pump() -> None:
    """Read the child's output into _log until it exits, then free the lock."""
    global _proc
    p = _proc
    assert p is not None
    try:
        for raw in iter(p.stdout.readline, ""):
            log(raw.rstrip("\n"))
        p.wait()
        _state["status"] = "done" if p.returncode == 0 else "failed"
        _state["rc"] = p.returncode
        if p.returncode == 0:
            _state["video"] = _pending.get("video")
            _state["title"] = _pending.get("title")
    finally:
        # The lock has to be released here, not in the request handler: the
        # build outlives the POST that started it, so releasing on return left
        # every later build rejected with "already running" and the browser
        # polling for a completion that had already been reported.
        _proc = None
        if _lock.locked():
            _lock.release()


_pending: dict = {}


def start_build(url: str, start: float, length: float) -> tuple[bool, str]:
    global _proc
    # The ID is only used for the log line; rebuild.py parses it again, so a
    # failure here must not stop the build.
    vid = subprocess.run(
        [sys.executable, "-c",
         "import sys;sys.path.insert(0,'.');import rebuild;"
         "print(rebuild.video_id(sys.argv[1]))", url],
        capture_output=True, text=True, cwd=ROOT).stdout.strip() or "?"

    cmd = [sys.executable, "rebuild.py", url, str(start), str(length),
           f"YouTube {vid}"]
    env = dict(os.environ)
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env[k] = PROXY
    _pending.clear()
    _pending.update({"video": vid, "title": f"YouTube {vid}"})

    if not _lock.acquire(blocking=False):
        return False, "a build is already running"
    _log.clear()
    _state.update(status="building", video=vid, title=None, rc=None)
    _proc = subprocess.Popen(cmd, cwd=ROOT, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             bufsize=1)
    threading.Thread(target=pump, daemon=True).start()
    return True, "building"


def current() -> dict:
    """What the player is showing right now, for prefilling the input."""
    try:
        m = json.loads((PLAYER / "player_captions.json").read_text("utf-8"))
        return {"videoId": m.get("videoId"), "title": m.get("title"),
                "duration": m.get("duration"),
                "sentences": len(m.get("captions", []))}
    except Exception:
        return {"videoId": None, "title": None}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(PLAYER), **kw)

    def send_head(self):
        """Honour Range requests.

        Without this the browser reports seekable = [0, 0] for the audio and
        silently refuses every seek: setting currentTime does nothing, so the
        scrub bar, the caption jumps and the position-preserving speed switch
        all stop working. It only showed up on short clips, because a larger
        file misses the memory cache and forces a real ranged request, so a
        check that passed on a 3-minute build failed on a 90-second one.
        """
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(f.fileno()).st_size
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        if not m:
            f.close()
            self.send_error(400, "Malformed Range")
            return None
        start_s, end_s = m.group(1), m.group(2)
        if start_s:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        else:                       # suffix range: last N bytes
            start = max(0, size - int(end_s or 0))
            end = size - 1
        if start >= size or start > end:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        end = min(end, size - 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        f.seek(start)
        return _Limited(f, end - start + 1)

    def log_message(self, *a):        # keep the build log readable
        pass

    def end_headers(self):
        # A rebuild rewrites player_captions.json and energy.json in place, so
        # the browser must not serve the previous build from its cache. The
        # audio files are large and immutable per build, but they change too,
        # and a stale <audio> is far more confusing than a re-fetch.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/build":
            with _lock:
                running = _proc is not None
            return self._json(200, {"running": running, "status": _state["status"],
                                    "log": _log[-400:], "rc": _state["rc"],
                                    **current()})
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/build":
            return self._json(404, {"error": "no such endpoint"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": f"bad json: {e}"})

        url = (body.get("url") or "").strip()
        if not url:
            return self._json(400, {"error": "请填写视频网址或 ID"})
        try:
            start = float(body.get("start", 0))
            length = float(body.get("length", 180))
        except (TypeError, ValueError):
            return self._json(400, {"error": "起点或时长不是数字"})
        if start < 0:
            return self._json(400, {"error": "起点不能为负数"})
        # length 0 is a real request — "take the whole video" — not a default
        if length < 0:
            return self._json(400, {"error": "时长不能为负数,0 表示完整音频"})

        ok, msg = start_build(url, start, length)
        if not ok:
            return self._json(409, {"error": msg})
        return self._json(200, {"ok": True, "message": msg})


def main() -> int:
    # a rebuild is long; the browser polls, so nothing here blocks on it
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    here = current()
    print(f"Myna player  ->  http://127.0.0.1:{PORT}/index.html")
    if here.get("videoId"):
        print(f"current clip: {here['title']}  ({here['videoId']})")
    print("Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
