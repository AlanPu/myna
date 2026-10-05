#!/bin/bash
# Start the Myna shadowing player on a local HTTP server.
#
# Why a server instead of opening index.html directly:
# the page fetches player_captions.json, and browsers block fetch() on
# file:// URLs (CORS). Opening the file directly yields an empty player with
# "Fetch API cannot load file://..." in the console.
#
# Why serve.py rather than `python -m http.server`:
# it serves the same static files and additionally exposes POST /api/build, so
# the "换视频" box in the page can rebuild a different clip.

set -e
cd "$(dirname "$0")"

PORT="${1:-8777}"
URL="http://127.0.0.1:${PORT}/index.html"

# reuse an already-running server on this port
if curl -sS -o /dev/null --max-time 2 "$URL" 2>/dev/null; then
  echo "✓ 已有服务在运行,直接打开:"
  echo "  $URL"
  open "$URL"
  exit 0
fi

# prefer the project venv's python, fall back to system python3
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi

echo "启动本地服务…"
echo "  地址: $URL"
echo "  停止: Ctrl+C"
echo

# background the server, then open the browser
MYNA_PORT="$PORT" "$PY" serve.py &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# wait until it answers
for _ in $(seq 1 20); do
  if curl -sS -o /dev/null --max-time 1 "$URL" 2>/dev/null; then
    echo "✓ 服务就绪,已打开浏览器"
    open "$URL"
    break
  fi
  sleep 0.25
done

wait $SERVER_PID
