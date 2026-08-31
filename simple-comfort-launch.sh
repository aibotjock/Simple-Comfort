#!/usr/bin/env bash
# Simple-Comfort launcher
# Starts the local web server (if not already running) and opens the browser.
APP_DIR="/home/ubuntu/simple_comfort"
PIDFILE="$APP_DIR/.sc.pid"
URL="http://localhost:7788"

# Prevent duplicate instances — if already running, just open the browser.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    xdg-open "$URL" 2>/dev/null || true
    exit 0
fi

cd "$APP_DIR" || exit 1
python3 app.py &
SC_PID=$!
echo "$SC_PID" > "$PIDFILE"

# Wait for the server to be ready (max 15s).
for _ in $(seq 1 15); do
    sleep 1
    if curl -sf "$URL/" > /dev/null 2>&1; then
        break
    fi
done

# Open the browser.
xdg-open "$URL" 2>/dev/null || true

# Keep the launcher alive alongside the server, then clean up.
wait "$SC_PID"
rm -f "$PIDFILE"
