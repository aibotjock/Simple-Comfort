#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Simple-Comfort — Single GPU launcher
#
# Starts ONE ComfyUI instance (GPU 0 on port 8188), waits for it to respond,
# points Simple-Comfort at it, and launches the Simple-Comfort web app.
#
# On exit (Ctrl-C or app shutdown) the ComfyUI instance is stopped.
# ---------------------------------------------------------------------------
set -u

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT0=8188
HOST=127.0.0.1

COMFY0_PID=""

cleanup() {
    echo ""
    echo "[single-gpu] Shutting down ComfyUI..."
    [ -n "$COMFY0_PID" ] && kill "$COMFY0_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[single-gpu] Done."
}
trap cleanup EXIT INT TERM

# --- Check GPU count -------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "[single-gpu] Detected $GPU_COUNT GPU(s)."
else
    echo "[single-gpu] WARNING: nvidia-smi not found — cannot verify GPU count. Continuing anyway."
fi

# --- Locate ComfyUI --------------------------------------------------------
find_comfyui() {
    if [ -n "${COMFYUI_PATH:-}" ] && [ -f "$COMFYUI_PATH/main.py" ]; then
        echo "$COMFYUI_PATH"
        return 0
    fi
    for cand in "$HOME/ComfyUI" "$HOME/comfyui" "/opt/ComfyUI" "$HOME/AI/ComfyUI"; do
        if [ -f "$cand/main.py" ]; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

COMFY_DIR="$(find_comfyui)"
if [ -z "$COMFY_DIR" ]; then
    echo "[single-gpu] ERROR: Could not find ComfyUI." >&2
    echo "  Searched: \$COMFYUI_PATH, ~/ComfyUI, ~/comfyui, /opt/ComfyUI, ~/AI/ComfyUI" >&2
    echo "  Fix: install ComfyUI, or set COMFYUI_PATH=/path/to/ComfyUI and re-run." >&2
    exit 1
fi
echo "[single-gpu] Using ComfyUI at: $COMFY_DIR"

# --- Activate venv if present ----------------------------------------------
if [ -d "$APP_DIR/venv" ]; then
    # shellcheck disable=SC1091
    source "$APP_DIR/venv/bin/activate"
    echo "[single-gpu] Activated virtual environment: $APP_DIR/venv"
fi

# --- Pick a python ---------------------------------------------------------
PYTHON=python3
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

wait_for() {
    local url="$1" name="$2" i
    for i in $(seq 1 60); do
        if curl -sf "$url/system_stats" >/dev/null 2>&1; then
            echo "[single-gpu] $name is up ($url)."
            return 0
        fi
        sleep 1
    done
    echo "[single-gpu] ERROR: $name did not respond within 60s ($url)." >&2
    return 1
}

# --- Launch ComfyUI instance 0 (GPU 0, port 8188) --------------------------
echo "[single-gpu] Starting ComfyUI on GPU 0 (port $PORT0)..."
( cd "$COMFY_DIR" && CUDA_VISIBLE_DEVICES=0 "$PYTHON" main.py --port "$PORT0" --listen "$HOST" ) \
    > "$APP_DIR/comfyui-gpu0.log" 2>&1 &
COMFY0_PID=$!

# --- Wait for it to respond -----------------------------------------------
wait_for "http://$HOST:$PORT0" "ComfyUI GPU 0" || exit 1

# --- Write COMFY_ENDPOINTS to .env ----------------------------------------
ENV_FILE="$APP_DIR/.env"
ENDPOINTS="http://$HOST:$PORT0"
if [ -f "$ENV_FILE" ] && grep -q '^COMFY_ENDPOINTS=' "$ENV_FILE"; then
    grep -v '^COMFY_ENDPOINTS=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
    mv "$ENV_FILE.tmp" "$ENV_FILE"
fi
echo "COMFY_ENDPOINTS=$ENDPOINTS" >> "$ENV_FILE"
echo "[single-gpu] Wrote COMFY_ENDPOINTS=$ENDPOINTS to $ENV_FILE"

# --- Start Simple-Comfort --------------------------------------------------
echo "[single-gpu] Starting Simple-Comfort..."
cd "$APP_DIR" || exit 1
"$PYTHON" app.py
