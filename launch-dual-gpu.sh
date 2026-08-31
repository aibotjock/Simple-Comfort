#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Simple-Comfort — Dual GPU launcher
#
# Starts TWO ComfyUI instances (GPU 0 on port 8188, GPU 1 on port 8189),
# waits for both to respond, points Simple-Comfort at both endpoints, and
# launches the Simple-Comfort web app. Jobs are automatically alternated
# across the two GPUs (round-robin) by the WorkerPool.
#
# On exit (Ctrl-C or app shutdown) both ComfyUI instances are stopped.
# ---------------------------------------------------------------------------
set -u

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT0=8188
PORT1=8189
HOST=127.0.0.1

COMFY0_PID=""
COMFY1_PID=""

cleanup() {
    echo ""
    echo "[dual-gpu] Shutting down ComfyUI instances..."
    [ -n "$COMFY0_PID" ] && kill "$COMFY0_PID" 2>/dev/null || true
    [ -n "$COMFY1_PID" ] && kill "$COMFY1_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[dual-gpu] Done."
}
trap cleanup EXIT INT TERM

# --- Check GPU count -------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "[dual-gpu] Detected $GPU_COUNT GPU(s)."
    if [ "$GPU_COUNT" -lt 2 ]; then
        echo "[dual-gpu] WARNING: fewer than 2 GPUs detected. Consider launch-single-gpu.sh."
    fi
else
    echo "[dual-gpu] WARNING: nvidia-smi not found — cannot verify GPU count. Continuing anyway."
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
    echo "[dual-gpu] ERROR: Could not find ComfyUI." >&2
    echo "  Searched: \$COMFYUI_PATH, ~/ComfyUI, ~/comfyui, /opt/ComfyUI, ~/AI/ComfyUI" >&2
    echo "  Fix: install ComfyUI, or set COMFYUI_PATH=/path/to/ComfyUI and re-run." >&2
    exit 1
fi
echo "[dual-gpu] Using ComfyUI at: $COMFY_DIR"

# --- Pick a python ---------------------------------------------------------
PYTHON=python3
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

wait_for() {
    local url="$1" name="$2" i
    for i in $(seq 1 60); do
        if curl -sf "$url/system_stats" >/dev/null 2>&1; then
            echo "[dual-gpu] $name is up ($url)."
            return 0
        fi
        sleep 1
    done
    echo "[dual-gpu] ERROR: $name did not respond within 60s ($url)." >&2
    return 1
}

# --- Launch ComfyUI instance 0 (GPU 0, port 8188) --------------------------
echo "[dual-gpu] Starting ComfyUI on GPU 0 (port $PORT0)..."
( cd "$COMFY_DIR" && CUDA_VISIBLE_DEVICES=0 "$PYTHON" main.py --port "$PORT0" --listen "$HOST" ) \
    > "$APP_DIR/comfyui-gpu0.log" 2>&1 &
COMFY0_PID=$!

# --- Launch ComfyUI instance 1 (GPU 1, port 8189) --------------------------
echo "[dual-gpu] Starting ComfyUI on GPU 1 (port $PORT1)..."
( cd "$COMFY_DIR" && CUDA_VISIBLE_DEVICES=1 "$PYTHON" main.py --port "$PORT1" --listen "$HOST" ) \
    > "$APP_DIR/comfyui-gpu1.log" 2>&1 &
COMFY1_PID=$!

# --- Wait for both to respond ---------------------------------------------
wait_for "http://$HOST:$PORT0" "ComfyUI GPU 0" || exit 1
wait_for "http://$HOST:$PORT1" "ComfyUI GPU 1" || exit 1

# --- Write COMFY_ENDPOINTS to .env ----------------------------------------
ENV_FILE="$APP_DIR/.env"
ENDPOINTS="http://$HOST:$PORT0,http://$HOST:$PORT1"
if [ -f "$ENV_FILE" ] && grep -q '^COMFY_ENDPOINTS=' "$ENV_FILE"; then
    # Replace existing line (portable sed).
    grep -v '^COMFY_ENDPOINTS=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
    mv "$ENV_FILE.tmp" "$ENV_FILE"
fi
echo "COMFY_ENDPOINTS=$ENDPOINTS" >> "$ENV_FILE"
echo "[dual-gpu] Wrote COMFY_ENDPOINTS=$ENDPOINTS to $ENV_FILE"

# --- Start Simple-Comfort --------------------------------------------------
echo "[dual-gpu] Starting Simple-Comfort..."
cd "$APP_DIR" || exit 1
"$PYTHON" app.py
