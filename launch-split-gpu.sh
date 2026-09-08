#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Simple-Comfort — Split GPU launcher (one generation across BOTH GPUs)
#
# Starts ONE ComfyUI instance that sees every installed GPU and arms
# Simple-Comfort's split-GPU mode (SC_SPLIT_GPUS=1): the LTX-2.5 face-lock
# backend wraps its model with MultiGPU_WorkUnits, so each sampling step's
# conditioning work units (cond + uncond) are divided evenly across the
# cards — both 3090s share every single generation instead of alternating
# whole jobs.
#
# Flags mirror the multi-GPU mode proven on this hardware:
#   * NO --cuda-device / CUDA_VISIBLE_DEVICES pinning — pinning hides the
#     second GPU from the process.
#   * --disable-async-offload — the pinned HostBuffer offload path trips
#     "HostBuffer.truncate failed" with MultiGPU deepclones (ComfyUI 0.34).
#   * --fast-disk — a 30GB-RAM box cannot stage a 20GB int8 transformer
#     plus its deepclone in RAM; stream weights from NVMe instead.
#
# Switching modes stops any ComfyUI already running on 8188/8189 first —
# both GPUs must be free for the split instance. Use launch-dual-gpu.sh to
# go back to one-instance-per-GPU (jobs alternate).
# ---------------------------------------------------------------------------
set -u

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT0=8188
HOST=127.0.0.1

COMFY_PID=""

cleanup() {
    echo ""
    echo "[split-gpu] Shutting down the ComfyUI instance we started..."
    [ -n "$COMFY_PID" ] && kill "$COMFY_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[split-gpu] Done."
}
trap cleanup EXIT INT TERM

is_alive() {
    curl -sf "http://$HOST:$1/system_stats" >/dev/null 2>&1
}

# --- Mode switch: free both GPUs -------------------------------------------
for port in 8188 8189; do
    if is_alive "$port"; then
        pids="$(pgrep -f "main\.py.*--port $port" || true)"
        if [ -n "$pids" ]; then
            echo "[split-gpu] Stopping ComfyUI on port $port (mode switch — both GPUs must be free): pid $pids"
            kill $pids 2>/dev/null || true
            for _ in $(seq 1 20); do is_alive "$port" || break; sleep 1; done
            is_alive "$port" && { kill -9 $pids 2>/dev/null || true; sleep 2; }
        fi
    fi
done

# --- Locate ComfyUI ---------------------------------------------------------
find_comfyui() {
    if [ -n "${COMFYUI_PATH:-}" ] && [ -f "$COMFYUI_PATH/main.py" ]; then
        echo "$COMFYUI_PATH"
        return 0
    fi
    for cand in "/data/comfyui" "$HOME/ComfyUI" "$HOME/comfyui" "/opt/ComfyUI" "$HOME/AI/ComfyUI"; do
        if [ -f "$cand/main.py" ]; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

COMFY_DIR="$(find_comfyui)" || {
    echo "[split-gpu] ERROR: Could not find ComfyUI." >&2
    echo "  Searched: \$COMFYUI_PATH, /data/comfyui, ~/ComfyUI, ~/comfyui, /opt/ComfyUI, ~/AI/ComfyUI" >&2
    exit 1
}
echo "[split-gpu] Using ComfyUI at: $COMFY_DIR"

if [ -d "$APP_DIR/venv" ]; then
    # shellcheck disable=SC1091
    source "$APP_DIR/venv/bin/activate"
    echo "[split-gpu] Activated virtual environment: $APP_DIR/venv"
fi

PYTHON=python3
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
COMFY_PY="$COMFY_DIR/.venv/bin/python"
[ -x "$COMFY_PY" ] || COMFY_PY="$PYTHON"
echo "[split-gpu] ComfyUI python: $COMFY_PY"

wait_for() {
    local url="$1" name="$2" i
    for i in $(seq 1 120); do
        if curl -sf "$url/system_stats" >/dev/null 2>&1; then
            echo "[split-gpu] $name is up ($url)."
            return 0
        fi
        sleep 1
    done
    echo "[split-gpu] ERROR: $name did not respond within 120s ($url)." >&2
    return 1
}

# --- Launch the single split instance --------------------------------------
echo "[split-gpu] Starting ComfyUI with every GPU visible (multi-GPU work split)..."
(
    cd "$COMFY_DIR" && "$COMFY_PY" main.py --port "$PORT0" --listen "$HOST" \
        --disable-async-offload --fast-disk \
        > "$APP_DIR/comfyui-splitgpu.log" 2>&1
) &
COMFY_PID=$!

wait_for "http://$HOST:$PORT0" "ComfyUI (split)" || exit 1

# --- Confirm the instance really sees 2+ GPUs ------------------------------
DEVICES="$(curl -sf "http://$HOST:$PORT0/system_stats" | "$PYTHON" -c \
    'import json,sys; print(len(json.load(sys.stdin).get("devices", [])))' 2>/dev/null || echo 0)"
if [ "${DEVICES:-0}" -lt 2 ]; then
    echo "[split-gpu] WARNING: ComfyUI reports $DEVICES GPU(s) — work splitting will be idle." >&2
    echo "  Check comfyui-splitgpu.log (a stale --cuda-device pin or CUDA_VISIBLE_DEVICES hides GPUs)." >&2
else
    echo "[split-gpu] ComfyUI sees $DEVICES GPUs — each generation splits across them."
fi

# --- Write .env --------------------------------------------------------------
ENV_FILE="$APP_DIR/.env"
set_env_line() {
    grep -v "^$1=" "$ENV_FILE" > "$ENV_FILE.tmp" 2>/dev/null || true
    mv "$ENV_FILE.tmp" "$ENV_FILE"
    echo "$1=$2" >> "$ENV_FILE"
}
touch "$ENV_FILE"
set_env_line "COMFY_ENDPOINTS" "http://$HOST:$PORT0"
set_env_line "SC_SPLIT_GPUS" "1"
echo "[split-gpu] Wrote COMFY_ENDPOINTS=http://$HOST:$PORT0 and SC_SPLIT_GPUS=1 to $ENV_FILE"

# --- Start Simple-Comfort ----------------------------------------------------
echo "[split-gpu] Starting Simple-Comfort..."
cd "$APP_DIR" || exit 1
SC_SPLIT_GPUS=1 COMFY_ENDPOINTS="http://$HOST:$PORT0" "$PYTHON" app.py
