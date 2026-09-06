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
    echo "[dual-gpu] Shutting down ComfyUI instances we started..."
    # Only stop instances THIS script launched — never touch an already-running
    # ComfyUI session the user started themselves.
    [ -n "$COMFY0_PID" ] && kill "$COMFY0_PID" 2>/dev/null || true
    [ -n "$COMFY1_PID" ] && kill "$COMFY1_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[dual-gpu] Done."
}
trap cleanup EXIT INT TERM

# Return 0 if a ComfyUI instance is already alive on the given port.
is_alive() {
    curl -sf "http://$HOST:$1/system_stats" >/dev/null 2>&1
}

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

# --- Detect already-running ComfyUI instances ------------------------------
# If the user already has ComfyUI running on a port, we connect to it rather
# than starting (or killing) their session.
GPU0_RUNNING=false
GPU1_RUNNING=false
if is_alive "$PORT0"; then
    GPU0_RUNNING=true
    echo "[dual-gpu] ComfyUI already running on port $PORT0 — skipping (will connect to it)."
fi
if is_alive "$PORT1"; then
    GPU1_RUNNING=true
    echo "[dual-gpu] ComfyUI already running on port $PORT1 — skipping (will connect to it)."
fi

# --- Locate ComfyUI (only needed if we must start an instance) -------------
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

COMFY_DIR=""
if [ "$GPU0_RUNNING" = false ] || [ "$GPU1_RUNNING" = false ]; then
    COMFY_DIR="$(find_comfyui)"
    if [ -z "$COMFY_DIR" ]; then
        echo "[dual-gpu] ERROR: Could not find ComfyUI (needed to start a missing instance)." >&2
        echo "  Searched: \$COMFYUI_PATH, /data/comfyui, ~/ComfyUI, ~/comfyui, /opt/ComfyUI, ~/AI/ComfyUI" >&2
        echo "  Fix: install ComfyUI, or set COMFYUI_PATH=/path/to/ComfyUI and re-run." >&2
        exit 1
    fi
    echo "[dual-gpu] Using ComfyUI at: $COMFY_DIR"
else
    echo "[dual-gpu] Both ComfyUI instances already running — no ComfyUI install lookup needed."
fi

# --- Activate venv if present ----------------------------------------------
if [ -d "$APP_DIR/venv" ]; then
    # shellcheck disable=SC1091
    source "$APP_DIR/venv/bin/activate"
    echo "[dual-gpu] Activated virtual environment: $APP_DIR/venv"
fi

# --- Pick pythons -----------------------------------------------------------
# The web app runs on the app venv (activated above) or system python.
PYTHON=python3
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
# ComfyUI must run on ITS OWN venv (torch/CUDA live there), never the app's —
# fall back to the app/system python only if ComfyUI has no venv of its own.
COMFY_PY="$COMFY_DIR/.venv/bin/python"
[ -x "$COMFY_PY" ] || COMFY_PY="$PYTHON"
echo "[dual-gpu] ComfyUI python: $COMFY_PY"

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

# --- Verify GPUs / CUDA before starting anything ----------------------------
# Without a working driver ComfyUI comes up CPU-only and every job crawls —
# fail fast with the fix instead. (Skipped when both instances are already
# running.)
#
# The NVIDIA driver (nvidia-smi) is the SOURCE OF TRUTH for whether the GPUs
# exist and are usable. The torch probe below is only ADVISORY: the app's own
# python (used here as a fallback when ComfyUI has no venv) frequently has no
# torch installed, which previously made this script abort with a bogus
# "no CUDA devices" error even on a perfectly healthy dual-3090 box.
if [ "$GPU0_RUNNING" = false ] || [ "$GPU1_RUNNING" = false ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        SMI_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
        if [ "${SMI_COUNT:-0}" -eq 0 ]; then
            echo "[dual-gpu] ERROR: nvidia-smi reports 0 GPUs — NVIDIA driver missing or broken." >&2
            echo "  Check: nvidia-smi" >&2
            echo "  Fix (Ubuntu): sudo ubuntu-drivers autoinstall  (then reboot)" >&2
            echo "  See GPU_SETUP.md for dual RTX 3090 driver setup on Ubuntu 26." >&2
            exit 1
        fi
        # Confirm BOTH cards are individually addressable — a common failure
        # mode is one card falling off the bus (bad riser/power) so only GPU 0
        # shows up. Each ComfyUI instance is pinned to a single GPU index, so
        # every index we intend to use must resolve on its own.
        for idx in 0 1; do
            if ! CUDA_VISIBLE_DEVICES="$idx" nvidia-smi -i 0 \
                    --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
                echo "[dual-gpu] WARNING: GPU index $idx not individually addressable — the instance pinned to it may fail." >&2
            fi
        done
        if [ "$SMI_COUNT" -lt 2 ]; then
            echo "[dual-gpu] WARNING: only $SMI_COUNT GPU(s) visible to the driver — the second instance may fail." >&2
        else
            echo "[dual-gpu] Driver sees $SMI_COUNT GPU(s) — good."
        fi
    else
        echo "[dual-gpu] WARNING: nvidia-smi not found — cannot verify the driver. Continuing anyway." >&2
        echo "  If jobs run on CPU, install the driver: sudo ubuntu-drivers autoinstall (then reboot)." >&2
    fi

    # Advisory torch probe: only meaningful if the chosen python actually has
    # torch. Never fatal — absence of torch here says nothing about the driver.
    if "$COMFY_PY" -c 'import torch' >/dev/null 2>&1; then
        CUDA_DEVICES="$(CUDA_VISIBLE_DEVICES=0,1 "$COMFY_PY" \
            -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
        if [ "$CUDA_DEVICES" = "0" ]; then
            echo "[dual-gpu] WARNING: torch in $COMFY_PY sees 0 CUDA devices — check the ComfyUI venv's CUDA build (see GPU_SETUP.md)." >&2
        elif [ "$CUDA_DEVICES" != "2" ]; then
            echo "[dual-gpu] WARNING: torch sees $CUDA_DEVICES CUDA device(s), expected 2 — the second instance may fail." >&2
        else
            echo "[dual-gpu] ComfyUI torch sees 2 CUDA devices — good."
        fi
    else
        echo "[dual-gpu] Note: torch not importable from $COMFY_PY — skipping torch CUDA probe (driver check above is authoritative)."
    fi
fi

# --- Launch ComfyUI instance 0 (GPU 0, port 8188) --------------------------
if [ "$GPU0_RUNNING" = true ]; then
    echo "[dual-gpu] Reusing existing ComfyUI on GPU 0 (port $PORT0)."
else
    echo "[dual-gpu] Starting ComfyUI on GPU 0 (port $PORT0)..."
    ( cd "$COMFY_DIR" && CUDA_VISIBLE_DEVICES=0 "$COMFY_PY" main.py --port "$PORT0" --listen "$HOST" ) \
        > "$APP_DIR/comfyui-gpu0.log" 2>&1 &
    COMFY0_PID=$!
fi

# --- Launch ComfyUI instance 1 (GPU 1, port 8189) --------------------------
if [ "$GPU1_RUNNING" = true ]; then
    echo "[dual-gpu] Reusing existing ComfyUI on GPU 1 (port $PORT1)."
else
    echo "[dual-gpu] Starting ComfyUI on GPU 1 (port $PORT1)..."
    ( cd "$COMFY_DIR" && CUDA_VISIBLE_DEVICES=1 "$COMFY_PY" main.py --port "$PORT1" --listen "$HOST" ) \
        > "$APP_DIR/comfyui-gpu1.log" 2>&1 &
    COMFY1_PID=$!
fi

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
