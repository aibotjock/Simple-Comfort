#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Simple-Comfort — GPU doctor
#
# One-shot diagnostic for a multi-GPU box (built for dual NVIDIA RTX 3090 on
# Ubuntu 26, but works for any count). It verifies, in order:
#
#   1. The NVIDIA driver is installed and nvidia-smi works.
#   2. How many GPUs the driver sees, with per-card memory.
#   3. Each GPU index is individually addressable (CUDA_VISIBLE_DEVICES=N).
#   4. torch (in ComfyUI's venv, if found) can see every CUDA device.
#   5. Whether ComfyUI is already up on the dual-GPU ports (8188 / 8189).
#
# It never changes anything — it only reports and prints the exact fix for
# each problem it finds. Run it before launch-dual-gpu.sh whenever the GPUs
# "aren't detected" or the load isn't splitting evenly.
#
#   bash check-gpu.sh
# ---------------------------------------------------------------------------
set -u

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}[ ok ]${RESET}  $*"; }
bad()  { echo -e "${RED}[fail]${RESET}  $*"; }
warn() { echo -e "${YELLOW}[warn]${RESET}  $*"; }
info() { echo -e "${CYAN}[info]${RESET}  $*"; }
head() { echo -e "\n${BOLD}$*${RESET}"; }

HOST=127.0.0.1
PORT0=8188
PORT1=8189
PROBLEMS=0

echo -e "${BOLD}"
echo "  Simple-Comfort GPU doctor"
echo -e "${RESET}"

# --- 1. Driver / nvidia-smi -------------------------------------------------
head "1. NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    bad "nvidia-smi not found — the NVIDIA driver is not installed (or not on PATH)."
    echo "     Fix (Ubuntu): sudo ubuntu-drivers autoinstall   then reboot"
    echo "     Then re-run: bash check-gpu.sh"
    PROBLEMS=$((PROBLEMS + 1))
    # Nothing else is meaningful without the driver.
    echo ""
    exit 1
fi
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
CUDA_VER="$(nvidia-smi 2>/dev/null | grep -oiE 'CUDA Version: *[0-9.]+' | grep -oE '[0-9.]+' | head -n1)"
if [ -n "$DRIVER_VER" ]; then
    ok "Driver $DRIVER_VER installed (nvidia-smi CUDA runtime ${CUDA_VER:-unknown})."
else
    bad "nvidia-smi present but returned no driver version — the driver may be broken."
    echo "     Fix (Ubuntu): sudo ubuntu-drivers autoinstall   then reboot"
    PROBLEMS=$((PROBLEMS + 1))
fi

# --- 2. GPU count + per-card memory ----------------------------------------
head "2. GPUs seen by the driver"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [ "${GPU_COUNT:-0}" -eq 0 ]; then
    bad "Driver sees 0 GPUs."
    echo "     Check power/riser cables and 'lspci | grep -i nvidia', then reboot."
    PROBLEMS=$((PROBLEMS + 1))
else
    ok "Driver sees $GPU_COUNT GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=',' read -r idx name total used util; do
            echo "       GPU$(echo "$idx" | tr -d ' '): $(echo "$name" | sed 's/^ *//')  |  ${used// /} / ${total// /} MiB used  |  ${util// /}% util"
        done
    if [ "$GPU_COUNT" -lt 2 ]; then
        warn "Fewer than 2 GPUs visible — dual-GPU mode needs two. Use launch-single-gpu.sh, or fix the missing card."
        PROBLEMS=$((PROBLEMS + 1))
    fi
fi

# --- 3. Per-index addressability -------------------------------------------
head "3. Each GPU index individually addressable"
LAST_IDX=$((GPU_COUNT - 1))
if [ "$LAST_IDX" -ge 0 ]; then
    for idx in $(seq 0 "$LAST_IDX"); do
        if CUDA_VISIBLE_DEVICES="$idx" nvidia-smi -i 0 --query-gpu=name \
                --format=csv,noheader >/dev/null 2>&1; then
            ok "GPU index $idx is addressable (a ComfyUI instance can pin to it)."
        else
            bad "GPU index $idx is NOT addressable — the instance pinned to it will fail."
            PROBLEMS=$((PROBLEMS + 1))
        fi
    done
fi

# --- 4. torch / CUDA in the ComfyUI venv -----------------------------------
head "4. torch CUDA visibility (ComfyUI venv)"
find_comfyui() {
    if [ -n "${COMFYUI_PATH:-}" ] && [ -f "$COMFYUI_PATH/main.py" ]; then echo "$COMFYUI_PATH"; return 0; fi
    for cand in "/data/comfyui" "$HOME/ComfyUI" "$HOME/comfyui" "/opt/ComfyUI" "$HOME/AI/ComfyUI"; do
        [ -f "$cand/main.py" ] && { echo "$cand"; return 0; }
    done
    return 1
}
COMFY_DIR="$(find_comfyui || true)"
if [ -z "$COMFY_DIR" ]; then
    warn "ComfyUI not found (searched \$COMFYUI_PATH, /data/comfyui, ~/ComfyUI, ~/comfyui, /opt/ComfyUI, ~/AI/ComfyUI)."
    echo "     Set COMFYUI_PATH=/path/to/ComfyUI to include the torch check."
else
    info "ComfyUI at: $COMFY_DIR"
    COMFY_PY="$COMFY_DIR/.venv/bin/python"
    [ -x "$COMFY_PY" ] || COMFY_PY="$(command -v python3 || command -v python)"
    if "$COMFY_PY" -c 'import torch' >/dev/null 2>&1; then
        N="$(CUDA_VISIBLE_DEVICES= "$COMFY_PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
        BUILD="$("$COMFY_PY" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '?')"
        if [ "$N" = "0" ]; then
            bad "torch ($COMFY_PY) sees 0 CUDA devices (torch CUDA build: $BUILD)."
            echo "     Likely a CPU-only torch. Reinstall a CUDA build in the ComfyUI venv, e.g.:"
            echo "       $COMFY_PY -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124"
            echo "     See GPU_SETUP.md."
            PROBLEMS=$((PROBLEMS + 1))
        elif [ "$N" -lt "$GPU_COUNT" ]; then
            warn "torch sees $N CUDA device(s) but the driver sees $GPU_COUNT (torch CUDA build: $BUILD)."
            PROBLEMS=$((PROBLEMS + 1))
        else
            ok "torch sees $N CUDA device(s) (torch CUDA build: $BUILD)."
        fi
    else
        warn "torch not importable from $COMFY_PY — install ComfyUI's deps in its venv (see GPU_SETUP.md)."
    fi
fi

# --- 5. ComfyUI ports -------------------------------------------------------
head "5. ComfyUI endpoints"
port_alive() { curl -sf "http://$HOST:$1/system_stats" >/dev/null 2>&1; }
for p in "$PORT0" "$PORT1"; do
    if port_alive "$p"; then
        ok "ComfyUI is responding on port $p (http://$HOST:$p)."
    else
        info "No ComfyUI on port $p yet — launch-dual-gpu.sh will start one there."
    fi
done

# --- Summary ----------------------------------------------------------------
head "Summary"
echo "  Even split model: Simple-Comfort runs ONE ComfyUI per GPU (GPU0->8188,"
echo "  GPU1->8189) and alternates jobs round-robin, so the two RTX 3090s share"
echo "  the load 50/50. Confirm it live with:  nvidia-smi -l 1  (both cards busy)."
echo ""
if [ "$PROBLEMS" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}All checks passed — you're ready for launch-dual-gpu.sh.${RESET}"
else
    echo -e "  ${YELLOW}${BOLD}$PROBLEMS problem(s) found — fix the items above, then re-run bash check-gpu.sh.${RESET}"
fi
echo ""
exit 0
