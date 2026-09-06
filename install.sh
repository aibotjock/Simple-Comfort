#!/usr/bin/env bash
# =============================================================================
#  Simple-Comfort — Full Installer
#  https://github.com/aibotjock/Simple-Comfort
#
#  Usage:
#    bash install.sh                   # default install to ~/simple_comfort
#    bash install.sh /custom/path      # install to a custom directory
#
#  What this does:
#    1. Checks system requirements (Python 3.10+, git, ffmpeg)
#    2. Clones Simple-Comfort from GitHub into ~/simple_comfort
#    3. Creates a Python virtual environment and installs dependencies
#    4. Detects GPU count and configures .env (single or dual GPU)
#    5. Installs the desktop icon and app-menu entry
#    6. Prints a clear next-steps summary
#
#  Requirements:
#    - Python 3.10 or later
#    - git
#    - ComfyUI already installed (with LTX 2.5, Wan 2.2, or FramePack models)
#
#  This script does NOT start ComfyUI or modify your ComfyUI installation.
# =============================================================================
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[info]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[ ok ]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[warn]${RESET}  $*"; }
fail()    { echo -e "${RED}[fail]${RESET}  $*"; exit 1; }
section() { echo -e "\n${BOLD}$*${RESET}"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║       Simple-Comfort Installer       ║"
echo "  ║    Local AI Video Generator v1.0     ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${RESET}"

# ── Install directory ─────────────────────────────────────────────────────────
INSTALL_DIR="${1:-$HOME/simple_comfort}"
REPO_URL="https://github.com/aibotjock/Simple-Comfort.git"

info "Install location : $INSTALL_DIR"

# ── 1. System requirements ────────────────────────────────────────────────────
section "Step 1 — Checking requirements"

# Python
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        MAJOR="${VER%%.*}"; MINOR="${VER##*.}"
        if [[ "$MAJOR" -ge 3 && "$MINOR" -ge 10 ]]; then
            PYTHON="$cmd"
            ok "Python $VER found ($cmd)"
            break
        fi
    fi
done
[ -z "$PYTHON" ] && fail "Python 3.10+ is required. Install it from https://python.org and re-run."

# git
command -v git &>/dev/null && ok "git found" || fail "git is required. Install it with: sudo apt install git"

# ffmpeg (optional but recommended)
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg found"
else
    warn "ffmpeg not found — video concat/interpolation will be disabled. Install with: sudo apt install ffmpeg"
fi

# ── 2. Clone or update ────────────────────────────────────────────────────────
section "Step 2 — Downloading Simple-Comfort"

if [ -d "$INSTALL_DIR/.git" ]; then
    info "Existing install found — pulling latest changes..."
    git -C "$INSTALL_DIR" pull --ff-only origin main
    ok "Updated to latest version"
elif [ -d "$INSTALL_DIR" ] && [ "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]; then
    fail "Directory $INSTALL_DIR exists and is not empty. Remove it or choose a different path:\n  bash install.sh /path/to/new/dir"
else
    info "Cloning from GitHub..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    ok "Cloned to $INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# ── 3. Python virtual environment ─────────────────────────────────────────────
section "Step 3 — Installing Python dependencies"

if [ ! -d "venv" ]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv venv
    ok "Virtual environment created"
else
    ok "Existing virtual environment found"
fi

# Activate
# shellcheck disable=SC1091
source venv/bin/activate

info "Installing packages (this may take a minute)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
ok "All dependencies installed"

# ── 4. GPU detection & .env configuration ────────────────────────────────────
section "Step 4 — Detecting GPUs and configuring endpoints"

GPU_COUNT=0
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
else
    warn "nvidia-smi not found — NVIDIA driver may be missing."
    warn "Install it (Ubuntu): sudo ubuntu-drivers autoinstall  then reboot. See GPU_SETUP.md."
fi

if [ "$GPU_COUNT" -eq 0 ]; then
    warn "No GPUs detected by the driver. Run 'bash check-gpu.sh' to diagnose (see GPU_SETUP.md)."
fi

if [ ! -f ".env" ]; then
    if [ "$GPU_COUNT" -ge 2 ]; then
        ok "Detected $GPU_COUNT GPUs — configuring dual-GPU endpoints"
        cat > .env <<EOF
# Simple-Comfort configuration — auto-generated by installer
COMFY_ENDPOINTS=http://127.0.0.1:8188,http://127.0.0.1:8189
PORT=7788
LOG_LEVEL=INFO
EOF
        ok "Dual-GPU .env written (ports 8188 + 8189)"
    else
        info "Single GPU (or GPU count unknown) — configuring single endpoint"
        cat > .env <<EOF
# Simple-Comfort configuration — auto-generated by installer
COMFY_ENDPOINTS=http://127.0.0.1:8188
PORT=7788
LOG_LEVEL=INFO
EOF
        ok "Single-GPU .env written (port 8188)"
    fi
else
    ok ".env already exists — keeping your existing configuration"
fi

# ── 5. Desktop icon & app menu ────────────────────────────────────────────────
section "Step 5 — Installing desktop icon"

if [[ "$(uname -s)" == "Linux" ]]; then
    bash install_desktop_icon.sh
elif [[ "$(uname -s)" == "Darwin" ]]; then
    bash install_desktop_icon.sh
else
    warn "Windows detected — run install_desktop_icon.bat for the desktop icon."
fi

# ── 6. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║     Installation Complete!           ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "  ${BOLD}Installed to:${RESET}  $INSTALL_DIR"
echo ""
echo -e "  ${BOLD}To launch Simple-Comfort:${RESET}"
echo ""
if [ "$GPU_COUNT" -ge 2 ]; then
    echo -e "    ${CYAN}Verify GPUs first:${RESET}"
    echo "      bash $INSTALL_DIR/check-gpu.sh"
    echo ""
    echo -e "    ${CYAN}Dual GPU (recommended — even 50/50 split across both cards):${RESET}"
    echo "      $INSTALL_DIR/launch-dual-gpu.sh"
    echo ""
    echo -e "    ${CYAN}Or single GPU:${RESET}"
fi
echo "      $INSTALL_DIR/launch.sh"
echo "      then open  http://localhost:7788"
echo ""
echo -e "  ${BOLD}Desktop icon:${RESET}  Simple-Comfort in your app menu or Desktop"
if [ -f "$HOME/Desktop/simple-comfort.desktop" ]; then
    echo -e "  ${YELLOW}Note:${RESET} If the desktop icon shows as a text file, right-click → Allow Launching."
fi
echo ""
echo -e "  ${BOLD}ComfyUI:${RESET}  Start ComfyUI first, then launch Simple-Comfort."
echo "    Simple-Comfort connects to your running ComfyUI — it does NOT start it."
if [ "$GPU_COUNT" -ge 2 ]; then
    echo "    For dual GPU, ensure ComfyUI is running on ports 8188 AND 8189."
fi
echo ""
