#!/usr/bin/env bash
# Simple-Comfort setup — run once before first launch (Linux / macOS).
# Creates a Python virtual environment and installs all dependencies.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m venv venv || python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. Run './launch.sh' to start Simple-Comfort."
