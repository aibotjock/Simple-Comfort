"""Shared fixtures and the hermetic environment for the Simple-Comfort suite.

pytest imports this conftest BEFORE any test module (and therefore before any
application module), so the environment overrides below are already in place
when ``config.Settings`` reads them at import time:

* ``OUTPUT_DIR``      -> a per-session temp dir; tests never touch the real
                         ``outputs/`` tree, and the live app's clips never
                         leak into test assertions.
* ``COMFY_ENDPOINTS`` -> a guaranteed-unreachable localhost endpoint
                         (nothing listens on port 9), so the suite never
                         depends on — or talks to — a live ComfyUI and can
                         never queue a render.
* ``COMFY_INPUT_DIR`` -> a per-session temp dir for staged reference images
                         (adapters copy references into this directory).

Every test is hermetic: no GPU, no ComfyUI, no live app process. The API
tests drive the FastAPI app in-process via ``TestClient``.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make the repo root importable (adapters, app, ... live there).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Hermetic environment (set BEFORE the first app import) --------------- #
SESSION_TMP = Path(tempfile.mkdtemp(prefix="simple-comfort-tests-"))
os.environ["OUTPUT_DIR"] = str(SESSION_TMP / "outputs")
os.environ["COMFY_ENDPOINTS"] = "http://127.0.0.1:9"  # nothing listens here
os.environ["COMFY_INPUT_DIR"] = str(SESSION_TMP / "comfy-input")


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """A fresh OUTPUT_DIR for project/clip tests, isolated per test."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out


@pytest.fixture
def reference_image(tmp_path: Path) -> Path:
    """A fake reference image on disk (content irrelevant — only copied)."""
    img = tmp_path / "reference.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-png-but-only-copied")
    return img
