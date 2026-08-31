# Simple-Comfort

> A deliberately simple local WebUI for generating 30-second AI video clips.

Simple-Comfort is a thin, local-only web front end that wraps a running
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance and turns it into
a one-click **30-second AI video generator**. You pick a model, type a prompt
(or drop a reference image), and it handles all the internal sectioning,
frame-to-frame continuity, and stitching to produce a single 30-second clip.

No cloud calls, no API keys, no telemetry — everything runs on your machine.

---

## Features

- **One-click 30s clips** — the app automatically splits generation into
  model-appropriate sections, carries the last frame of each section into the
  next for continuity, and concatenates everything with FFmpeg.
- **Three model back-ends** — LTX-2.3, Wan 2.2 and FramePack, each behind a
  self-contained adapter.
- **Text-to-Video and Image-to-Video** — with drag-and-drop reference images.
- **Continuity** — continue a new clip from the last frame of a previous one.
- **Live progress** — real-time progress, ETA and preview frames over WebSocket.
- **Multi-GPU** — point it at several ComfyUI endpoints and it load-balances
  across them.
- **Lip-Sync Ready** — an optional prompt-conditioning mode that favours stable,
  clearly visible faces for downstream lip-sync tooling.

---

## Requirements

- **Python 3.10+**
- **FFmpeg** (both `ffmpeg` and `ffprobe` on your `PATH`)
- A running **ComfyUI** instance (local by default at `http://127.0.0.1:8188`)
- At least one of the supported model families installed in ComfyUI:
  - **LTX-2.3**
  - **Wan 2.2** (TI2V-5B)
  - **FramePack**

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Launch

Start ComfyUI first, then run:

```bash
python app.py
```

Then open <http://localhost:7788> in your browser.

You can also use the convenience launchers (no install required):

- **Linux / macOS:** `./launch.sh`
- **Windows:** double-click `launch.bat`

---

## Desktop icon

To install a proper desktop icon / application-menu entry:

- **Linux / macOS:**
  ```bash
  bash install_desktop_icon.sh
  ```
  On Linux this installs the icon into your XDG icon theme, generates an
  executable `simple-comfort-launch.sh` wrapper, writes a `.desktop` entry that
  points at that wrapper, and (on GNOME) marks the desktop shortcut as trusted.
  On macOS it builds a `Simple-Comfort.app` bundle in `~/Applications`.

- **Windows:**
  ```bat
  install_desktop_icon.bat
  ```

The bundled `simple-comfort.desktop` is a **template**: it contains
`{{APP_DIR}}` placeholders that the installer substitutes with real absolute
paths. Do not use it directly — always run the installer.

---

## Models

| Model | Modes | Strategy |
|-------|-------|----------|
| **LTX-2.3** | T2V, I2V | Two ~15-second sections; section 2 is conditioned on the last frame of section 1. Fast and lightweight. |
| **Wan 2.2** (TI2V-5B) | T2V, I2V | Six 5-second sections rendered natively, each conditioned on the previous section's last frame. Strong prompt adherence and motion. |
| **FramePack** | I2V (T2V if supported by your build) | Progressive single-call generation of the full 30 seconds with built-in anti-drift; I2V always available. |

Model availability is auto-detected from your ComfyUI installation — models that
are missing required checkpoints, VAEs, text encoders or custom nodes are shown
as unavailable in the UI with an explanation.

---

## Configuration

All settings are read from environment variables (or a local `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFY_ENDPOINTS` | `http://127.0.0.1:8188` | Comma-separated list of ComfyUI endpoints. Each is treated as an independent GPU worker. |
| `PORT` | `7788` | Port the web UI binds to (localhost only). |
| `HOST` | `127.0.0.1` | Bind address. |
| `OUTPUT_DIR` | `outputs` | Base directory for generated clips and projects. |
| `MAX_CONCURRENT_JOBS` | `2` | Maximum generations running at once. |
| `COMFY_TIMEOUT` | `300` | Per-section render timeout (seconds). |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |

Example `.env`:

```env
COMFY_ENDPOINTS=http://127.0.0.1:8188,http://127.0.0.1:8189
PORT=7788
OUTPUT_DIR=outputs
```

---

## Governing principle

> **Keep it simple.** Simple-Comfort exists to make one thing effortless:
> generating a good 30-second clip locally. Every feature earns its place by
> serving that goal — no accounts, no cloud, no ceremony.

---

## License

Released under the [MIT License](LICENSE).
