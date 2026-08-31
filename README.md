# Simple-Comfort

> A deliberately simple local WebUI for generating 30-second AI video clips.

Simple-Comfort is a thin, local-only web front end that wraps a running
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance and turns it into
a one-click **30-second AI video generator**. You pick a model, type a prompt
(or drop a reference image), and it handles all the internal sectioning,
frame-to-frame continuity, and stitching to produce a single 30-second clip.

No cloud calls, no API keys, no telemetry — everything runs on your machine.

---

## Quick Start

1. Install **Python 3.10+**.
2. Run `./setup.sh` (Linux/macOS) or `setup.bat` (Windows) — creates a virtual
   environment and installs all dependencies.
3. Install and start **ComfyUI** on port **8188**.
4. Run `./launch.sh` (Linux/macOS), double-click `launch.bat` (Windows), or use
   the desktop icon.
5. Open <http://localhost:7788> in your browser.

> **If you see "Server error" in the UI**, run `setup.sh` (or `setup.bat`) first
> to install the required dependencies, then relaunch.

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

## Dual GPU Setup (RTX 3090 / Multi-GPU)

If you have two (or more) GPUs, Simple-Comfort can run one ComfyUI instance per
GPU and **alternate jobs between them (round-robin)** — job 1 → GPU 0, job 2 →
GPU 1, job 3 → GPU 0, and so on. This roughly doubles throughput when you queue
multiple clips.

### Quick start

Instead of `python3 app.py`, run:

```bash
./launch-dual-gpu.sh
```

This script:

1. **Checks whether ComfyUI is already running** on ports 8188 and 8189
   (`curl .../system_stats`). **If an instance is already running on a port, it
   is left alone — Simple-Comfort connects to it and does NOT restart it.**
2. Cold-starts **only** the instance(s) that are not already responding
   (`CUDA_VISIBLE_DEVICES=0` on port 8188, `CUDA_VISIBLE_DEVICES=1` on port 8189).
3. Waits for both instances to respond.
4. Writes `COMFY_ENDPOINTS=http://127.0.0.1:8188,http://127.0.0.1:8189` to `.env`.
5. Launches Simple-Comfort, which automatically load-balances across both GPUs.

> **Already running ComfyUI?** If you already have ComfyUI up on both ports
> (e.g. started by your own launcher), Simple-Comfort connects to them
> automatically — it will **not** start or kill your existing ComfyUI session.
> In that case you can also just copy `.env.example` to `.env` (which already
> lists both dual-GPU endpoints) and run `./launch.sh` or the desktop icon.

When both ComfyUI instances are up, the header shows a **status chip per GPU**
(`GPU 0`, `GPU 1`) with a colored dot — green = idle, yellow = busy, red =
offline — and a `✓N` badge counting completed jobs on each GPU.

On exit (Ctrl-C), only the ComfyUI instances that **this script started** are
stopped — an already-running session you started yourself is left untouched.

### Single GPU

For a single GPU, use:

```bash
./launch-single-gpu.sh
```

This starts one ComfyUI instance on GPU 0 (port 8188), writes
`COMFY_ENDPOINTS=http://127.0.0.1:8188` to `.env`, and launches Simple-Comfort.

### ComfyUI location

The launchers look for ComfyUI in `~/ComfyUI`, `~/comfyui`, `/opt/ComfyUI`, and
`~/AI/ComfyUI`. If yours is elsewhere, point them at it:

```bash
COMFYUI_PATH=/path/to/ComfyUI ./launch-dual-gpu.sh
```

### Manual configuration

You can skip the launch scripts and configure endpoints yourself. Start your
ComfyUI instances however you like, then set in `.env`:

```ini
COMFY_ENDPOINTS=http://127.0.0.1:8188,http://127.0.0.1:8189
```

Simple-Comfort treats each endpoint as an independent GPU worker and alternates
jobs across all alive workers.

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
