# Simple-Comfort

> Local, one-click 30-second AI video — LTX 2.5, Wan 2.2 and FramePack behind one deliberately simple web UI.

Simple-Comfort is a local-only FastAPI app that drives a running
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance. Pick an engine,
type a prompt (or drop a reference image), and it handles sectioning,
last-frame continuity, FFmpeg stitching and per-GPU scheduling to deliver one
finished clip. No cloud calls, no API keys, no telemetry.

## Quick start

```bash
./setup.sh      # 1. create venv + install dependencies (once)
#                2. start ComfyUI (default http://127.0.0.1:8188)
./launch.sh     # 3. start Simple-Comfort
#                4. open http://localhost:7788
```

Requires Python 3.10+, FFmpeg (`ffmpeg` + `ffprobe`) on PATH, and the model
files listed below. Windows equivalents: `setup.bat`, `launch.bat`.

## Engines

| Engine | Modes | How it renders |
|--------|-------|----------------|
| **LTX 2.5** | T2V, I2V | 22B int8 distilled transformer. ~6 s sections chained on each section's last frame (24 fps). `draft` = single-stage, 8 steps; `balanced`/`quality` = two-stage, 8+3 steps decoding at 2x resolution. Joint audio/video model — every section ships generated audio. The Gemma prompt enhancer runs **inside the graph**, so every prompt is rewritten locally. |
| **Wan 2.2** | T2V, I2V | 14B fp8 dual-expert (high-noise + low-noise pairs for both t2v and i2v). Six 5 s sections at 16 fps. `draft`/`balanced` use the official lightx2v 4-step distilled config; `quality` switches to the standard 20-step path. |
| **FramePack** | I2V only | Hunyuan fp8. One progressive call renders the whole clip (30 fps) with built-in anti-drift. T2V is honestly disabled in the UI — the sampler requires a start image. |

Engine availability is auto-detected from your ComfyUI (nodes + model
folders); anything missing is shown in the UI with the exact reason.

## Model files

Exact filenames the built-in workflows load, with approximate on-disk sizes.
Place each in the matching ComfyUI models subfolder. Verify your install with
`./venv/bin/python tools/check_models.py` (reads `COMFY_PATH`/`COMFY_ENDPOINTS`
from the environment or the repo `.env`; exits non-zero if anything is missing —
see `tools/MODELS.md` for sources).

| File | Size | Used by |
|------|------|---------|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 20 GB | LTX 2.5 transformer (int8) |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 14 GB | LTX 2.5 text encoder |
| `gemma4_e2b_it_int8_convrot.safetensors` | 5 GB | LTX 2.5 in-graph prompt enhancer |
| `ltx-2.5-video-vae-bf16.safetensors` | 1.4 GB | LTX 2.5 video VAE |
| `ltx-2.5-audio-vae-bf16.safetensors` | 0.3 GB | LTX 2.5 audio VAE |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 0.9 GB | LTX 2.5 two-stage upscaler |
| `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | 13 GB | Wan t2v high-noise expert |
| `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | 13 GB | Wan t2v low-noise expert |
| `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | 13 GB | Wan i2v high-noise expert |
| `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | 13 GB | Wan i2v low-noise expert |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6 GB | Wan text encoder |
| `wan_2.1_vae.safetensors` | 0.2 GB | Wan VAE |
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` | 1.1 GB | Wan t2v 4-step LoRA |
| `wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors` | 1.1 GB | Wan t2v 4-step LoRA |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | 1.1 GB | Wan i2v 4-step LoRA |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | 1.1 GB | Wan i2v 4-step LoRA |
| `FramePackI2V_HY_fp8_e4m3fn.safetensors` | 15 GB | FramePack sampler model |
| `llava_llama3_fp8_scaled.safetensors` | 8.5 GB | FramePack LLaVA text encoder |
| `clip_l.safetensors` | 0.2 GB | FramePack dual-CLIP (with LLaVA) |
| `hunyuan_video_vae_bf16.safetensors` | 0.5 GB | FramePack VAE |
| `sigclip_vision_patch14_384.safetensors` | 0.8 GB | FramePack CLIP vision |

## Custom workflows

Drop any ComfyUI workflow JSON onto the **Workflows** panel:

- **API-format** exports (`Save (API format)`) import as-is.
- **Editor/UI-format** exports are converted to API format automatically
  (ComfyUI must be running so node definitions can be resolved).
- **Subgraph** exports are rejected with guidance to re-export flat via
  `Save (API format)` — subgraphs cannot be flattened.

A form is generated from the graph's own inputs (prompts, seeds, resolutions,
model pickers) and the workflow becomes a selectable engine. One example ships
in `custom_workflows/`: a Wan t2v lightx2v 4-step graph.

## Panels

- **Avatars** — three reusable character slots: portrait, thumbnail, notes,
  linked voice.
- **Voices** — audio samples (wav/mp3/m4a) linkable to avatars; attach one to
  a clip as its narration track.
- **B-roll** — 8–10 s supporting shots from fixed categories (cinematic,
  explanatory, product, environment, technology, abstract, transition);
  deterministic prompts.
- **Workflows** — custom workflow import (above).
- **Sequence** — per-project timeline: reorder clips, b-roll and bridge clips.
- **Export** — zip a project as video-only, a lip-sync handoff package
  (MP4s + last frames + narration + manifest) or a full project archive.

**Bridge clips**: from a finished clip's action row, generate a short I2V
clip conditioned on that clip's final frame, then insert it in the sequence
for a smooth transition to the next clip.

## Dual GPU & launchers

| Script | What it does |
|--------|--------------|
| `setup.sh` | Create venv + install dependencies (once). |
| `launch.sh` | Start the app (ComfyUI assumed already running). |
| `launch-single-gpu.sh` | Cold-start ComfyUI on GPU 0 (port 8188), point the app at it. |
| `launch-dual-gpu.sh` | One ComfyUI per GPU (ports 8188/8189), skipping any instance already running, then launch. |
| `check-gpu.sh` | GPU doctor: verifies the driver, both cards, per-index CUDA visibility, ComfyUI's torch build and the ComfyUI ports — prints the exact fix for anything wrong. Run it before `launch-dual-gpu.sh`. |
| `install.sh` | Standalone installer: dependencies, GPU auto-detection, `.env`, desktop icon. |

With two endpoints, jobs alternate round-robin across GPUs and the header
shows a status chip per GPU — green idle, yellow busy, red offline — with a
completed-jobs badge. The launchers only ever stop ComfyUI instances they
started themselves.

**Even 50/50 split:** dual-GPU mode runs **one ComfyUI per card** (GPU 0 →
8188, GPU 1 → 8189, each pinned with `CUDA_VISIBLE_DEVICES`) and alternates
jobs round-robin, so each RTX 3090 does ~half the work with the full 24 GB
available to every model. Confirm it live with `nvidia-smi -l 1`, or read the
per-card memory/utilization now included in `GET /api/status` (`"gpus"`).

**Full driver / CUDA setup** — including dual RTX 3090 on Ubuntu 26, fixing a
CPU-only torch, and running LM Studio / other LLMs alongside (time-share,
per-card pinning, or a `50,50` tensor split) — is in
[GPU_SETUP.md](GPU_SETUP.md). GPU/CUDA packages for the ComfyUI/LLM environment
(torch+cuda, llama-cpp-python with CUDA, transformers/accelerate) are in
[requirements-gpu.txt](requirements-gpu.txt) — install those into ComfyUI's
venv, **not** the app's.

## Configuration

All settings come from environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFY_ENDPOINTS` | `http://127.0.0.1:8188` | Comma-separated ComfyUI endpoints; each is one GPU worker. |
| `PORT` | `7788` | Web UI port. |
| `HOST` | `127.0.0.1` | Bind address. |
| `OUTPUT_DIR` | `outputs` | Base directory for clips, projects and uploads. |
| `MAX_CONCURRENT_JOBS` | `2` | Generations running at once. |
| `COMFY_TIMEOUT` | `300` | Per-section render timeout (seconds). |
| `MODEL_CACHE_TTL` | `60` | Seconds to cache model-detection results. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |

## Current limits

- **FramePack is I2V-only** — the UI disables t2v for it; the sampler's start
  image is required.
- **LTX 2.5 two-stage wants VRAM headroom** — it decodes at 2x resolution. On
  a 24 GB card, close other VRAM consumers or use `draft` (single-stage).
- **Unload resident model servers (e.g. LM Studio) before rendering** — these
  engines need most of a 24 GB card each.
- **Custom workflows whose defaults reference missing models fail ComfyUI
  validation at queue time** — the job error names the offending node; pick an
  installed file in the form's dropdown.
- **LTX 2.5 ignores steps/guidance overrides** — the distilled sigma schedule
  fixes them; effective values are reported truthfully in job metadata.

## License

Released under the [MIT License](LICENSE).
