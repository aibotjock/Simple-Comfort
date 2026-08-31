# Simple-Comfort

A deliberately simple local video-generation WebUI that sits on top of ComfyUI.

## Scope

Simple-Comfort is only a video generator. It exposes exactly three local video-generation families:

- LTX-2.3
- Wan 2.2
- FramePack

It supports:

- Text-to-video
- Image-to-video
- A target finished clip length of 30 seconds
- Continuity handoff between clips
- Lip-sync-ready generation for a separate downstream lip-sync tool
- Local-only generation through ComfyUI / local model runners

It intentionally does **not** include:

- Audio generation
- Lip syncing
- Timeline editing
- Multi-clip episode assembly
- Cloud/API generation
- General-purpose ComfyUI node editing

The complete implementation prompt is in [`BUILD_PROMPT.md`](BUILD_PROMPT.md).

## Design principle

Hide workflow complexity. The user should choose a model, enter a prompt or image, and press **Generate 30s Clip**. Model-specific workflow logic belongs behind the interface.
