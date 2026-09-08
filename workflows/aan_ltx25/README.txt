APPLIED AI NEWS — LTX-2.5 10-WORKFLOW CANDIDATE PACK
======================================================

STATUS
------
These files are STRUCTURALLY VALIDATED CANDIDATES, not Golden workflows.
They were generated from current official LTX-2.5 / ComfyUI node patterns and
checked for JSON integrity, link integrity, required model references, one-prompt
UI, prompt enhancer presence, image inputs, optional audio conditioning, and
required LTX node paths.

They have NOT been executed on your local ComfyUI installation. It would be
false to claim runtime validation without access to the target machine.

IMPORTANT FIRST TEST
--------------------
Load workflow 08 FAST_PREVIEW first and test at 3–5 seconds before changing it
to 20 seconds. Only after a workflow renders successfully on your machine should
it be promoted to Golden.

COMMON INTERFACE
----------------
Every workflow has:
- SCRIPT / VIDEO REQUEST — one creative prompt only
- Prompt Enhance toggle (default ON)
- AVATAR / CHARACTER drag-and-drop image
- SCENE / B-ROLL IMAGE drag-and-drop image
- Use Image Conditioning toggle
- Reference Source Avatar/Scene toggle
- MASTER AUDIO drag-and-drop
- Use Uploaded Master Audio toggle
- Duration (default 20 s)
- FPS (default 24)
- Working width / height
- Reference / Avatar strength
- Audio Start
- Seed control

AUDIO BEHAVIOR
--------------
When Use Uploaded Master Audio = ON:
- audio is trimmed to Audio Start + Duration
- encoded with the LTX audio VAE
- assigned a zero noise mask so the audio latent is frozen during diffusion
- used in the joint A/V latent
- the ORIGINAL trimmed waveform is muxed into the final video

When OFF:
- the workflow creates an empty LTX audio latent
- LTX generates audio jointly with the video
- generated audio latent is decoded for final muxing

This design is based on the official LTX-2.5 A2V pattern. It does NOT claim
perfect speech lip-sync. Audio conditioning and lip-sync are different claims.

IMAGE / AVATAR BEHAVIOR
-----------------------
Only one image is used as the primary I2V conditioning source per render:
- Avatar toggle ON -> AVATAR / CHARACTER image
- Avatar toggle OFF -> SCENE / B-ROLL IMAGE

Both upload nodes are present so Simple Comfort can use the same UI contract.
Reference strength is an I2V conditioning control; it is not a guarantee of
perfect identity preservation.

20-SECOND FRAME COUNT
---------------------
The graph computes:
  1 + floor(FPS * duration / 8) * 8

At 24 fps and 20 s -> 481 frames.
At 30 fps and 20 s -> 601 frames.

FILES
-----
01 AUDIO_FIRST_ANCHOR
02 CHARACTER_LOCK_ANCHOR
03 STUDIO_ANCHOR
04 DYNAMIC_ANCHOR
05 T2V_BROLL
06 I2V_BROLL
07 AUDIO_TIMED_BROLL
08 FAST_PREVIEW
09 TWO_STAGE_PRODUCTION
10 MASTER_PIXEL_SPATIAL

WORKFLOW 09
-----------
Uses LTX-2.5 latent spatial x2 upscaling followed by the short official-style
high-resolution refinement schedule.

WORKFLOW 10
-----------
Uses a separate Pixel Spatial IC-LoRA x2 V2V finishing pass. This is the most
experimental/heaviest file in the pack and MUST be runtime-tested locally before
production use. Pixel Spatial is generative enhancement and can synthesize
details; factual graphics should not be passed through it when exact fidelity
matters.

FIXED MODEL FILENAMES
---------------------
Transformer:
  ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors

Text encoder:
  gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors

Prompt enhancer:
  gemma4_e2b_it_int8_convrot.safetensors

Video VAE:
  ltx-2.5-video-vae-bf16.safetensors

Audio VAE:
  ltx-2.5-audio-vae-bf16.safetensors

Latent spatial upscaler (workflow 09):
  ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors

Pixel Spatial IC-LoRA (workflow 10):
  ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors
