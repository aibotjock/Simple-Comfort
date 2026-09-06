# Model manifest

Human-readable manifest of every model file the generation engines need.
Kept in sync with `tools/check_models.py`, which derives the same list
live from `workflows/*.json` loader nodes and `adapters/*.py` — **update
this file together with the checker** whenever a workflow, adapter or
source entry changes.

Run the checker (read-only, never downloads, never prompts):

```bash
./venv/bin/python tools/check_models.py          # table + verdicts, exit 0/1
./venv/bin/python tools/check_models.py --json   # machine-readable
```

Environment: `COMFY_PATH` (models root, default `/data/comfyui`),
`COMFY_ENDPOINTS` (default `http://127.0.0.1:8188`; when reachable the
live `/object_info` combo lists cross-check each file).

Directories are relative to `$COMFY_PATH/models`. Sizes are approximate
on-disk sizes measured from the reference install. "verify" in the source
column means the org is official but the exact repo slug/path should be
confirmed before downloading — no URL is invented for those.

## LTX 2.5

Workflows: `ltx25_t2v.json`, `ltx25_t2v_2stage.json`, `ltx25_i2v.json`,
`ltx25_i2v_2stage.json`

| File | Directory | Approx size | Source |
|---|---|---|---|
| ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors | diffusion_models | 20.0 GiB | Lightricks/LTX-2.5 (verify) |
| gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors | text_encoders | 14.3 GiB | Lightricks/LTX-2.5 (verify) |
| gemma4_e2b_it_int8_convrot.safetensors (in-graph prompt enhancer) | text_encoders | 4.8 GiB | Lightricks/LTX-2.5 (verify) |
| ltx-2.5-video-vae-bf16.safetensors | vae | 1.4 GiB | Lightricks/LTX-2.5 (verify) |
| ltx-2.5-audio-vae-bf16.safetensors | vae | 348 MiB | Lightricks/LTX-2.5 (verify) |
| ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors (2-stage workflows only) | latent_upscale_models | 950 MiB | Lightricks/LTX-2.5 (verify) |

## Wan 2.2

Workflows: `wan22_t2v.json`, `wan22_i2v.json`. Each mode loads a
high-noise + low-noise expert pair plus its matching lightx2v 4-step LoRA.

| File | Directory | Approx size | Source |
|---|---|---|---|
| wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors | diffusion_models | 13.3 GiB | Comfy-Org/Wan_2.2_ComfyUI_repackaged |
| wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors | diffusion_models | 13.3 GiB | Comfy-Org/Wan_2.2_ComfyUI_repackaged |
| wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors | diffusion_models | 13.3 GiB | Comfy-Org/Wan_2.2_ComfyUI_repackaged |
| wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors | diffusion_models | 13.3 GiB | Comfy-Org/Wan_2.2_ComfyUI_repackaged |
| wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors | loras | 1.1 GiB | lightx2v (verify) |
| wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors | loras | 1.1 GiB | lightx2v (verify) |
| wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors | loras | 1.1 GiB | lightx2v (verify) |
| wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors | loras | 1.1 GiB | lightx2v (verify) |
| umt5_xxl_fp8_e4m3fn_scaled.safetensors | text_encoders | 6.3 GiB | Comfy-Org/Wan_2.2_ComfyUI_repackaged |
| wan_2.1_vae.safetensors | vae | 242 MiB | Comfy-Org/Wan_2.2_ComfyUI_repackaged |

## FramePack

Workflow: `framepack_i2v.json` (image-to-video only). The checkpoint is
loaded by the `LoadFramePackModel` custom node, which reads the
`diffusion_models` folder.

| File | Directory | Approx size | Source |
|---|---|---|---|
| FramePackI2V_HY_fp8_e4m3fn.safetensors | diffusion_models | 15.2 GiB | lllyasviel/FramePackI2V_HY |
| clip_l.safetensors | text_encoders | 235 MiB | comfyanonymous/flux_text_encoders |
| llava_llama3_fp8_scaled.safetensors | text_encoders | 8.5 GiB | lllyasviel FramePack ecosystem (verify) |
| hunyuan_video_vae_bf16.safetensors | vae | 470 MiB | lllyasviel / Kijai HunyuanVideo ecosystem (verify) |
| sigclip_vision_patch14_384.safetensors | clip_vision | 817 MiB | Comfy-Org/Wan_2.2_ComfyUI_repackaged |

## Notes

- Total footprint: ~106 GiB across 21 files.
- Only workflows whose filename prefix maps to an engine registered in
  `adapters/__init__.py:build_adapters()` (ltx25 / wan22 / framepack)
  count as required; any other workflow file in `workflows/` is reported
  as skipped, never as missing.
- Incomplete downloads (`*.part`, `*.incomplete`) never count as present;
  the checker flags them on MISSING rows so an interrupted fetch is
  obvious.
- `tools/check_models.py` exits 0 only when every engine is READY — usable
  as a setup gate in scripts/CI.
