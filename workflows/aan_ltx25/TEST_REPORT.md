# Applied AI News LTX-2.5 Candidate Workflow Test Report

## Critical status

**Runtime status: NOT TESTED ON THE TARGET COMFYUI INSTALLATION.**

The files passed local static/structural checks only. The target machine must load and execute them before any file is called Golden or production-ready.

Static checks performed for every JSON:
- JSON serialization / parseability
- unique node IDs
- unique link IDs
- every link origin/target exists
- every link socket index is in range
- node input links agree with link table
- node output link lists agree with link table
- no LTX-2.3 model references
- one visible creative prompt (no visible `Prompt (positive)` / `Prompt (negative)` nodes)
- local Gemma prompt enhancer present
- avatar and scene `LoadImage` nodes present
- `LoadAudio` present
- uploaded-audio encode + zero-noise freeze path present
- generated-audio fallback latent path present
- LTX-2.5 Distilled 8-step sigma schedule present
- INT8 ConvRot transformer / encoder references present

| # | Workflow | Static | ComfyUI Load | 3–5s Render | 20s Render | Avatar | Audio | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | AUDIO_FIRST_ANCHOR | PASS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | Single-stage |
| 2 | CHARACTER_LOCK_ANCHOR | PASS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | Single-stage |
| 3 | STUDIO_ANCHOR | PASS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | Single-stage |
| 4 | DYNAMIC_ANCHOR | PASS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | Single-stage |
| 5 | T2V_BROLL | PASS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | Single-stage |
| 6 | I2V_BROLL | PASS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | Single-stage |
| 7 | AUDIO_TIMED_BROLL | PASS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | Single-stage |
| 8 | FAST_PREVIEW | PASS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | Single-stage |
| 9 | TWO_STAGE_PRODUCTION | PASS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | Two-stage |
| 10 | MASTER_PIXEL_SPATIAL | PASS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | Pixel Spatial V2V finish |

## Required local promotion sequence

1. Start with workflow 08 and set duration to 3–5 seconds.
2. Load in the exact local ComfyUI installation.
3. Resolve any missing-node/schema/version errors before rendering.
4. Test text-only with `Use Image Conditioning = OFF` and `Use Uploaded Master Audio = OFF`.
5. Test avatar I2V.
6. Test uploaded narration with `Use Uploaded Master Audio = ON`.
7. Test Prompt Enhance ON and inspect the Effective Prompt.
8. Test fixed seed reproducibility.
9. Increase to 20 seconds at 24 fps.
10. Record peak VRAM, elapsed time, output duration, and any quality/identity problems.
11. Promote only passing files to Golden.

Workflow 09 and especially 10 should be tested after the single-stage workflows because their refinement stages are materially heavier.