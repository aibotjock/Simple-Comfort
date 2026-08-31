# Build Prompt — Simple-Comfort

You are a senior Python engineer, ComfyUI integration engineer, and local generative-video systems architect. Build a production-ready but deliberately simple local WebUI named **Simple-Comfort**.

The governing principle is: **simplicity is superior to feature count**. Do not recreate ComfyUI. Do not expose a node graph. Do not add unrelated tools. Hide model-specific workflow complexity behind a minimal interface.

## Objective

Build a local WebUI that sits on top of an existing ComfyUI installation and provides exactly three video-generation model families:

1. **LTX-2.3**
2. **Wan 2.2**
3. **FramePack**

The application is only for generating video clips. It must support:

- Text-to-video
- Image-to-video
- A target finished clip length of **30 seconds**
- Local generation only
- A simple continuity system so one 30-second clip can naturally continue into the next
- Output designed to be clean input for a separate future lip-sync WebUI

Do **not** build lip syncing here. Do **not** build a timeline editor here. Do **not** build long-form episode assembly here. Those will be separate applications.

## Hard constraints

- No cloud generation APIs.
- No paid partner nodes.
- No remote inference fallbacks.
- Do not require accounts, API keys, subscriptions, or hosted services.
- The normal runtime must work entirely on localhost.
- Use ComfyUI as the local inference backend wherever practical.
- Do not modify ComfyUI core.
- Communicate with ComfyUI through its local HTTP/WebSocket API and stored workflow JSON.
- Keep model-specific logic in adapters so the WebUI remains simple.
- Only the three video model families above may appear in the main model selector.
- Never silently download very large model files. Detect missing files and show a clear setup command or explicit install action.
- Preserve reproducibility: save model, seed, prompt, settings, workflow version, and generation metadata with every output.

## Technology choice

Prefer a Python-only stack to minimize complexity:

- FastAPI backend
- Uvicorn
- Vanilla HTML/CSS/JavaScript frontend served by FastAPI
- Pydantic models
- httpx / websockets for ComfyUI communication
- FFmpeg for concatenating internal subsegments into one final 30-second clip and for extracting first/last frames

Avoid React, Node, Electron, databases, Docker, Redis, Celery, and other infrastructure unless absolutely necessary. A local JSON project store is sufficient.

The application should start with one command such as:

```bash
python app.py
```

and open at a localhost address such as:

```text
http://127.0.0.1:8288
```

Assume ComfyUI is normally available at:

```text
http://127.0.0.1:8188
```

Make this configurable through `.env` or a small settings file.

## User experience

The main page should have only the controls needed to generate a clip.

### Main controls

**Mode**
- Text to Video
- Image to Video

**Model**
- LTX-2.3
- Wan 2.2
- FramePack

**Prompt**
- Large text box

**Reference image**
- Visible only in Image-to-Video mode
- Drag/drop or file picker

**Duration**
- Default and primary target: `30 seconds`
- Keep 30 seconds selected by default
- Shorter diagnostic options may be available under Advanced, but do not clutter the main UI

**Aspect ratio**
- 16:9
- 9:16
- 1:1

Default: 16:9

**Quality preset**
- Draft
- Balanced
- Quality

Default: Balanced

**Lip-Sync Ready**
- Toggle, default ON

**Continuity**
- New Clip
- Continue Previous Clip

**Generate 30s Clip**
- One primary action button

### Status area

Show:
- Selected model
- GPU worker being used
- Queue state
- Current internal segment number
- Progress percentage where available
- Elapsed time
- Preview when available
- Final 30-second video
- Open output folder
- Continue From This Clip button

Do not expose ComfyUI node names in normal mode.

## 30-second generation strategy

A critical requirement: the UI should present **one 30-second job**, while each model adapter may use the safest internal strategy for that model.

Do not falsely claim every model produces a perfect 30-second native pass.

### LTX-2.3 adapter

Use local ComfyUI-native LTX-2.3 workflows for T2V and I2V.

Default target strategy:

- Generate approximately 15 seconds
- Extract the last clean frame
- Use that frame plus continuity metadata as conditioning for the second approximately 15-second generation
- Concatenate the two internal sections into one 30-second MP4 with FFmpeg

The user should see one job, not two unrelated jobs.

Support the model's useful native capabilities when they improve quality without complicating the main UI:

- Text-to-video
- Image-to-video
- Strong portrait support
- First/last-frame or continuation conditioning when useful
- Model-supported camera/control conditioning when installed
- Spatial upscaling when selected by a quality preset
- Seed reuse

Do not enable LTX audio generation in this app. Final output should be silent because audio and lip sync are handled elsewhere.

### Wan 2.2 adapter

Prefer the **Wan 2.2 TI2V 5B hybrid** where it is the best fit because it supports both text-to-video and image-to-video and is suitable for consumer-class GPUs.

Default 30-second strategy:

- Generate short stable sections, initially 5 seconds each unless testing proves a longer segment is reliable
- Use the last frame of each section as the image condition for the next section
- Preserve the same prompt/style/seed strategy across sections
- Produce six approximately 5-second sections for a 30-second final clip
- Concatenate automatically with FFmpeg

If a stable installed Wan workflow supports longer reliable sections on the current hardware, the adapter may reduce the number of joins, but correctness and continuity are more important than minimizing section count.

Expose advanced cinematic controls only inside the collapsed Advanced panel, such as:

- motion strength where supported
- guidance
- steps
- seed
- camera language/prompt hints
- resolution
- frame count

Do not expose raw workflow nodes.

### FramePack adapter

FramePack is the long-duration specialist.

Use its progressive next-frame-section generation approach for the full 30-second target when available and stable.

Support:

- Image-to-video through a starting reference image
- Text-to-video when the installed FramePack/P1 implementation supports it
- progressive long-video generation
- anti-drift/history controls when available
- section-level progress preview

If the installed FramePack implementation only supports image-conditioned generation reliably, do not fake text-to-video support. Instead, for Text-to-Video mode either:

1. use a verified local FramePack text-to-video implementation, or
2. clearly mark that model/mode combination unavailable until the required local workflow is installed.

Never silently route a FramePack request to another model while still labeling it FramePack.

## Model capability detection

At startup, detect the actual local environment.

For each model adapter report:

- Available / missing
- Required checkpoint names
- Required text encoders / VAEs / LoRAs
- Required custom nodes, if any
- Supported local modes: T2V, I2V
- Maximum tested internal section duration configured for this installation

The UI should gracefully disable unsupported combinations instead of failing after generation starts.

Create a simple `Model Status` modal or page.

## Lip-Sync Ready mode

This app does not lip sync.

The goal is to generate footage that a later lip-sync system can modify successfully.

When **Lip-Sync Ready** is ON, automatically add model-appropriate conditioning/prompt guidance that favors:

- stable facial identity
- face remaining visible
- unobstructed mouth
- no hands crossing the mouth
- limited extreme head rotation
- limited rapid camera movement during speaking-style shots
- stable jaw and facial geometry
- natural blinking
- realistic facial micro-motion
- no exaggerated random mouth movement
- no generated dialogue
- no burned-in subtitles
- no microphone covering the mouth
- consistent lighting across the face
- reasonable framing for a future face/lip-sync pass

Prefer a relaxed neutral or lightly parted mouth rather than random speech-like articulation.

Do not promise accurate spoken phonemes without audio. The later lip-sync application will create the final mouth articulation.

Export at a constant frame rate appropriate for downstream processing. Store FPS in metadata.

## Continuity framework

Continuity should be simple but deliberately designed from the beginning.

Every generated 30-second clip must save:

- project ID
- clip ID
- parent clip ID, if continued
- model
- generation mode
- full user prompt
- resolved model prompt
- negative prompt if used
- seed
- aspect ratio
- width / height
- FPS
- internal section settings
- model/checkpoint filenames
- workflow version
- reference image path/hash if used
- first-frame image
- last-frame image
- thumbnail
- final video path
- generation timestamp

Store this in a sidecar JSON file next to the clip.

### Continue Previous Clip

When the user clicks **Continue From This Clip**:

1. Load the previous clip's last frame as the next clip's starting reference.
2. Carry forward its model unless the user explicitly changes it.
3. Carry forward aspect ratio, resolution, FPS, style settings, seed strategy, and Lip-Sync Ready state.
4. Keep the prior prompt visible and editable.
5. Add a small optional `What changes next?` field.
6. Build the next prompt from the original continuity context plus the new action/change.

The user must be able to create:

```text
Clip 001 -> Clip 002 -> Clip 003 -> Clip 004
```

without manually extracting or re-uploading frames.

Do not attempt to merge those separate 30-second clips into a long movie. That belongs in the separate future editor WebUI.

## Prompt handling

Do not use a cloud LLM to rewrite prompts.

Implement a small deterministic local prompt composer that combines:

- user prompt
- model-specific recommended syntax
- Lip-Sync Ready guidance
- continuity guidance
- optional camera instruction

Keep the user's wording dominant. Do not transform a concise prompt into several paragraphs unnecessarily.

Provide a small `Show resolved prompt` link so advanced users can inspect what was actually submitted.

## Advanced panel

Keep it collapsed by default.

Only include settings that materially matter:

- Seed: random / locked / explicit
- Resolution
- FPS
- Internal segment duration
- Steps
- Guidance / CFG when supported
- Motion strength when supported
- Camera behavior when supported
- Continuity strength when supported
- Model-specific anti-drift option
- Optional upscale
- ComfyUI worker/GPU endpoint

Each adapter should translate these generic controls into model-specific workflow parameters.

Unsupported controls should disappear for that model rather than being shown disabled everywhere.

## Multi-GPU support

Keep this simple.

Support one or more local ComfyUI endpoints through configuration, for example:

```env
COMFY_ENDPOINTS=http://127.0.0.1:8188,http://127.0.0.1:8189
```

If multiple endpoints exist:

- Treat them as independent workers
- Queue whole internal generation sections to an available worker
- Never assume VRAM is pooled
- Do not split a single diffusion step across GPUs unless the underlying workflow explicitly supports it
- Allow Auto, GPU 0, GPU 1 selection in Advanced

If only one endpoint exists, behave normally with no extra UI clutter.

## ComfyUI integration

Create a clean abstraction:

```text
ModelAdapter
  generate_text_to_video(...)
  generate_image_to_video(...)
  continue_segment(...)
  validate_environment(...)
  capabilities()
```

Implement:

```text
LTX23Adapter
Wan22Adapter
FramePackAdapter
```

Each adapter owns its workflow JSON templates and parameter mapping.

The rest of the application must not know ComfyUI node IDs.

Create a local ComfyUI client that supports:

- `/prompt` submission
- queue monitoring
- WebSocket progress if available
- history/output retrieval
- interrupt/cancel
- worker health check

Fail cleanly if ComfyUI is not running and tell the user exactly how to fix it.

## Output layout

Use a predictable structure such as:

```text
outputs/
  projects/
    <project-id>/
      clip-001/
        final.mp4
        metadata.json
        first_frame.png
        last_frame.png
        thumbnail.jpg
        sections/
          section-001.mp4
          section-002.mp4
      clip-002/
        ...
```

Internal sections may be retained or deleted through a setting. Default to retain them until the final video is verified.

## Model installation policy

Create a command:

```bash
python tools/check_models.py
```

It should print a clear table showing what is present and what is missing.

Optionally create:

```bash
python tools/setup_models.py --model ltx23
python tools/setup_models.py --model wan22
python tools/setup_models.py --model framepack
```

Before downloading anything large, show:

- exact files
- source repository
- approximate disk requirement when known
- destination directories

Require explicit confirmation.

Do not invent model URLs. Use official or well-established sources and keep them centralized in a manifest so they can be updated later.

## Safety and reliability

- Never execute arbitrary shell text received from the browser.
- Validate uploaded images.
- Restrict file access to configured project/model directories.
- Sanitize filenames.
- Bind to localhost by default.
- Do not expose ComfyUI to the network automatically.
- Do not send telemetry.
- Do not make internet requests during normal generation.
- Log errors clearly but do not expose secrets or unrelated local paths in the browser.

## Repository structure

Prefer something close to:

```text
Simple-Comfort/
  app.py
  requirements.txt
  .env.example
  README.md
  simple_comfort/
    config.py
    models.py
    projects.py
    continuity.py
    prompts.py
    ffmpeg.py
    comfy_client.py
    workers.py
    adapters/
      base.py
      ltx23.py
      wan22.py
      framepack.py
    workflows/
      ltx23/
      wan22/
      framepack/
    web/
      index.html
      app.js
      styles.css
  tools/
    check_models.py
    setup_models.py
  tests/
```

Do not create dozens of tiny files merely for architectural purity. Consolidate where it improves readability.

## Implementation order

Build in this order and keep every stage runnable:

1. Minimal FastAPI server and single-page UI.
2. ComfyUI health check and connection.
3. Project/output storage.
4. LTX-2.3 T2V and I2V adapter.
5. LTX 30-second continuation + FFmpeg assembly.
6. Wan 2.2 T2V and I2V adapter.
7. Wan 30-second segmented continuation.
8. FramePack integration.
9. Lip-Sync Ready prompt mode.
10. Continue Previous Clip workflow.
11. Multi-worker support.
12. Model-status/setup tooling.
13. Tests and documentation.

Do not begin with authentication, databases, packaging systems, themes, or plugin ecosystems.

## Acceptance criteria

The build is complete when all of the following are true:

1. `python app.py` launches the WebUI locally.
2. The UI has only LTX-2.3, Wan 2.2, and FramePack as generation-model choices.
3. The user can generate from text.
4. The user can generate from an image wherever the selected model locally supports it.
5. The main action targets one finished 30-second video.
6. LTX can transparently create the 30-second result through continuation when needed.
7. Wan can transparently create the 30-second result through shorter chained sections when needed.
8. FramePack can progressively generate the target duration through a verified local implementation.
9. Final video is silent and ready for downstream audio/lip-sync processing.
10. Lip-Sync Ready mode improves facial/mouth stability but makes no false claim of phoneme synchronization.
11. Clicking Continue From This Clip automatically carries the last frame and continuity metadata into the next 30-second generation.
12. Every clip has reproducibility metadata and saved first/last frames.
13. Missing models/workflows are detected before generation.
14. There are no paid/cloud calls in the generation path.
15. The application remains understandable to a user who never opens the underlying ComfyUI graph.

## Non-goals

Do not add any of the following to this repository:

- voice generation
- speech synthesis
- audio mixing
- lip-sync inference
- subtitles
- timeline editing
- transitions between separate 30-second clips
- long-form episode assembly
- YouTube upload
- research/news gathering
- script generation
- image generation unrelated to video conditioning
- generic ComfyUI workflow editing
- cloud model support

## Final build behavior

The ideal user experience is:

```text
Choose: LTX-2.3 | Wan 2.2 | FramePack
Choose: Text-to-Video | Image-to-Video
Enter prompt
Optionally upload image
Lip-Sync Ready: ON
Continuity: New Clip

[ Generate 30s Clip ]
```

After generation:

```text
[ Play ] [ Download/Open Folder ] [ Continue From This Clip ]
```

That is the product.

Before declaring completion, run the application, test the health-check path, validate the workflow parameter mappings, test generation with short diagnostic durations before a 30-second test, confirm FFmpeg assembly, test continuity metadata, and verify that no cloud/API route is called during generation.
