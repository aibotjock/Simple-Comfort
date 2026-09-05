"""FramePack adapter (kijai ComfyUI-FramePackWrapper).

FramePack renders a whole clip progressively inside a single workflow call:
the sampler walks 9-latent (36-pixel-frame) windows anchored on the start
image, so this engine always uses ONE section and lets the model handle
internal sectioning itself (``total_second_length`` drives it).

Only image-to-video is supported — the sampler's ``start_latent`` is a
required input, so a reference image is mandatory.

The graph follows the official ``framepack_hv_example`` bar workflow with
environment-justified substitutions:

* core ``ImageScale`` replaces essentials ``ImageResize+`` (not installed)
  for the bucket resize driven by ``FramePackFindNearestBucket``;
* core ``CreateVideo`` -> ``SaveVideo`` replaces ``VHS_VideoCombine``;
* the negative conditioning is ``ConditioningZeroOut`` of the positive — the
  official distilled model runs at cfg 1.0, where the negative prompt is
  zeroed by design.

FramePack generates natively at 30fps, so the reported fps is fixed at 30
regardless of the request. Actual pixel dimensions come from the model's
resolution bucket — ``resolve_resolution`` values are advisory only.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Any, Dict, List, Optional

from .base import (ModelAdapter, GenerationParams, SectionConfig,
                   DEFAULT_NEGATIVE)

# Sampler window: latent_window_size * 4 pixel frames per internal section.
LATENT_WINDOW_SIZE = 9
PIXEL_FRAMES_PER_SECTION = LATENT_WINDOW_SIZE * 4  # 36

# gpu_memory_preservation node bounds (anti-drift headroom control).
GPU_MEMORY_MIN = 0.0
GPU_MEMORY_MAX = 128.0
# total_second_length node bound (FramePackSampler).
MAX_TOTAL_SECONDS = 120.0


class FramePackAdapter(ModelAdapter):
    MODEL_NAME = "FramePack"
    MODEL_KEY = "framepack"
    SUPPORTED_MODES = {"i2v"}  # start_latent is required: no text-to-video
    TARGET_SECONDS = 30
    NATIVE_FPS = 30  # the sampler derives sections from seconds * 30

    REQUIRED_NODES = ["LoadFramePackModel", "FramePackSampler",
                      "FramePackFindNearestBucket"]
    CHECKPOINT_HINTS = ["framepack"]
    VAE_HINTS = ["hunyuan_video"]
    TEXT_ENCODER_HINTS = ["llava"]

    I2V_TEMPLATE = "framepack_i2v.json"

    # FramePack quality tuning (steps, guidance_scale); cfg stays 1.0 in-graph
    # because the model is distilled (guidance_scale is the effective knob).
    QUALITY = {
        "draft": {"steps": 20, "guidance": 10.0},
        "balanced": {"steps": 30, "guidance": 10.0},  # official default
        "quality": {"steps": 40, "guidance": 10.0},
    }

    # Official FramePack workflows use an empty (zeroed) negative at cfg 1.0.
    DEFAULT_NEGATIVE_PROMPT = ""
    # Anti-drift default: GPU headroom preserved while sampling (GB).
    DEFAULT_GPU_MEMORY_PRESERVATION = 6.0

    def get_section_strategy(self, params: GenerationParams) -> List[SectionConfig]:
        """FramePack renders the whole clip in a single call.

        The sampler derives its internal sections from
        ``total_second_length``: ``round(seconds * 30 / 36)`` windows of 36
        frames plus the start frame, i.e. a 30s target yields 901 frames
        (verified live: a 1-section render produced exactly 37 frames).
        """
        target = params.effective_target_seconds(self.TARGET_SECONDS)
        # FramePackSampler.total_second_length node bound.
        target = min(target, MAX_TOTAL_SECONDS)
        frames = int(round(target * self.NATIVE_FPS)) + 1
        return [SectionConfig(
            index=0,
            frames=frames,
            fps=self.NATIVE_FPS,
            is_first=True,
            use_reference=True,
            approx_seconds=target,
        )]

    def _quality(self, params: GenerationParams) -> Dict[str, float]:
        preset = self.QUALITY.get(params.quality, self.QUALITY["balanced"]).copy()
        if params.steps is not None:
            preset["steps"] = params.steps
        if params.guidance is not None:
            preset["guidance"] = params.guidance
        return preset

    @staticmethod
    def _stage_reference(image_path: str) -> str:
        """Copy ``image_path`` into the ComfyUI input dir, return its name.

        ComfyUI's LoadImage rejects absolute paths, so references are staged
        into the input directory under a deterministic, path-hashed name.
        Names already present in the input dir (uploads) pass through.
        """
        inp = os.environ.get("COMFY_INPUT_DIR", "/data/comfyui/input")
        os.makedirs(inp, exist_ok=True)
        if os.path.exists(image_path):
            name = ("sc-"
                    + hashlib.sha1(image_path.encode()).hexdigest()[:12]
                    + os.path.splitext(image_path)[1])
            dest = os.path.join(inp, name)
            if not os.path.exists(dest):
                shutil.copy2(image_path, dest)
            return name
        if os.path.exists(os.path.join(inp, image_path)):
            return image_path
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    def map_parameters(self, params: GenerationParams, section: SectionConfig,
                       last_frame_path: Optional[str],
                       quality_params: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
        quality = self._quality(params)
        resolved = params.resolved_prompt or self.resolve_prompt(params)
        seed = params.seed if params.seed is not None else 0
        qp = quality_params if quality_params is not None else params.quality_params

        image_path = params.reference_image if section.is_first else last_frame_path
        if not image_path:
            raise ValueError(
                "FramePack requires a reference image (image-to-video only).")
        reference_name = self._stage_reference(image_path)

        # Anti-drift maps to the sampler's gpu_memory_preservation headroom.
        anti_drift = self.DEFAULT_GPU_MEMORY_PRESERVATION
        if qp and qp.get("anti_drift") is not None:
            anti_drift = float(qp["anti_drift"])
        anti_drift = min(max(anti_drift, GPU_MEMORY_MIN), GPU_MEMORY_MAX)

        # House style: when the caller sends the base default negative, use
        # this engine's official default instead. FramePack zeroes the
        # negative at cfg 1.0, so the graph never consumes the text — the
        # value is kept for parity with the other adapters.
        negative = params.negative_prompt
        if negative == DEFAULT_NEGATIVE:
            negative = self.DEFAULT_NEGATIVE_PROMPT

        # Advisory only: the bucket node decides the actual pixel size.
        width, height = self.resolve_resolution(params)

        workflow = self.load_workflow(self.I2V_TEMPLATE)
        mapping = {
            "POSITIVE_PROMPT": resolved,
            "NEGATIVE_PROMPT": negative,  # unused in-graph (zeroed at cfg 1.0)
            "STEPS": int(quality["steps"]),
            "GUIDANCE": float(quality["guidance"]),
            "SEED": seed,
            "FPS": section.fps,
            "TOTAL_SECONDS": float(min(section.approx_seconds
                                       or self.TARGET_SECONDS,
                                       MAX_TOTAL_SECONDS)),
            "ANTI_DRIFT": anti_drift,
            "REFERENCE_IMAGE": reference_name,
            "WIDTH": width,   # reporting only — not in the template
            "HEIGHT": height,
        }
        return self._substitute(workflow, mapping)
