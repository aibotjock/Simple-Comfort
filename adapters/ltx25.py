"""LTX-2.5 adapter (distilled single- and two-stage, joint audio/video).

Mirrors the official ComfyUI LTX-2.5 distilled workflow: a 22B int8
transformer, the finetuned Gemma4-12B text encoder and the Gemma4 E2B prompt
enhancer running INSIDE the graph (``TextGenerateLTX2Prompt``) so every
section's positive prompt is rewritten on the box — no external API. The
model is A/V joint, so the graph carries an audio-latent branch end to end
and every section ships with generated audio.

Strategy: LTX-2.5 renders long durations natively, so sections exist purely
to bound peak memory and give last-frame chaining points — each is ~6s
(145 frames @ 24fps, frame counts of the form ``8*k + 1``). Section 0 is T2V
(or I2V from the user's reference); later sections are I2V conditioned on
the previous section's last frame.

Quality presets map to graph variants, not step counts: ``draft`` uses the
single-stage template (8 sampler steps) while ``balanced`` and ``quality``
use the recommended two-stage template (8 + 3 = 11 steps, decoding at 2x the
base resolution). The step count and guidance are FIXED by the distilled
sigma schedules baked into the templates, so user ``steps``/``guidance``
overrides are intentionally ignored — :meth:`resolve_quality` reports the
effective values so job metadata stays truthful.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from .base import (ModelAdapter, GenerationParams, SectionConfig,
                   DEFAULT_NEGATIVE)

log = logging.getLogger("simple_comfort.adapter.ltx25")


def _ltx_frame_count(seconds: float, fps: int) -> int:
    """Round to the nearest LTX-friendly frame count (8*k + 1)."""
    raw = max(int(round(seconds * fps)), 9)
    k = round((raw - 1) / 8)
    return int(k * 8 + 1)


class LTX25Adapter(ModelAdapter):
    MODEL_NAME = "LTX 2.5"
    MODEL_KEY = "ltx25"
    SUPPORTED_MODES = {"t2v", "i2v"}
    TARGET_SECONDS = 30

    REQUIRED_NODES = [
        "UNETLoader", "CLIPLoader", "VAELoader", "TextGenerateLTX2Prompt",
        "LTXVConditioning", "EmptyLTXVLatentVideo", "LTXVImgToVideoInplace",
        "LTXVPreprocess", "LTXVConcatAVLatent", "LTXVSeparateAVLatent",
        "SamplerCustomAdvanced", "ManualSigmas", "VAEDecodeTiled",
        "CreateVideo", "SaveVideo", "ResizeImageMaskNode",
    ]
    CHECKPOINT_HINTS = ["ltx-2.5"]
    VAE_HINTS = ["ltx-2.5"]
    TEXT_ENCODER_HINTS = ["gemma"]

    T2V_TEMPLATE = "ltx25_t2v.json"
    I2V_TEMPLATE = "ltx25_i2v.json"
    T2V_TEMPLATE_2STAGE = "ltx25_t2v_2stage.json"
    I2V_TEMPLATE_2STAGE = "ltx25_i2v_2stage.json"

    # Official LTX-2.5 base resolution is 960x544 (16:9); multiples of 32.
    ASPECT_RESOLUTIONS: Dict[str, tuple] = {
        "16:9": (960, 544),
        "9:16": (544, 960),
        "1:1": (768, 768),
    }

    # Official LTX-2.5 negative prompt, used when the caller leaves the app
    # default in place. The negative is encoded as written and never enhanced.
    OFFICIAL_NEGATIVE = (
        "pc game, console game, video game, cartoon, childish, ugly"
    )

    # ~6s sections keep peak memory bounded while chaining stays smooth.
    SECTION_SECONDS = 6.0

    # Effective sampler budget fixed by the distilled sigma schedules.
    SINGLE_STAGE_STEPS = 8       # sigmas: 1.0 ... 0.0 (9 values)
    TWO_STAGE_STEPS = 11         # 8 (stage 1) + 3 (stage 2 from sigma 0.85)
    DISTILLED_GUIDANCE = 1.0     # CFGGuider cfg used by both stages

    def get_section_strategy(self, params: GenerationParams) -> List[SectionConfig]:
        fps = params.fps or 24
        target = params.effective_target_seconds(self.TARGET_SECONDS)
        section_seconds = params.segment_length or self.SECTION_SECONDS
        num_sections = max(1, round(target / section_seconds))
        frames = _ltx_frame_count(section_seconds, fps)
        sections: List[SectionConfig] = []
        for i in range(num_sections):
            first = i == 0
            sections.append(SectionConfig(
                index=i,
                frames=frames,
                fps=fps,
                is_first=first,
                # Section 0 uses the reference only in I2V mode; later
                # sections always condition on the previous last frame.
                use_reference=(not first) or (params.mode == "i2v"),
                approx_seconds=frames / fps,
            ))
        return sections

    def resolve_resolution(self, params: GenerationParams) -> tuple:
        """Model-specific resolution map (960x544 base, multiples of 32)."""
        if params.width and params.height:
            # EmptyLTXVLatentVideo validates against a step of 32.
            width = max(64, int(round(params.width / 32)) * 32)
            height = max(64, int(round(params.height / 32)) * 32)
            return width, height
        return self.ASPECT_RESOLUTIONS.get(params.aspect_ratio, (960, 544))

    def _two_stage(self, params: GenerationParams) -> bool:
        """Two-stage is the recommended default; only ``draft`` opts out."""
        return params.quality != "draft"

    def resolve_quality(self, params: GenerationParams) -> Dict[str, float]:
        """Report the EFFECTIVE steps/guidance of the chosen graph variant.

        The distilled sigma schedules fix both, so user overrides are
        deliberately ignored (reporting them would be a lie in metadata).
        """
        steps = self.TWO_STAGE_STEPS if self._two_stage(params) \
            else self.SINGLE_STAGE_STEPS
        return {"steps": steps, "guidance": self.DISTILLED_GUIDANCE}

    def _resolve_negative(self, params: GenerationParams) -> str:
        """Swap the app-default negative for the official LTX-2.5 one."""
        if not params.negative_prompt or \
                params.negative_prompt.strip() == DEFAULT_NEGATIVE:
            return self.OFFICIAL_NEGATIVE
        return params.negative_prompt

    @staticmethod
    def _stage_reference_image(image_path: str) -> Optional[str]:
        """Copy a reference into the ComfyUI input dir; return its bare name.

        LoadImage rejects absolute paths (input-dir containment is enforced
        server-side), so images must be staged and referenced by name.
        """
        if not image_path:
            return None
        if not os.path.isfile(image_path):
            log.warning("Reference image not found: %s", image_path)
            return None
        inp = os.environ.get("COMFY_INPUT_DIR", "/data/comfyui/input")
        os.makedirs(inp, exist_ok=True)
        name = ("sc-" + hashlib.sha1(image_path.encode()).hexdigest()[:12]
                + os.path.splitext(image_path)[1])
        dest = os.path.join(inp, name)
        if not os.path.exists(dest):
            shutil.copy2(image_path, dest)
        return name

    def map_parameters(self, params: GenerationParams, section: SectionConfig,
                       last_frame_path: Optional[str]) -> Dict[str, Any]:
        width, height = self.resolve_resolution(params)
        resolved = params.resolved_prompt or self.resolve_prompt(params)
        seed = params.seed if params.seed is not None else 0
        section_seed = seed + section.index  # vary per section

        # Which conditioning image applies to this section.
        image_path = last_frame_path if not section.is_first \
            else params.reference_image
        staged = self._stage_reference_image(image_path) \
            if section.use_reference else None

        two_stage = self._two_stage(params)
        if section.use_reference and staged:
            template = self.I2V_TEMPLATE_2STAGE if two_stage \
                else self.I2V_TEMPLATE
        else:
            # No usable reference (pure t2v, or missing image) — the t2v
            # graph never references the reference tokens at all.
            template = self.T2V_TEMPLATE_2STAGE if two_stage \
                else self.T2V_TEMPLATE
        workflow = self.load_workflow(template)

        mapping = {
            "POSITIVE_PROMPT": resolved,
            "NEGATIVE_PROMPT": self._resolve_negative(params),
            "WIDTH": width,
            "HEIGHT": height,
            "SEED": section_seed,
            "SEED_STAGE2": section_seed + 1,  # independent stage-2 noise draw
            "FRAMES": section.frames,
            "FPS": section.fps,
        }
        if staged:
            mapping["REFERENCE_IMAGE"] = staged
        return self._substitute(workflow, mapping)
