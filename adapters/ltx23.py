"""LTX-2.3 adapter.

Strategy: two ~15-second sections concatenated to reach ~30s.

* Section 0 is T2V (or I2V from the user's reference image).
* Section 1 is I2V, conditioned on the last frame of section 0.

Frame counts are derived from the target FPS so the two sections total ~30s.
LTX-Video renders best with frame counts of the form ``8*k + 1``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import (ModelAdapter, GenerationParams, SectionConfig,
                   DEFAULT_NEGATIVE)


def _ltx_frame_count(seconds: float, fps: int) -> int:
    """Round to the nearest LTX-friendly frame count (8*k + 1)."""
    raw = max(int(round(seconds * fps)), 9)
    k = round((raw - 1) / 8)
    return int(k * 8 + 1)


class LTX23Adapter(ModelAdapter):
    MODEL_NAME = "LTX-2.3"
    MODEL_KEY = "ltx23"
    SUPPORTED_MODES = {"t2v", "i2v"}
    TARGET_SECONDS = 30

    REQUIRED_NODES = ["LTXVLoader", "LTXVModelConfigurator"]
    CHECKPOINT_HINTS = ["ltx", "ltxv"]
    VAE_HINTS = ["ltx"]
    TEXT_ENCODER_HINTS = ["t5", "t5xxl"]

    T2V_TEMPLATE = "ltx23_t2v.json"
    I2V_TEMPLATE = "ltx23_i2v.json"

    # LTX quality tuning (steps, guidance).
    QUALITY = {
        "draft": {"steps": 20, "guidance": 3.0},
        "balanced": {"steps": 30, "guidance": 3.5},
        "quality": {"steps": 45, "guidance": 4.5},
    }

    def get_section_strategy(self, params: GenerationParams) -> List[SectionConfig]:
        fps = params.fps or 24
        half = self.TARGET_SECONDS / 2.0
        if params.segment_length:
            half = params.segment_length
        frames = _ltx_frame_count(half, fps)
        sections: List[SectionConfig] = []
        for i in range(2):
            first = i == 0
            sections.append(SectionConfig(
                index=i,
                frames=frames,
                fps=fps,
                is_first=first,
                # Section 0 uses reference only in I2V mode; later sections
                # always condition on the previous section's last frame.
                use_reference=(not first) or (params.mode == "i2v"),
                approx_seconds=frames / fps,
            ))
        return sections

    def _quality(self, params: GenerationParams) -> Dict[str, float]:
        preset = self.QUALITY.get(params.quality, self.QUALITY["balanced"]).copy()
        if params.steps is not None:
            preset["steps"] = params.steps
        if params.guidance is not None:
            preset["guidance"] = params.guidance
        return preset

    def map_parameters(self, params: GenerationParams, section: SectionConfig,
                       last_frame_path: Optional[str]) -> Dict[str, Any]:
        width, height = self.resolve_resolution(params)
        quality = self._quality(params)
        resolved = params.resolved_prompt or self.resolve_prompt(params)
        seed = params.seed if params.seed is not None else 0

        use_image = section.use_reference
        # Which conditioning image to use for this section.
        image_path = last_frame_path if not section.is_first else params.reference_image
        if not section.is_first:
            image_path = last_frame_path

        template = self.I2V_TEMPLATE if use_image and image_path else self.T2V_TEMPLATE
        workflow = self.load_workflow(template)

        mapping = {
            "POSITIVE_PROMPT": resolved,
            "NEGATIVE_PROMPT": params.negative_prompt or DEFAULT_NEGATIVE,
            "STEPS": int(quality["steps"]),
            "GUIDANCE": float(quality["guidance"]),
            "WIDTH": width,
            "HEIGHT": height,
            "SEED": seed + section.index,  # vary per section for motion continuity
            "FRAMES": section.frames,
            "FPS": section.fps,
            "CHECKPOINT_NAME": "ltx-video-2b-v0.9.5.safetensors",
            "REFERENCE_IMAGE": image_path or "",
        }
        return self._substitute(workflow, mapping)
