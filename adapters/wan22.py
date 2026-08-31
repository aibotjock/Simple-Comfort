"""Wan 2.2 adapter.

Strategy: six 5-second sections concatenated to reach ~30s. Wan 2.2 TI2V-5B
handles both T2V and I2V in a single model.

* Section 0 in T2V mode uses no conditioning image (pure text-to-video).
* Section 0 in I2V mode uses the user's reference image.
* Every subsequent section conditions on the last frame of the previous one.

Wan 2.2 renders natively at 16fps with 81-frame chunks (~5s).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import (ModelAdapter, GenerationParams, SectionConfig,
                   DEFAULT_NEGATIVE)


class Wan22Adapter(ModelAdapter):
    MODEL_NAME = "Wan 2.2"
    MODEL_KEY = "wan22"
    SUPPORTED_MODES = {"t2v", "i2v"}
    TARGET_SECONDS = 30
    SECTION_SECONDS = 5
    NATIVE_FPS = 16
    NATIVE_FRAMES = 81  # ~5s at 16fps

    REQUIRED_NODES = ["WanVideoModelLoader", "WanVideoSampler"]
    CHECKPOINT_HINTS = ["wan", "wan2.2", "ti2v", "wan22"]
    VAE_HINTS = ["wan", "wan2"]
    TEXT_ENCODER_HINTS = ["umt5", "t5"]

    T2V_TEMPLATE = "wan22_t2v.json"
    I2V_TEMPLATE = "wan22_i2v.json"

    QUALITY = {
        "draft": {"steps": 15, "guidance": 4.0},
        "balanced": {"steps": 25, "guidance": 5.0},
        "quality": {"steps": 35, "guidance": 6.0},
    }

    def get_section_strategy(self, params: GenerationParams) -> List[SectionConfig]:
        fps = params.fps or self.NATIVE_FPS
        section_seconds = params.segment_length or self.SECTION_SECONDS
        target = params.effective_target_seconds(self.TARGET_SECONDS)
        num_sections = max(1, round(target / section_seconds))
        # Frames per section derived from target fps.
        frames = max(int(round(section_seconds * fps)), 17)
        # Keep 4n+1 friendly for Wan.
        frames = ((frames - 1) // 4) * 4 + 1

        sections: List[SectionConfig] = []
        for i in range(num_sections):
            first = i == 0
            sections.append(SectionConfig(
                index=i,
                frames=frames,
                fps=fps,
                is_first=first,
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
                       last_frame_path: Optional[str],
                       quality_params: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
        width, height = self.resolve_resolution(params)
        quality = self._quality(params)
        resolved = params.resolved_prompt or self.resolve_prompt(params)
        seed = params.seed if params.seed is not None else 0
        qp = quality_params if quality_params is not None else params.quality_params

        if section.is_first:
            image_path = params.reference_image if params.mode == "i2v" else None
        else:
            image_path = last_frame_path

        use_image = bool(image_path)
        template = self.I2V_TEMPLATE if use_image else self.T2V_TEMPLATE
        workflow = self.load_workflow(template)

        # Motion strength can be overridden by the professional quality panel.
        motion = params.motion_strength
        if qp and qp.get("motion_strength") is not None:
            motion = float(qp["motion_strength"])

        mapping = {
            "POSITIVE_PROMPT": resolved,
            "NEGATIVE_PROMPT": params.negative_prompt or DEFAULT_NEGATIVE,
            "STEPS": int(quality["steps"]),
            "GUIDANCE": float(quality["guidance"]),
            "WIDTH": width,
            "HEIGHT": height,
            "SEED": seed + section.index,
            "FRAMES": section.frames,
            "FPS": section.fps,
            "MOTION_STRENGTH": float(motion if motion is not None else 1.0),
            "CHECKPOINT_NAME": "wan2.2_ti2v_5B_fp16.safetensors",
            "REFERENCE_IMAGE": image_path or "",
        }
        if qp:
            if qp.get("camera_motion") is not None:
                mapping["CAMERA_MOTION"] = float(qp["camera_motion"])
            if qp.get("face_stability") is not None:
                mapping["FACE_STABILITY"] = float(qp["face_stability"])
        return self._substitute(workflow, mapping)
