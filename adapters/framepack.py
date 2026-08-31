"""FramePack adapter.

FramePack uses a progressive frame-section approach and produces a long clip in
a single workflow call — the model handles internal sectioning itself. We
therefore target the full 30 seconds in one render.

* I2V is always supported.
* T2V is only exposed if the installed implementation actually provides a
  text-only entry node (detected at runtime).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from model_detector import ModelStatus

from .base import (ModelAdapter, GenerationParams, SectionConfig,
                   DEFAULT_NEGATIVE)


class FramePackAdapter(ModelAdapter):
    MODEL_NAME = "FramePack"
    MODEL_KEY = "framepack"
    SUPPORTED_MODES = {"i2v", "t2v"}  # t2v gated by detection
    TARGET_SECONDS = 30

    REQUIRED_NODES = ["FramePackSampler"]
    # T2V is only available when this node exists.
    T2V_NODE = "FramePackTextEncode"
    CHECKPOINT_HINTS = ["framepack", "hunyuan"]
    VAE_HINTS = ["hunyuan", "vae"]
    TEXT_ENCODER_HINTS = ["llama", "clip_l", "llava"]

    I2V_TEMPLATE = "framepack_i2v.json"
    T2V_TEMPLATE = "framepack_i2v.json"  # reused; blank init frame when T2V

    QUALITY = {
        "draft": {"steps": 15, "guidance": 6.0},
        "balanced": {"steps": 25, "guidance": 10.0},
        "quality": {"steps": 35, "guidance": 12.0},
    }

    def detect_capabilities(self, object_info: Dict[str, Any],
                            model_files: Dict[str, List[str]]) -> ModelStatus:
        status = super().detect_capabilities(object_info, model_files)
        # I2V always supported when available; T2V requires the text node.
        if status.available:
            status.i2v_capable = True
            has_t2v_node = bool(object_info) and self.T2V_NODE in object_info
            status.t2v_capable = has_t2v_node
            if not has_t2v_node:
                status.detail += " (T2V not supported by this FramePack build)"
        return status

    def get_section_strategy(self, params: GenerationParams) -> List[SectionConfig]:
        """FramePack renders the whole clip in one call."""
        fps = params.fps or 30
        frames = int(round(self.TARGET_SECONDS * fps))
        return [SectionConfig(
            index=0,
            frames=frames,
            fps=fps,
            is_first=True,
            use_reference=(params.mode == "i2v"),
            approx_seconds=self.TARGET_SECONDS,
        )]

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

        image_path = params.reference_image if params.mode == "i2v" else last_frame_path
        workflow = self.load_workflow(self.I2V_TEMPLATE)

        mapping = {
            "POSITIVE_PROMPT": resolved,
            "NEGATIVE_PROMPT": params.negative_prompt or DEFAULT_NEGATIVE,
            "STEPS": int(quality["steps"]),
            "GUIDANCE": float(quality["guidance"]),
            "WIDTH": width,
            "HEIGHT": height,
            "SEED": seed,
            "FRAMES": section.frames,
            "FPS": section.fps,
            # Anti-drift / continuity controls (used where the node supports them).
            "ANTI_DRIFT": float(params.motion_strength or 1.0),
            "TOTAL_SECONDS": self.TARGET_SECONDS,
            "CHECKPOINT_NAME": "FramePackI2V_HY_bf16.safetensors",
            "REFERENCE_IMAGE": image_path or "",
        }
        return self._substitute(workflow, mapping)
