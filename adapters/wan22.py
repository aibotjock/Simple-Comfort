"""Wan 2.2 14B adapter (dual-expert, ComfyUI core nodes).

Rebuilt against the official ComfyUI Wan 2.2 14B workflows: the high-noise
and low-noise experts are each loaded with their own ``UNETLoader`` ->
``LoraLoaderModelOnly`` -> ``ModelSamplingSD3`` chain and drive two chained
``KSamplerAdvanced`` nodes split at an interpolation step.

* Default path is the official Comfy-Org lightx2v 4-step distilled config
  (steps 4, cfg 1.0, shift 5.0, split 2, LoRA strength 1.0).
* The ``quality`` preset switches to the official standard path (steps 20,
  cfg 3.5, shift 8.0, split 10) by setting the LoRA strength to 0 — one
  template pair serves both configs.

Wan 2.2 renders natively at 16fps in 4n+1 frame chunks (81 frames ~ 5s).
Strategy: six 5-second sections concatenated to reach ~30s.

* Section 0 in T2V mode uses no conditioning image (pure text-to-video).
* Section 0 in I2V mode uses the user's reference image.
* Every subsequent section conditions on the last frame of the previous one.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Any, Dict, List, Optional

from model_detector import ModelStatus

from .base import (ModelAdapter, GenerationParams, SectionConfig,
                   DEFAULT_NEGATIVE)

# Official Wan negative prompt (from the ComfyUI Wan 2.2 example workflows).
# Substituted whenever the caller only passes the app-wide default.
WAN22_DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _wan_frame_count(seconds: float, fps: int) -> int:
    """Round to the nearest Wan-friendly frame count (4n + 1)."""
    raw = max(int(round(seconds * fps)), 17)
    n = round((raw - 1) / 4)
    return int(n * 4 + 1)


class Wan22Adapter(ModelAdapter):
    MODEL_NAME = "Wan 2.2"
    MODEL_KEY = "wan22"
    SUPPORTED_MODES = {"t2v", "i2v"}
    TARGET_SECONDS = 30
    SECTION_SECONDS = 5
    NATIVE_FPS = 16
    NATIVE_FRAMES = 81  # ~5s at 16fps

    REQUIRED_NODES = ["UNETLoader", "LoraLoaderModelOnly", "ModelSamplingSD3",
                      "CLIPLoader", "VAELoader", "CLIPTextEncode",
                      "KSamplerAdvanced", "EmptyHunyuanLatentVideo",
                      "WanImageToVideo", "VAEDecode", "CreateVideo",
                      "SaveVideo"]
    # Both dual-expert families must exist (high-noise files are the marker).
    CHECKPOINT_HINTS = ["wan2.2_t2v_high", "wan2.2_i2v_high"]
    VAE_HINTS = ["wan_2.1"]
    TEXT_ENCODER_HINTS = ["umt5"]

    T2V_TEMPLATE = "wan22_t2v.json"
    I2V_TEMPLATE = "wan22_i2v.json"

    # Official Wan 2.2 resolutions (multiples of 16, from the example flows).
    ASPECT_RESOLUTIONS = {
        "16:9": (1280, 704),
        "9:16": (704, 1280),
        "1:1": (640, 640),
    }

    # draft/balanced: official lightx2v 4-step distilled config.
    # quality: official standard config; LoRA nodes stay in the graph at
    # strength 0 so one template serves both paths.
    QUALITY = {
        "draft": {"steps": 4, "guidance": 1.0, "shift": 5.0,
                  "lora_strength": 1.0, "step_split": 2},
        "balanced": {"steps": 4, "guidance": 1.0, "shift": 5.0,
                     "lora_strength": 1.0, "step_split": 2},
        "quality": {"steps": 20, "guidance": 3.5, "shift": 8.0,
                    "lora_strength": 0.0, "step_split": 10},
    }

    # ------------------------------------------------------------------ #
    # Capability detection
    # ------------------------------------------------------------------ #
    def detect_capabilities(self, object_info: Dict[str, Any],
                            model_files: Dict[str, List[str]]) -> ModelStatus:
        """Require BOTH experts (t2v + i2v high-noise files) plus shared assets.

        This engine is only 'available' when both modes' experts exist, so the
        dual-expert templates can always be satisfied.
        """
        status = ModelStatus(model=self.MODEL_KEY, display_name=self.MODEL_NAME)
        missing: List[str] = []

        available_nodes = set(object_info.keys()) if object_info else set()
        missing_nodes = [n for n in self.REQUIRED_NODES if n not in available_nodes]

        unet_pool: List[str] = []
        for folder in ("checkpoints", "unet", "diffusion_models"):
            unet_pool.extend(model_files.get(folder, []))
        unet_pool = [p.lower() for p in unet_pool]
        has_t2v = any("wan2.2_t2v_high" in p for p in unet_pool)
        has_i2v = any("wan2.2_i2v_high" in p for p in unet_pool)

        vae_pool = [p.lower() for p in model_files.get("vae", [])]
        has_vae = any("wan_2.1" in p for p in vae_pool)

        te_pool = [p.lower() for p in
                   model_files.get("text_encoders", []) + model_files.get("clip", [])]
        has_te = any("umt5" in p for p in te_pool)

        if missing_nodes:
            missing.extend(f"node:{n}" for n in missing_nodes)
        if not has_t2v:
            missing.append("wan2.2_t2v_high_noise expert")
        if not has_i2v:
            missing.append("wan2.2_i2v_high_noise expert")
        if not has_vae:
            missing.append("vae")
        if not has_te:
            missing.append("text_encoder")

        if missing_nodes:
            status.status = "missing_nodes"
        elif not (has_t2v and has_i2v):
            status.status = "missing_checkpoint"
        elif not has_vae:
            status.status = "missing_vae"
        elif not has_te:
            status.status = "missing_text_encoder"
        else:
            status.status = "available"

        status.available = status.status == "available"
        status.missing = missing
        status.t2v_capable = status.available
        status.i2v_capable = status.available
        if status.available:
            status.detail = ("Dual-expert T2V + I2V 14B models, umt5 and "
                             "wan 2.1 VAE detected.")
        else:
            status.detail = ("Missing: " + ", ".join(missing)) if missing \
                else "Model assets not fully detected."
        return status

    # ------------------------------------------------------------------ #
    # Sectioning
    # ------------------------------------------------------------------ #
    def get_section_strategy(self, params: GenerationParams) -> List[SectionConfig]:
        """Split the target duration into 5s (81-frame) Wan sections.

        Wan 2.2 renders natively at 16fps regardless of the requested output
        fps, so every section is forced to 16fps.
        """
        fps = self.NATIVE_FPS
        section_seconds = params.segment_length or self.SECTION_SECONDS
        target = params.effective_target_seconds(self.TARGET_SECONDS)
        num_sections = max(1, round(target / section_seconds))
        # Frames per section at the native fps, kept 4n+1 friendly.
        frames = _wan_frame_count(section_seconds, fps)

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

    # ------------------------------------------------------------------ #
    # Parameter mapping
    # ------------------------------------------------------------------ #
    def _quality(self, params: GenerationParams,
                 quality_params: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, float]:
        """Resolve the Wan quality preset with user/pro-panel overrides."""
        preset = self.QUALITY.get(params.quality, self.QUALITY["balanced"]).copy()
        if params.steps is not None:
            preset["steps"] = params.steps
        if params.guidance is not None:
            preset["guidance"] = params.guidance
        qp = quality_params if quality_params is not None \
            else (params.quality_params or {})
        for key in ("shift", "step_split", "lora_strength"):
            if qp.get(key) is not None:
                preset[key] = qp[key]
        return preset

    def resolve_resolution(self, params: GenerationParams) -> tuple:
        """Official Wan 2.2 resolutions (multiples of 16)."""
        if params.width and params.height:
            return params.width, params.height
        return self.ASPECT_RESOLUTIONS.get(params.aspect_ratio,
                                           self.ASPECT_RESOLUTIONS["16:9"])

    @staticmethod
    def _stage_reference(image_path: str) -> Optional[str]:
        """Copy a reference image into ComfyUI's input dir.

        ComfyUI's LoadImage rejects absolute paths, so the image is staged
        under a stable hashed name and referenced by its bare filename.
        Returns ``None`` when the image is missing (caller falls back to T2V).
        """
        if not image_path:
            return None
        if not os.path.isfile(image_path):
            return None
        input_dir = os.environ.get("COMFY_INPUT_DIR", "/data/comfyui/input")
        os.makedirs(input_dir, exist_ok=True)
        name = ("sc-" + hashlib.sha1(image_path.encode()).hexdigest()[:12]
                + os.path.splitext(image_path)[1])
        dest = os.path.join(input_dir, name)
        if not os.path.exists(dest):
            shutil.copy2(image_path, dest)
        return name

    def _negative_prompt(self, params: GenerationParams) -> str:
        """Use Wan's official negative unless the user supplied a custom one."""
        negative = params.negative_prompt
        if not negative or negative == DEFAULT_NEGATIVE:
            return WAN22_DEFAULT_NEGATIVE
        return negative

    def map_parameters(self, params: GenerationParams, section: SectionConfig,
                       last_frame_path: Optional[str],
                       quality_params: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
        width, height = self.resolve_resolution(params)
        quality = self._quality(params, quality_params)
        resolved = params.resolved_prompt or self.resolve_prompt(params)
        seed = params.seed if params.seed is not None else 0

        if section.is_first:
            image_path = params.reference_image if params.mode == "i2v" else None
        else:
            image_path = last_frame_path

        staged = self._stage_reference(image_path) if image_path else None
        template = self.I2V_TEMPLATE if staged else self.T2V_TEMPLATE
        workflow = self.load_workflow(template)

        # Dual-expert sampler split: high-noise expert covers steps
        # [0, split), low-noise expert [split, steps).
        steps = max(int(quality["steps"]), 2)
        step_split = int(min(max(int(quality["step_split"]), 1), steps - 1))

        mapping = {
            "POSITIVE_PROMPT": resolved,
            "NEGATIVE_PROMPT": self._negative_prompt(params),
            "STEPS": steps,
            "GUIDANCE": float(quality["guidance"]),
            "SHIFT": float(quality["shift"]),
            "LORA_STRENGTH": float(quality["lora_strength"]),
            "STEP_SPLIT": step_split,
            "WIDTH": width,
            "HEIGHT": height,
            "SEED": seed + section.index,
            "FRAMES": section.frames,
            "FPS": section.fps,
            "REFERENCE_IMAGE": staged or "",
        }
        return self._substitute(workflow, mapping)
