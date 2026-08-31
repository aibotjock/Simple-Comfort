"""Model adapter base class and shared data structures.

An adapter encapsulates *everything* model-specific: the ComfyUI workflow
template, capability detection, parameter mapping, the 30-second sectioning
strategy and continuation logic. The rest of the application never references
individual ComfyUI node IDs — it only talks to adapters.
"""
from __future__ import annotations

import copy
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from model_detector import ModelStatus

log = logging.getLogger("simple_comfort.adapter")

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"

# Aspect ratio -> (width, height). Adapters may override per-model constraints.
ASPECT_RESOLUTIONS: Dict[str, tuple] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (960, 960),
}

# Lip-sync guidance appended to prompts when Lip-Sync Ready is enabled.
LIP_SYNC_SUFFIX = (
    "stable facial identity, consistent face, stable jaw geometry, "
    "unobstructed mouth, clearly visible face, realistic natural blinking, "
    "natural facial micro-movements, moderate head movement, "
    "relaxed neutral lightly parted mouth, consistent face lighting, "
    "no hands near mouth, no microphone, no subtitles, no text overlays"
)

DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted face, deformed, extra limbs, "
    "watermark, text, subtitles, jpeg artifacts, flickering"
)


@dataclass
class GenerationParams:
    """Normalised generation request passed to an adapter."""

    prompt: str
    mode: str  # "t2v" | "i2v"
    aspect_ratio: str = "16:9"
    quality: str = "balanced"  # draft | balanced | quality
    lip_sync: bool = True
    seed: Optional[int] = None
    fps: int = 24
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    segment_length: Optional[float] = None
    motion_strength: Optional[float] = None
    negative_prompt: str = DEFAULT_NEGATIVE
    reference_image: Optional[str] = None  # absolute path on disk / uploaded name
    resolved_prompt: str = ""  # filled in by adapter (prompt + lip-sync suffix)
    # Optional shorter target duration (e.g. b-roll wants ~8-10s instead of 30).
    target_seconds: Optional[float] = None
    # Professional-quality adapter parameters (model-specific, all optional):
    # e.g. camera_motion, face_stability, upscale, anti_drift, denoise,
    # interpolate_fps. Adapters read only the keys they support.
    quality_params: Dict[str, Any] = field(default_factory=dict)

    def effective_target_seconds(self, default: float) -> float:
        """Return the requested target duration, falling back to ``default``."""
        if self.target_seconds and self.target_seconds > 0:
            return float(self.target_seconds)
        return float(default)


@dataclass
class SectionConfig:
    """Configuration for one internal render section."""

    index: int
    frames: int
    fps: int
    is_first: bool
    use_reference: bool  # whether this section conditions on an input frame
    approx_seconds: float = 0.0


@dataclass
class ModelCapabilities:
    """Capability summary produced during detection."""

    available: bool = False
    t2v_capable: bool = False
    i2v_capable: bool = False
    status: str = "unknown"
    missing: List[str] = field(default_factory=list)
    detail: str = ""


# Quality preset -> (steps, guidance). Adapters override per-model tuning.
QUALITY_PRESETS: Dict[str, Dict[str, float]] = {
    "draft": {"steps": 12, "guidance": 2.5},
    "balanced": {"steps": 25, "guidance": 3.5},
    "quality": {"steps": 40, "guidance": 5.0},
}


class ModelAdapter(ABC):
    """Abstract base for all model adapters."""

    MODEL_NAME: str = "unknown"
    MODEL_KEY: str = "unknown"
    SUPPORTED_MODES: Set[str] = set()
    TARGET_SECONDS: int = 30

    # Required assets used by :meth:`detect_capabilities`.
    REQUIRED_NODES: List[str] = []
    CHECKPOINT_HINTS: List[str] = []
    VAE_HINTS: List[str] = []
    TEXT_ENCODER_HINTS: List[str] = []

    T2V_TEMPLATE: str = ""
    I2V_TEMPLATE: str = ""

    # ------------------------------------------------------------------ #
    # Workflow handling
    # ------------------------------------------------------------------ #
    def load_workflow(self, template_name: str) -> Dict[str, Any]:
        """Load and return a deep copy of a workflow template JSON."""
        path = WORKFLOW_DIR / template_name
        if not path.exists():
            raise FileNotFoundError(f"Workflow template not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Strip documentation-only top-level keys (kept out of the API payload).
        data.pop("_comment", None)
        data.pop("_documentation", None)
        return copy.deepcopy(data)

    @staticmethod
    def _substitute(workflow: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Replace ``{{TOKEN}}`` placeholders throughout a workflow dict.

        Values that are exactly a single ``{{TOKEN}}`` keep their native type
        (int/float/etc.); embedded tokens are string-substituted.
        """
        def convert(value):
            if isinstance(value, str):
                stripped = value.strip()
                # Whole-value token -> preserve type.
                for token, repl in mapping.items():
                    if stripped == "{{" + token + "}}":
                        return repl
                out = value
                for token, repl in mapping.items():
                    out = out.replace("{{" + token + "}}", str(repl))
                return out
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value

        return convert(workflow)

    # ------------------------------------------------------------------ #
    # Prompt handling
    # ------------------------------------------------------------------ #
    def build_lip_sync_suffix(self, prompt: str) -> str:
        """Append lip-sync guidance to a prompt (idempotent-ish)."""
        prompt = prompt.strip()
        if not prompt:
            return LIP_SYNC_SUFFIX
        return f"{prompt}, {LIP_SYNC_SUFFIX}"

    def resolve_prompt(self, params: GenerationParams) -> str:
        if params.lip_sync:
            return self.build_lip_sync_suffix(params.prompt)
        return params.prompt.strip()

    def resolve_resolution(self, params: GenerationParams) -> tuple:
        if params.width and params.height:
            return params.width, params.height
        return ASPECT_RESOLUTIONS.get(params.aspect_ratio, (1280, 720))

    def resolve_quality(self, params: GenerationParams) -> Dict[str, float]:
        preset = QUALITY_PRESETS.get(params.quality, QUALITY_PRESETS["balanced"]).copy()
        if params.steps is not None:
            preset["steps"] = params.steps
        if params.guidance is not None:
            preset["guidance"] = params.guidance
        return preset

    # ------------------------------------------------------------------ #
    # Capability detection (shared implementation)
    # ------------------------------------------------------------------ #
    def detect_capabilities(self, object_info: Dict[str, Any],
                            model_files: Dict[str, List[str]]) -> ModelStatus:
        """Default detection based on node + asset hints. Adapters may override."""
        status = ModelStatus(model=self.MODEL_KEY, display_name=self.MODEL_NAME)
        missing: List[str] = []

        available_nodes = set(object_info.keys()) if object_info else set()
        missing_nodes = [n for n in self.REQUIRED_NODES if n not in available_nodes]

        def any_match(hints: List[str], folders: List[str]) -> bool:
            if not hints:
                return True
            pool: List[str] = []
            for folder in folders:
                pool.extend(model_files.get(folder, []))
            pool_l = [p.lower() for p in pool]
            return any(any(h.lower() in p for p in pool_l) for h in hints)

        has_ckpt = any_match(
            self.CHECKPOINT_HINTS,
            ["checkpoints", "unet", "diffusion_models"],
        )
        has_vae = any_match(self.VAE_HINTS, ["vae"])
        has_te = any_match(
            self.TEXT_ENCODER_HINTS, ["text_encoders", "clip"]
        )

        if missing_nodes:
            missing.extend(f"node:{n}" for n in missing_nodes)
        if not has_ckpt:
            missing.append("checkpoint")
        if not has_vae:
            missing.append("vae")
        if not has_te:
            missing.append("text_encoder")

        if missing_nodes:
            status.status = "missing_nodes"
        elif not has_ckpt:
            status.status = "missing_checkpoint"
        elif not has_vae:
            status.status = "missing_vae"
        elif not has_te:
            status.status = "missing_text_encoder"
        else:
            status.status = "available"

        status.available = status.status == "available"
        status.missing = missing
        if status.available:
            status.t2v_capable = "t2v" in self.SUPPORTED_MODES
            status.i2v_capable = "i2v" in self.SUPPORTED_MODES
            status.detail = "All required assets detected."
        else:
            status.t2v_capable = False
            status.i2v_capable = False
            status.detail = "Missing: " + ", ".join(missing) if missing else \
                "Model assets not fully detected."
        return status

    # ------------------------------------------------------------------ #
    # Abstract, model-specific behaviour
    # ------------------------------------------------------------------ #
    @abstractmethod
    def get_section_strategy(self, params: GenerationParams) -> List[SectionConfig]:
        """Return the list of internal sections used to reach 30 seconds."""

    @abstractmethod
    def map_parameters(self, params: GenerationParams, section: SectionConfig,
                       last_frame_path: Optional[str]) -> Dict[str, Any]:
        """Build the concrete ComfyUI workflow JSON for one section."""
