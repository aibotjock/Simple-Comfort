"""LTX-2.5 adapter (AAN 10-workflow candidate pack, joint audio/video).

The backend graph set is the Applied AI News LTX-2.5 candidate pack
(``workflows/aan_ltx25/``): ten byte-faithful editor-format ComfyUI
workflows (verified against the pack's ``SHA256_MANIFEST.json``) that all
expose one COMMON INTERFACE — titled control nodes for the creative
prompt, duration, fps, working width/height, image-conditioning toggles,
uploads and seeds (see the pack's ``README.txt``). The adapter drives that
contract: it patches the control nodes, converts the graph to API format
with a pinned node-spec fixture (``pack_object_info.json``, captured from
ComfyUI 0.34.0), and queues the result.

The pack files store ``widgets_values`` in full input-spec order (values
for link-wired inputs included) — the editor and
:func:`custom_workflows.convert_ui_to_api` expect them with wired slots
removed, so :func:`_normalize_widgets` rewrites the arrays to that
canonical form before conversion. Without it, widget values silently land
on the wrong inputs (``batch_size`` swallowing ``width``, a float landing
on the boolean ``bypass``...).

Model stack (fixed filenames, see the pack's ``MODEL_REQUIREMENTS.txt``):
the 22B int8-convrot distilled transformer, the finetuned Gemma4-12B text
encoder and the Gemma4 E2B prompt enhancer running INSIDE the graph
(``TextGenerateLTX2Prompt``) so every section's prompt is rewritten
on the box — no external API. The model is A/V joint: an audio-latent
branch runs end to end and every section ships with generated audio.

Strategy: LTX-2.5 renders long durations natively, so sections exist
purely to bound peak memory and give last-frame chaining points — each is
~6s (145 frames @ 24fps). Section 0 is T2V (or I2V from the user's
reference); later sections are I2V conditioned on the previous section's
last frame.

The pack interface has ONE creative prompt and no negative pair (the
graphs carry an internal blank conditioning node instead), so the app's
``negative_prompt`` field is intentionally not injected — sending the
generic app negative would fight the pack's distilled schedules.

Quality presets map to pack profiles, not step counts: ``draft`` uses the
FAST_PREVIEW graph (8 sampler steps) while ``balanced`` and ``quality``
use TWO_STAGE_PRODUCTION (8 + 3 = 11 steps, decoding at 2x the base
resolution). Step count and guidance are FIXED by the distilled sigma
schedules baked into the graphs, so user ``steps``/``guidance`` overrides
are intentionally ignored — :meth:`resolve_quality` reports the effective
values so job metadata stays truthful. Any of the ten profiles can be
forced with ``quality_params["ltx_profile"]`` (e.g. the experimental
``pixel_spatial`` IC-LoRA finishing pass, which the pack flags as
generative enhancement — opt-in only).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple

from .base import (ModelAdapter, GenerationParams, SectionConfig,
                   WORKFLOW_DIR)

log = logging.getLogger("simple_comfort.adapter.ltx25")


def _ltx_frame_count(seconds: float, fps: int) -> int:
    """Round to the nearest LTX-friendly frame count (8*k + 1)."""
    raw = max(int(round(seconds * fps)), 9)
    k = round((raw - 1) / 8)
    return int(k * 8 + 1)


# ---------------------------------------------------------------------- #
# The AAN LTX-2.5 candidate pack (workflows/aan_ltx25/)
# ---------------------------------------------------------------------- #
PACK_OBJECT_INFO = "aan_ltx25/pack_object_info.json"

# Profile -> pack workflow (relative to the repo workflows/ directory).
# Keys are the stable public names accepted via quality_params["ltx_profile"].
PROFILES: Dict[str, str] = {
    "audio_first":    "aan_ltx25/AAN_LTX25_01_AUDIO_FIRST_ANCHOR.json",
    "character_lock": "aan_ltx25/AAN_LTX25_02_CHARACTER_LOCK_ANCHOR.json",
    "studio":         "aan_ltx25/AAN_LTX25_03_STUDIO_ANCHOR.json",
    "dynamic":        "aan_ltx25/AAN_LTX25_04_DYNAMIC_ANCHOR.json",
    "t2v_broll":      "aan_ltx25/AAN_LTX25_05_T2V_BROLL.json",
    "i2v_broll":      "aan_ltx25/AAN_LTX25_06_I2V_BROLL.json",
    "audio_timed":    "aan_ltx25/AAN_LTX25_07_AUDIO_TIMED_BROLL.json",
    "fast_preview":   "aan_ltx25/AAN_LTX25_08_FAST_PREVIEW.json",
    "two_stage":      "aan_ltx25/AAN_LTX25_09_TWO_STAGE_PRODUCTION.json",
    "pixel_spatial":  "aan_ltx25/AAN_LTX25_10_MASTER_PIXEL_SPATIAL.json",
    # The CCTech "MAX face-lock" graph — the user's own tuned two-stage
    # build (8-step base + 3-step refine, decoding at 2x). Identity is
    # anchored by a dedicated face/bust reference through
    # CCTechLTX25ReferenceConditioning on the BASE pass only, so chained
    # sections keep the same face while the first frame carries motion.
    "facelock":       "ltx25_facelock/LTX25_MAX_FACELOCK_CCTECH_2STAGE.json",
}

# Quality preset -> default profile. The face-lock graph is the production
# backend; only ``draft`` opts out with the single-stage FAST_PREVIEW
# (the pack's own promotion sequence starts there at short durations).
QUALITY_PROFILES: Dict[str, str] = {
    "draft": "fast_preview",
    "balanced": "facelock",
    "quality": "facelock",
}

# Graphs whose stage 2 latent-upscales x2 before decoding: the working
# grid is half the requested final resolution.
TWO_X_DECODE_PROFILES = frozenset({"facelock", "two_stage", "pixel_spatial"})

# Per-profile section length (seconds). The face-lock graph is built for
# ~10s continuous shots (its Duration control caps at 20s) and its CCTech
# identity reference holds the face across chains, so it wants fewer,
# longer sections than the pack's 6s default.
SECTION_SECONDS_BY_PROFILE: Dict[str, float] = {
    "facelock": 10.0,
}
# Hard ceiling a profile's section length may never exceed (the graph's
# own Duration control maximum).
SECTION_CEILING_BY_PROFILE: Dict[str, float] = {
    "facelock": 20.0,
}

# Interface control-node title prefixes (the pack README's COMMON
# INTERFACE contract). Each must match exactly one node per workflow.
CTRL_PROMPT = "SCRIPT / VIDEO REQUEST"
CTRL_USE_IMAGE = "Use Image Conditioning"
CTRL_REF_SOURCE = "Reference Source"
CTRL_USE_AUDIO = "Use Uploaded Master Audio"
CTRL_DURATION = "Duration in seconds"
CTRL_FPS = "FPS"
CTRL_WIDTH = "Working Width"
CTRL_HEIGHT = "Working Height"
CTRL_AVATAR_IMAGE = "AVATAR / CHARACTER"
CTRL_SCENE_IMAGE = "SCENE / B-ROLL IMAGE"
CTRL_AUDIO = "MASTER AUDIO"
CTRL_SEED = "SEED"

# Every control title prefix of the contract (each must resolve to exactly
# one node per workflow — pinned by the pack test suite).
CTRL_IMAGE_CONTROLS: Tuple[str, ...] = (
    CTRL_PROMPT, CTRL_USE_IMAGE, CTRL_REF_SOURCE, CTRL_USE_AUDIO,
    CTRL_DURATION, CTRL_FPS, CTRL_WIDTH, CTRL_HEIGHT, CTRL_AVATAR_IMAGE,
    CTRL_SCENE_IMAGE, CTRL_AUDIO, CTRL_SEED,
)

# Per-profile renames of the interface contract: profiles whose graphs
# titled a contract slot differently. Keys are contract prefixes, values
# the profile's actual title prefix.
PROFILE_CONTROL_OVERRIDES: Dict[str, Dict[str, str]] = {
    # The face-lock graph calls the conditioning subject FIRST FRAME.
    "facelock": {CTRL_AVATAR_IMAGE: "FIRST FRAME"},
}

# Extra face-lock-only interface controls.
CTRL_IDENTITY_IMAGE = "IDENTITY REFERENCE"
CTRL_REF_STRENGTH = "Reference / Avatar Strength"

# Seed widget modes the ComfyUI editor serialises after seed-like widgets
# (same set as custom_workflows.SEED_MODES).
_SEED_MODES = {"fixed", "increment", "decrement", "randomize"}

# Control node types that carry their value in widgets_values[0].
_VALUE_TYPES = {"PrimitiveStringMultiline", "PrimitiveBoolean",
                "PrimitiveFloat", "PrimitiveInt", "LoadImage", "LoadAudio"}

_object_info_cache: Optional[Dict[str, Any]] = None


def _pack_object_info() -> Dict[str, Any]:
    """Load (once) the pinned node-spec fixture used for offline
    conversion. Captured from the live ComfyUI 0.34.0 ``/object_info``,
    trimmed to the pack's node types; file-listing combo options are
    stubbed so no local filenames travel with the repo."""
    global _object_info_cache
    if _object_info_cache is None:
        path = WORKFLOW_DIR / PACK_OBJECT_INFO
        _object_info_cache = json.loads(
            path.read_text(encoding="utf-8"))
    return _object_info_cache


def _is_dynamic_combo(itype: Any, cfg: Dict[str, Any]) -> bool:
    if itype in ("COMFY_DYNAMICCOMBO_V3", "COMBO3D"):
        return True
    options = cfg.get("options")
    return isinstance(options, list) and options \
        and all(isinstance(o, dict) and "key" in o for o in options)


def _widgetable(iname: str, ispec: Any) -> bool:
    """Mirror convert_ui_to_api's widget-ability rule for one input spec,
    plus scalar union types ("FLOAT,INT") which the editor also renders
    as widgets (they occupy a slot in the author's full-order array)."""
    if not isinstance(ispec, list) or not ispec:
        return False
    cfg = ispec[1] if len(ispec) > 1 and isinstance(ispec[1], dict) else {}
    if cfg.get("forceInput"):
        return False
    itype = ispec[0]
    if _is_dynamic_combo(itype, cfg):
        return True  # the selection key itself is a widget
    if isinstance(itype, list):
        return True  # classic combo
    if isinstance(itype, str) and itype and \
            all(p in ("STRING", "INT", "FLOAT", "BOOLEAN", "COMBO")
                for p in itype.split(",")):
        return True
    return False


def _normalize_widgets(ui: Dict[str, Any],
                       object_info: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite pack widget arrays into the editor-canonical form.

    The pack stores ``widgets_values`` in FULL input-spec order — wired
    inputs keep their (stale) value in the array. The editor (and the
    converter) read them with wired slots absent, so every value after a
    wired input would land one slot off. This walks the spec order,
    consumes the author's full array, and keeps only the values of
    inputs that are NOT wired to links (seed "control" suffixes
    preserved). Nodes whose array cannot be consumed exactly are left
    untouched (the converter then falls back to sequential mapping) and
    logged — the pack test suite asserts that never happens.
    """
    for node in ui.get("nodes") or []:
        ctype = node.get("type", "")
        if ctype in ("Note", "MarkdownNote", "Reroute"):
            continue
        widgets = node.get("widgets_values")
        if not isinstance(widgets, list) or not widgets:
            continue
        spec = (object_info.get(ctype) or {}).get("input", {})
        ordered: List[Tuple[str, Any]] = [
            (iname, ispec)
            for section in ("required", "optional")
            for iname, ispec in (spec.get(section) or {}).items()]
        connected = {slot.get("name") for slot in node.get("inputs") or []
                     if slot.get("link") is not None}

        widget_slots = [iname for iname, ispec in ordered
                        if _widgetable(iname, ispec)]
        wi = 0
        keep: List[Any] = []

        def take() -> Any:
            nonlocal wi
            if wi < len(widgets):
                val = widgets[wi]
                wi += 1
                return val
            return None

        for iname, ispec in ordered:
            cfg = ispec[1] if isinstance(ispec, list) and len(ispec) > 1 \
                and isinstance(ispec[1], dict) else {}
            itype = ispec[0] if isinstance(ispec, list) and ispec else None
            if _is_dynamic_combo(itype, cfg):
                value = take()
                option = None
                for opt in cfg.get("options") or []:
                    if isinstance(opt, dict) and opt.get("key") == value:
                        option = opt
                        break
                if iname not in connected and value is not None:
                    keep.append(value)
                nested = (option or {}).get("inputs") or {}
                for section in ("required", "optional"):
                    for sub_name in nested.get(section) or {}:
                        dotted = f"{iname}.{sub_name}"
                        # the author keeps a slot for wired nested too
                        value = take()
                        if dotted not in connected and value is not None:
                            keep.append(value)
                continue
            if not _widgetable(iname, ispec):
                continue  # connection-only: no author slot
            value = take()
            extra = None
            has_cag = bool(cfg.get("control_after_generate")
                           or cfg.get("control_before_generate"))
            if (has_cag or (itype == "INT"
                            and iname.lower().endswith("seed"))) \
                    and wi < len(widgets) and widgets[wi] in _SEED_MODES:
                extra = widgets[wi]
                wi += 1
            if iname not in connected:
                keep.append(value)
                if extra is not None:
                    keep.append(extra)

        if wi == len(widgets):
            node["widgets_values"] = keep
        elif len(widget_slots) == 1:
            # Upload-style nodes carry trailing artefacts after the one
            # real value (e.g. LoadImage's ["name", "image"]).
            node["widgets_values"] = widgets[:1]
        else:
            log.warning("pack widgets not normalisable: %s#%s "
                        "(%d values, %d consumed)", ctype, node.get("id"),
                        len(widgets), wi)
    return ui


def _blank_png() -> bytes:
    """A deterministic 64x64 black PNG (placeholder for unused uploads).

    64x64, not 1x1: ComfyUI >= 0.34 decodes uploads through PyAV, and
    libavfilter rejects degenerate 1x1 frames with an opaque
    ``av.error.ArgumentError`` at execution time (queue-time validation
    still passes, so only a real render catches it)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    size = 64
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)
    # Grayscale rows: one filter byte (0) + `size` black pixels each.
    idat = zlib.compress(b"\x00" * (size * (size + 1)))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def _silence_wav() -> bytes:
    """A deterministic 0.1s silent WAV (placeholder for unused uploads)."""
    n = 800  # 8000 Hz, 16-bit mono -> 0.1 s
    data = b"\x00\x00" * n
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


class LTX25Adapter(ModelAdapter):
    MODEL_NAME = "LTX 2.5"
    MODEL_KEY = "ltx25"
    SUPPORTED_MODES = {"t2v", "i2v"}
    TARGET_SECONDS = 30

    # Union of executable node types across the pack graphs + face-lock
    # (MarkdownNote is editor-only and never appears in /object_info).
    REQUIRED_NODES = [
        "UNETLoader", "CLIPLoader", "CLIPTextEncode", "VAELoader",
        "TextGenerateLTX2Prompt", "LTXVConditioning", "EmptyLTXVLatentVideo",
        "LTXVImgToVideoInplace", "LTXVPreprocess", "LTXVConcatAVLatent",
        "LTXVSeparateAVLatent", "SamplerCustomAdvanced", "ManualSigmas",
        "VAEDecodeTiled", "CreateVideo", "SaveVideo", "ComfyMathExpression",
        "ComfySwitchNode", "PrimitiveStringMultiline", "PrimitiveBoolean",
        "PrimitiveFloat", "PrimitiveInt", "RandomNoise", "LoadImage",
        "LoadAudio", "TrimAudioDuration", "LTXVAudioVAEEncode",
        "LTXVAudioVAEDecode", "LTXVEmptyLatentAudio", "SolidMask",
        "SetLatentNoiseMask", "PreviewAny", "CFGGuider", "KSamplerSelect",
        "LTXVLatentUpsampler", "LatentUpscaleModelLoader",
        "LTXAddVideoICLoRAGuide", "LTXICLoRALoaderModelOnly",
        "LTXVCropGuides",
        # face-lock profile (ComfyUI-GGUF-Loader custom pack)
        "CCTechLTX25ReferenceConditioning", "LTXVDualCFGGuider",
    ]
    CHECKPOINT_HINTS = ["ltx-2.5"]
    VAE_HINTS = ["ltx-2.5"]
    TEXT_ENCODER_HINTS = ["gemma"]

    # Canonical template names (profile selection happens in
    # _select_profile; mode is an in-graph toggle, not a separate file).
    T2V_TEMPLATE = PROFILES["t2v_broll"]
    I2V_TEMPLATE = PROFILES["i2v_broll"]
    T2V_TEMPLATE_2STAGE = PROFILES["facelock"]
    I2V_TEMPLATE_2STAGE = PROFILES["facelock"]

    # Official LTX-2.5 base resolution is 960x544 (16:9); multiples of 32.
    ASPECT_RESOLUTIONS: Dict[str, tuple] = {
        "16:9": (960, 544),
        "9:16": (544, 960),
        "1:1": (768, 768),
    }

    # ~6s sections keep peak memory bounded while chaining stays smooth.
    SECTION_SECONDS = 6.0

    # Effective sampler budget fixed by the distilled sigma schedules.
    SINGLE_STAGE_STEPS = 8       # sigmas: 1.0 ... 0.0 (9 values)
    TWO_STAGE_STEPS = 11         # 8 (stage 1) + 3 (stage 2 from sigma 0.85)
    PIXEL_SPATIAL_STEPS = 16     # 8 (base) + 8 (pixel IC-LoRA pass)
    DISTILLED_GUIDANCE = 1.0     # CFGGuider cfg used by every stage

    # ------------------------------------------------------------------ #
    def get_section_strategy(self, params: GenerationParams) -> List[SectionConfig]:
        fps = params.fps or 24
        target = params.effective_target_seconds(self.TARGET_SECONDS)
        profile = self._select_profile(params)
        section_seconds = params.segment_length or \
            SECTION_SECONDS_BY_PROFILE.get(profile, self.SECTION_SECONDS)
        ceiling = SECTION_CEILING_BY_PROFILE.get(profile)
        if ceiling is not None:
            # The face-lock graph's Duration control caps at 20s and is
            # tuned for ~10s continuous shots — the UI's default 5s slider
            # value would needlessly double the chain seams. Clamp into the
            # graph's designed window instead of taking 5s literally.
            section_seconds = min(max(section_seconds, 10.0), ceiling)
        num_sections = max(1, round(target / section_seconds))
        # Without an explicit segment length, distribute the target exactly
        # across sections — otherwise e.g. a 20s target yields 3 x 6s =
        # 18.2s. Sections stay within ~1.5x SECTION_SECONDS so the peak
        # memory bound of the section length still holds. With an explicit
        # segment length, respect it as given.
        per_section = section_seconds
        if not params.segment_length:
            while target / num_sections > section_seconds * 1.5:
                num_sections += 1
            per_section = target / num_sections
        else:
            # An explicit (or clamped) section length never stretches a
            # shorter target — a 4s b-roll request must not render 10s.
            per_section = min(per_section, target / num_sections)
        frames = _ltx_frame_count(per_section, fps)
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
        """Working-grid resolution (multiples of 32; EmptyLTXVLatentVideo
        validates against a step of 32).

        ``params.width/height`` carry the user's FINAL target dimensions.
        x2-decoding profiles (two-stage refine) render the base grid at
        half that and let the stage-2 latent upsampler double it back;
        the exact final dimensions are then landed by the app's
        scale/crop finish (ffmpeg_utils.scale_video).
        """
        if params.width and params.height:
            width, height = float(params.width), float(params.height)
            if self._select_profile(params) in TWO_X_DECODE_PROFILES:
                width, height = width / 2.0, height / 2.0
            width = max(64, int(round(width / 32)) * 32)
            height = max(64, int(round(height / 32)) * 32)
            return width, height
        return self.ASPECT_RESOLUTIONS.get(params.aspect_ratio, (960, 544))

    def _select_profile(self, params: GenerationParams) -> str:
        """Quality preset -> pack profile, overridable via quality_params."""
        requested = (params.quality_params or {}).get("ltx_profile")
        if requested:
            if requested not in PROFILES:
                raise ValueError(
                    f"Unknown LTX-2.5 profile '{requested}' — expected one "
                    f"of: {', '.join(sorted(PROFILES))}")
            return requested
        return QUALITY_PROFILES.get(params.quality, "two_stage")

    def resolve_quality(self, params: GenerationParams) -> Dict[str, float]:
        """Report the EFFECTIVE steps/guidance of the chosen graph variant.

        The distilled sigma schedules fix both, so user overrides are
        deliberately ignored (reporting them would be a lie in metadata).
        """
        profile = self._select_profile(params)
        steps = {
            "two_stage": self.TWO_STAGE_STEPS,
            "facelock": self.TWO_STAGE_STEPS,   # 8-step base + 3-step refine
            "pixel_spatial": self.PIXEL_SPATIAL_STEPS,
        }.get(profile, self.SINGLE_STAGE_STEPS)
        return {"steps": steps, "guidance": self.DISTILLED_GUIDANCE}

    # ------------------------------------------------------------------ #
    # Interface patching (pack README "COMMON INTERFACE" contract)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _control(ui: Dict[str, Any], title_prefix: str,
                 types: Optional[set] = None) -> Optional[Dict[str, Any]]:
        """The unique node whose title starts with ``title_prefix``."""
        matches = [n for n in ui.get("nodes") or []
                   if (n.get("title") or "").startswith(title_prefix)
                   and (types is None or n.get("type") in types)]
        if len(matches) != 1:
            raise ValueError(
                f"pack interface control {title_prefix!r} matched "
                f"{len(matches)} nodes (expected exactly 1)")
        return matches[0]

    def _patch_interface(self, ui: Dict[str, Any], values: Dict[str, Any],
                         seeds: Dict[int, int]) -> None:
        """Set the control nodes' widget values on a normalised UI copy."""
        for prefix, value in values.items():
            node = self._control(ui, prefix, _VALUE_TYPES)
            node["widgets_values"][0] = value
        for node in ui.get("nodes") or []:
            if node.get("type") != "RandomNoise":
                continue
            seed = seeds.get(node.get("id"))
            if seed is None:
                continue
            node["widgets_values"][0] = seed
            if len(node["widgets_values"]) > 1:
                node["widgets_values"][1] = "fixed"

    @staticmethod
    def _ensure_input_asset(name: str, content: bytes) -> str:
        """Idempotently place a deterministic asset in the ComfyUI input
        dir and return its bare name (same staging convention as
        reference images — LoadImage/LoadAudio reject absolute paths)."""
        inp = os.environ.get("COMFY_INPUT_DIR", "/data/comfyui/input")
        os.makedirs(inp, exist_ok=True)
        dest = os.path.join(inp, name)
        if not os.path.exists(dest):
            with open(dest, "wb") as fh:
                fh.write(content)
        return name

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

    # ------------------------------------------------------------------ #
    def map_parameters(self, params: GenerationParams, section: SectionConfig,
                       last_frame_path: Optional[str]) -> Dict[str, Any]:
        profile = self._select_profile(params)
        width, height = self.resolve_resolution(params)
        resolved = params.resolved_prompt or self.resolve_prompt(params)
        seed = params.seed if params.seed is not None else 0
        section_seed = seed + section.index  # vary per section

        # Which conditioning image applies to this section.
        image_path = last_frame_path if not section.is_first \
            else params.reference_image
        staged = self._stage_reference_image(image_path) \
            if section.use_reference else None

        # Unused upload slots still need a VALID filename: ComfyUI combo
        # validation checks membership even for nodes the lazy switches
        # never execute.
        blank = self._ensure_input_asset("sc-blank.png", _blank_png())
        silence = self._ensure_input_asset("sc-silence.wav", _silence_wav())
        fill = staged or blank

        ui = self.load_workflow(PROFILES[profile])
        ui = _normalize_widgets(ui, _pack_object_info())

        # Deterministic seeds: the contract seed node anchors the section;
        # any extra noise sources (stage-2 / pixel pass) draw independently.
        seeds: Dict[int, int] = {}
        extras = 0
        for node in ui.get("nodes") or []:
            if node.get("type") == "RandomNoise":
                if (node.get("title") or "").startswith(CTRL_SEED):
                    seeds[node["id"]] = section_seed
                else:
                    extras += 1
                    seeds[node["id"]] = section_seed + extras

        # The face-lock graph keeps a SECOND reference — the identity
        # face/bust — that never changes across chained sections: the
        # first frame carries motion continuity, the identity image is
        # the face authority for every section (that is the whole point
        # of the CCTech reference conditioning). Note the graph's
        # "Reference / Avatar Strength" control is a DIFFERENT knob (it
        # scales the first-frame i2v injection and keeps its tuned 0.7);
        # the identity strength is the CCTech node's own widget, patched
        # post-conversion below.
        identity = None
        if profile == "facelock" and params.reference_image:
            identity = self._stage_reference_image(params.reference_image)

        controls = {
            CTRL_PROMPT: resolved,
            # One reference image is the conditioning subject: the app's
            # reference drives the avatar slot (the pack's scene slot /
            # uploaded-audio path stay off — audio is generated jointly,
            # the pack's documented OFF behaviour).
            CTRL_USE_IMAGE: bool(staged),
            CTRL_REF_SOURCE: True,
            CTRL_USE_AUDIO: False,
            CTRL_DURATION: round(section.approx_seconds, 6),
            CTRL_FPS: float(section.fps),
            CTRL_WIDTH: width,
            CTRL_HEIGHT: height,
            CTRL_AVATAR_IMAGE: fill,
            CTRL_SCENE_IMAGE: fill,
            CTRL_AUDIO: silence,
        }
        # Apply the profile's contract renames, then add profile extras.
        for old, new in PROFILE_CONTROL_OVERRIDES.get(profile, {}).items():
            controls[new] = controls.pop(old)
        if profile == "facelock":
            controls[CTRL_IDENTITY_IMAGE] = identity or fill

        self._patch_interface(ui, controls, seeds)
        # Imported here (not at module top) to avoid a cycle: custom_workflows
        # imports adapters.base.
        from custom_workflows import convert_ui_to_api
        prompt = convert_ui_to_api(ui, _pack_object_info())
        if profile == "facelock":
            for node in prompt.values():
                if node.get("class_type") == "SaveVideo":
                    # Keep Simple-Comfort renders out of the user's own
                    # research output namespace
                    # (video/LTX25_MAX_FACELOCK_RESEARCH).
                    node["inputs"]["filename_prefix"] = "SimpleComfort/ltx25"
                if node.get("class_type") == "CCTechLTX25ReferenceConditioning":
                    # Without an identity image the node's strength drops
                    # to 0.0 — its documented clean bypass — so the blank
                    # placeholder is never encoded as a face authority.
                    node["inputs"]["strength"] = 1.0 if identity else 0.0
        return self._apply_mgpu(prompt)

    # ------------------------------------------------------------------ #
    # Multi-GPU: split ONE generation across every installed GPU
    # ------------------------------------------------------------------ #
    MGPU_NODE = "MultiGPU_WorkUnits"

    @staticmethod
    def _mgpu_enabled() -> bool:
        """Split-GPU mode is armed by the launch-split-gpu.sh launcher
        (SC_SPLIT_GPUS=1 in .env). Off by default so hermetic tests and
        single-GPU installs keep exact old behaviour."""
        return os.environ.get("SC_SPLIT_GPUS", "").lower() \
            in ("1", "true", "on", "yes")

    def _apply_mgpu(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Insert MultiGPU_WorkUnits so a single generation's model work
        is shared evenly by every GPU (one cond unit + one uncond unit per
        step -> GPU 0 takes one, GPU 1 the other).

        Two pieces of mechanics make this work for the distilled graph:

        * MultiGPU_WorkUnits must wrap the MODEL *after* model-mutating
          nodes — for the face-lock graph that is after the CCTech
          reference conditioning (base pass) and straight off the loader
          (refine pass) — exactly where this hook wires them.
        * The LTXVDualCFGGuider collapses to single-CFG whenever both
          scales are equal (1.0/1.0 here) and ComfyUI's cfg==1 shortcut
          then SKIPS the uncond pass, leaving only one work unit and
          nothing to split. Setting audio_cfg a hair above video_cfg
          forces the dual path (``disable_cfg1_optimization``) so the
          uncond pass runs too. The blend is uncond + (cond-uncond)*cfg,
          so video output is mathematically identical at video_cfg 1.0
          and audio differs by 1e-4 relative — inaudible.

        Graphs without an LTXVDualCFGGuider (the pack profiles use a
        plain CFGGuider at cfg 1.0) are returned untouched: there is no
        second work unit to divide, and faking one would only burn FLOPs.
        """
        if not self._mgpu_enabled():
            return prompt
        guiders = {nid: node for nid, node in prompt.items()
                   if node.get("class_type") == "LTXVDualCFGGuider"}
        if not guiders:
            return prompt
        wrapped: Dict[tuple, str] = {}
        for node in guiders.values():
            src = node["inputs"].get("model")
            if not isinstance(src, list) or len(src) != 2:
                continue
            key = (src[0], src[1])
            if key not in wrapped:
                mgid = f"mgpu{len(wrapped)}"
                prompt[mgid] = {
                    "class_type": self.MGPU_NODE,
                    "inputs": {"model": [src[0], src[1]], "max_gpus": 2},
                    "_meta": {"title": "MultiGPU CFG Split"},
                }
                wrapped[key] = mgid
            node["inputs"]["model"] = [wrapped[key], 0]
            node["inputs"]["video_cfg"] = 1.0
            node["inputs"]["audio_cfg"] = 1.0001
        return prompt
