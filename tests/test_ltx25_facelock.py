"""Face-lock backend tests (hermetic: no ComfyUI, no network).

``workflows/ltx25_facelock/LTX25_MAX_FACELOCK_CCTECH_2STAGE.json`` is the
user's own CCTech two-stage LTX-2.5 graph and the production backend for
``balanced``/``quality``. These tests pin the behaviours that make it safe
to drive blind: byte fidelity to the exported file, identity-reference
persistence across chained sections, the no-identity bypass, the halved
working grid for x2-decode, and the split-GPU work-unit insertion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adapters import LTX25Adapter
from adapters.base import GenerationParams
from adapters.ltx25 import PROFILES

ADAPTER = LTX25Adapter()
FACELOCK_DIR = (Path(__file__).resolve().parent.parent / "workflows"
                / "ltx25_facelock")
# Pinned SHA256 of the exported graph (desktop original).
FACELOCK_SHA256 = "48d07ea5fdc945b30fa30b49dcfa7b469d70de2210d740ab880d66a6d9e25183"


def tiny_png(path: Path) -> str:
    """Write a deterministic 1x1 black PNG and return its path."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", idat) + chunk(b"IEND", b""))
    return str(path)


def facelock_params(**overrides) -> GenerationParams:
    base = dict(prompt="a news presenter speaks to camera", mode="t2v",
                aspect_ratio="9:16", quality="balanced", lip_sync=False,
                seed=42, fps=24, target_seconds=10, segment_length=5.0)
    base.update(overrides)
    return GenerationParams(**base)


def built(prompt: dict, class_type: str) -> list:
    return [n for n in prompt.values() if n["class_type"] == class_type]


def nodes_of_type(prompt: dict, class_type: str) -> list:
    return built(prompt, class_type)


# ---------------------------------------------------------------------- #
# Byte fidelity
# ---------------------------------------------------------------------- #
def test_facelock_file_matches_exported_original():
    data = (FACELOCK_DIR / "LTX25_MAX_FACELOCK_CCTECH_2STAGE.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == FACELOCK_SHA256
    assert PROFILES["facelock"] == \
        "ltx25_facelock/LTX25_MAX_FACELOCK_CCTECH_2STAGE.json"


def test_facelock_uses_cctech_reference_on_base_pass_only():
    params = facelock_params()
    prompt = ADAPTER.map_parameters(
        params, ADAPTER.get_section_strategy(params)[0], None)
    cct_ids = [nid for nid, n in prompt.items()
               if n["class_type"] == "CCTechLTX25ReferenceConditioning"]
    assert len(cct_ids) == 1
    cct_id = cct_ids[0]
    loader_ids = [nid for nid, n in prompt.items()
                  if n["class_type"] == "UNETLoader"]
    assert len(loader_ids) == 1
    guider_models = {tuple(g["inputs"]["model"])
                     for g in nodes_of_type(prompt, "LTXVDualCFGGuider")}
    # Stage 1 samples the CCTech-patched model...
    assert (cct_id, 0) in guider_models
    # ...while the refine pass rides the raw loader (a reference riding a
    # refine pass at a new grid is exactly what the pack warns against).
    assert (loader_ids[0], 0) in guider_models


# ---------------------------------------------------------------------- #
# Identity persistence + bypass
# ---------------------------------------------------------------------- #
def test_identity_reference_persists_across_chained_sections(tmp_path):
    ref = tiny_png(tmp_path / "avatar.png")
    params = facelock_params(mode="i2v", reference_image=ref,
                             target_seconds=20)
    sections = ADAPTER.get_section_strategy(params)
    assert len(sections) == 2
    last_frame = str(tmp_path / "section_00_last.png")
    tiny_png(Path(last_frame))

    first = ADAPTER.map_parameters(params, sections[0], None)
    second = ADAPTER.map_parameters(params, sections[1], last_frame)

    def identity_name(prompt: dict) -> str:
        cct = nodes_of_type(prompt, "CCTechLTX25ReferenceConditioning")[0]
        img_node = prompt[cct["inputs"]["image"][0]]
        assert img_node["class_type"] == "LoadImage"
        title = img_node["_meta"]["title"]
        assert title.startswith("IDENTITY REFERENCE")
        return img_node["inputs"]["image"]

    def first_frame_name(prompt: dict) -> str:
        node = [n for n in prompt.values()
                if n["class_type"] == "LoadImage"
                and (n["_meta"]["title"] or "").startswith("FIRST FRAME")][0]
        return node["inputs"]["image"]

    id0, id1 = identity_name(first), identity_name(second)
    ff0, ff1 = first_frame_name(first), first_frame_name(second)
    # Identity authority NEVER changes across sections...
    assert id0 == id1 and id0.startswith("sc-") and id0 != "sc-blank.png"
    # ...while the first frame follows the chain.
    assert ff0 != ff1
    # Strength stays engaged on every section.
    for prompt in (first, second):
        cct = nodes_of_type(prompt, "CCTechLTX25ReferenceConditioning")[0]
        assert cct["inputs"]["strength"] == 1.0


def test_no_identity_image_bypasses_cctech_cleanly(tmp_path):
    params = facelock_params(mode="t2v")  # no reference image
    prompt = ADAPTER.map_parameters(
        params, ADAPTER.get_section_strategy(params)[0], None)
    cct = nodes_of_type(prompt, "CCTechLTX25ReferenceConditioning")[0]
    # strength 0.0 is the node's documented clean bypass — a blank
    # placeholder must never be encoded as a face authority.
    assert cct["inputs"]["strength"] == 0.0


# ---------------------------------------------------------------------- #
# Resolution presets (720p / 1080p final targets)
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("final,w,h", [
    ((720, 1280), 352, 640),    # 720p portrait
    ((1280, 720), 640, 352),    # 720p landscape
    ((1080, 1920), 544, 960),   # 1080p portrait
    ((1920, 1080), 960, 544),   # 1080p landscape
])
def test_resolution_presets_halve_for_x2_decode(final, w, h):
    params = facelock_params(width=final[0], height=final[1])
    prompt = ADAPTER.map_parameters(
        params, ADAPTER.get_section_strategy(params)[0], None)
    latent = nodes_of_type(prompt, "EmptyLTXVLatentVideo")[0]["inputs"]
    # Working grid is half the final target, snapped to multiples of 32
    # (stage 2 latent-upscales x2; the app's ffmpeg finish lands the exact
    # final dimensions afterwards).
    assert (latent["width"], latent["height"]) == (w, h)


def test_draft_profile_renders_at_full_grid():
    # Single-stage FAST_PREVIEW has no x2 decode: the requested grid IS the
    # render grid (snapped to 32; 720 -> 704 via banker's rounding — the
    # ffmpeg finish covers the last few percent).
    params = facelock_params(quality="draft", width=720, height=1280)
    prompt = ADAPTER.map_parameters(
        params, ADAPTER.get_section_strategy(params)[0], None)
    latent = nodes_of_type(prompt, "EmptyLTXVLatentVideo")[0]["inputs"]
    assert (latent["width"], latent["height"]) == (704, 1280)


def test_output_is_namespaced_out_of_research_dir():
    params = facelock_params()
    prompt = ADAPTER.map_parameters(
        params, ADAPTER.get_section_strategy(params)[0], None)
    savers = nodes_of_type(prompt, "SaveVideo")
    assert len(savers) == 1
    assert savers[0]["inputs"]["filename_prefix"] == "SimpleComfort/ltx25"


# ---------------------------------------------------------------------- #
# Split-GPU work units
# ---------------------------------------------------------------------- #
def test_mgpu_off_by_default(monkeypatch):
    monkeypatch.delenv("SC_SPLIT_GPUS", raising=False)
    params = facelock_params()
    prompt = ADAPTER.map_parameters(
        params, ADAPTER.get_section_strategy(params)[0], None)
    assert not nodes_of_type(prompt, "MultiGPU_WorkUnits")
    for g in nodes_of_type(prompt, "LTXVDualCFGGuider"):
        assert g["inputs"]["audio_cfg"] == 1.0  # untouched graph values


def test_mgpu_inserts_work_units_and_nudges_audio_cfg(monkeypatch):
    monkeypatch.setenv("SC_SPLIT_GPUS", "1")
    params = facelock_params()
    prompt = ADAPTER.map_parameters(
        params, ADAPTER.get_section_strategy(params)[0], None)

    cct_ids = [nid for nid, n in prompt.items()
               if n["class_type"] == "CCTechLTX25ReferenceConditioning"]
    loader_ids = [nid for nid, n in prompt.items()
                  if n["class_type"] == "UNETLoader"]
    mgpu = {nid: n for nid, n in prompt.items()
            if n["class_type"] == "MultiGPU_WorkUnits"}
    guiders = nodes_of_type(prompt, "LTXVDualCFGGuider")

    # One wrapper per distinct model source: AFTER the model-mutating
    # CCTech node (base pass) and straight off the loader (refine pass).
    sources = {tuple(n["inputs"]["model"]) for n in mgpu.values()}
    assert sources == {(cct_ids[0], 0), (loader_ids[0], 0)}

    # Every guider consumes a wrapper and carries the audio_cfg nudge
    # that keeps the uncond pass alive (mathematically identical video,
    # 1e-4 relative on audio) so cond+uncond form two divisible work units.
    for g in guiders:
        assert g["inputs"]["model"][0] in mgpu
        assert g["inputs"]["video_cfg"] == 1.0
        assert g["inputs"]["audio_cfg"] == pytest.approx(1.0001)


def test_mgpu_leaves_pack_profiles_alone(monkeypatch):
    monkeypatch.setenv("SC_SPLIT_GPUS", "1")
    params = facelock_params(
        quality_params={"ltx_profile": "two_stage"})
    prompt = ADAPTER.map_parameters(
        params, ADAPTER.get_section_strategy(params)[0], None)
    # Pack graphs use a plain CFGGuider at cfg 1.0: there is no second
    # work unit to divide, so the hook must be a no-op there.
    assert not nodes_of_type(prompt, "MultiGPU_WorkUnits")
