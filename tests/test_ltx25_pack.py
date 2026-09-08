"""AAN LTX-2.5 candidate pack tests (hermetic: no ComfyUI, no network).

The ten pack workflows under ``workflows/aan_ltx25/`` are the app's LTX-2.5
backend set. These tests pin the three properties that make them safe to
drive blind:

* **Byte fidelity** — the repo copies match the pack's SHA256_MANIFEST.json.
* **Interface contract** — every COMMON INTERFACE control title (pack
  README.txt) resolves to exactly one node in every workflow.
* **Offline conversion** — ``LTX25Adapter.map_parameters`` converts the
  editor-format graphs to API prompts using only the pinned spec fixture
  (``pack_object_info.json``, captured from ComfyUI 0.34.0), fixing the
  widget-order landmines the hand-authored files contain (full-spec-order
  ``widgets_values`` arrays).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adapters import LTX25Adapter
from adapters.base import GenerationParams
from adapters.ltx25 import (CTRL_IMAGE_CONTROLS, PROFILES,
                            _normalize_widgets, _pack_object_info)

PACK_DIR = Path(__file__).resolve().parent.parent / "workflows" / "aan_ltx25"
MANIFEST = json.loads((PACK_DIR / "SHA256_MANIFEST.json").read_text())
ADAPTER = LTX25Adapter()


def pack_ui(profile: str) -> dict:
    """A freshly-parsed UI graph for one profile (byte-faithful file)."""
    from adapters.base import WORKFLOW_DIR
    return json.loads((WORKFLOW_DIR / PROFILES[profile])
                      .read_text(encoding="utf-8"))


def built_prompt(profile: str, **overrides) -> dict:
    """map_parameters output for one profile (t2v by default)."""
    params = GenerationParams(prompt="a calm ocean at dawn", mode="t2v",
                              quality="balanced", lip_sync=False, seed=42,
                              fps=24, quality_params={"ltx_profile": profile},
                              **overrides)
    section = ADAPTER.get_section_strategy(params)[0]
    return ADAPTER.map_parameters(params, section, None)


def nodes_of(prompt: dict, class_type: str) -> list:
    return [n for n in prompt.values() if n["class_type"] == class_type]


# ---------------------------------------------------------------------- #
# Byte fidelity
# ---------------------------------------------------------------------- #
def test_pack_copies_match_sha256_manifest():
    assert len(MANIFEST) == 10
    for name, rec in sorted(MANIFEST.items()):
        data = (PACK_DIR / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == rec["sha256"], name


def test_pack_docs_travel_with_the_workflows():
    for doc in ("README.txt", "MODEL_REQUIREMENTS.txt",
                "SHA256_MANIFEST.json", "BUILD_STANDARD_EXCERPT.txt",
                "TEST_REPORT.md", "pack_object_info.json"):
        assert (PACK_DIR / doc).is_file(), doc


# ---------------------------------------------------------------------- #
# Interface contract (pack README "COMMON INTERFACE")
# ---------------------------------------------------------------------- #
def test_interface_controls_unique_in_every_workflow():
    from adapters.ltx25 import PROFILE_CONTROL_OVERRIDES
    for profile in PROFILES:
        ui = pack_ui(profile)
        overrides = PROFILE_CONTROL_OVERRIDES.get(profile, {})
        for prefix in CTRL_IMAGE_CONTROLS:
            effective = overrides.get(prefix, prefix)
            matches = [n for n in ui["nodes"]
                       if (n.get("title") or "").startswith(effective)]
            assert len(matches) == 1, \
                f"{profile}: control {effective!r} matched {len(matches)} nodes"


def test_required_nodes_union_equals_pack_types():
    union: set = set()
    for profile in PROFILES:
        union |= {n["type"] for n in pack_ui(profile)["nodes"]
                  if n["type"] not in ("Note", "MarkdownNote", "Reroute")}
    assert set(ADAPTER.REQUIRED_NODES) == union
    assert "MarkdownNote" not in union  # editor-only, never in /object_info


def test_pinned_fixture_covers_every_pack_node_type():
    fixture = _pack_object_info()
    for profile in PROFILES:
        for node in pack_ui(profile)["nodes"]:
            if node["type"] in ("Note", "MarkdownNote", "Reroute"):
                continue
            spec = fixture.get(node["type"], {})
            assert spec.get("input", {}).get("required") is not None, \
                f"{profile}: {node['type']} missing from pinned fixture"


# ---------------------------------------------------------------------- #
# Widget normalisation + offline conversion
# ---------------------------------------------------------------------- #
def test_normalization_fixes_widget_order_landmines():
    """The pack stores widgets in full spec order; the canonical arrays the
    converter expects must drop wired slots — otherwise batch_size eats
    width, a float lands on the boolean bypass, etc."""
    fixture = _pack_object_info()
    for profile in PROFILES:
        ui = _normalize_widgets(pack_ui(profile), fixture)
        for node in ui["nodes"]:
            if node["type"] == "EmptyLTXVLatentVideo":
                assert node["widgets_values"] == [1], profile  # batch_size
            if node["type"] == "LTXVEmptyLatentAudio":
                assert node["widgets_values"] == [1], profile
            if node["type"] == "LTXVImgToVideoInplace":
                # bypass is the trailing boolean (the primary conditioning
                # node has strength wired away, the hi-res reapply keeps it)
                assert node["widgets_values"][-1] is False, profile
            if node["type"] == "LatentUpscaleModelLoader":
                assert node["widgets_values"] == \
                    ["ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"]
            if node["type"] == "LTXICLoRALoaderModelOnly":
                assert node["widgets_values"] == \
                    ["ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
                     1.0]


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_converted_prompts_are_wellformed(profile):
    prompt = built_prompt(profile)
    assert prompt and all(isinstance(n, dict) and n.get("class_type")
                          and isinstance(n.get("inputs"), dict) and n["inputs"]
                          for n in prompt.values())
    for node in nodes_of(prompt, "TextGenerateLTX2Prompt"):
        inp = node["inputs"]
        assert isinstance(inp["max_length"], int)
        assert isinstance(inp["sampling_mode"], str)
        assert isinstance(inp["thinking"], bool)
        assert inp["prompt"] == "a calm ocean at dawn"
        assert isinstance(inp["sampling_mode.seed"], int)
    for node in nodes_of(prompt, "ComfyMathExpression"):
        if "floor" not in node["inputs"].get("expression", ""):
            continue  # pixel stage's width*2 helper
        assert "values.a" in node["inputs"]   # fps
        assert "values.b" in node["inputs"]   # duration
    latents = nodes_of(prompt, "EmptyLTXVLatentVideo")
    for node in latents:
        assert node["inputs"]["batch_size"] == 1
    # The base latent carries the 16:9 default resolution (the pixel-stage
    # latent derives its width from the in-graph width*2 math node).
    assert any(n["inputs"].get("width") == 960 and
               n["inputs"].get("height") == 544 for n in latents)
    for node in nodes_of(prompt, "LoadAudio"):
        assert node["inputs"]["audio"] == "sc-silence.wav"
    for node in nodes_of(prompt, "LoadImage"):
        assert node["inputs"]["image"] == "sc-blank.png"  # t2v placeholder


def test_stage2_and_pixel_noise_seeds_are_deterministic():
    prompt = built_prompt("two_stage")
    seeds = sorted(n["inputs"]["noise_seed"]
                   for n in nodes_of(prompt, "RandomNoise"))
    assert seeds == [42, 43]  # contract seed + independent stage-2 draw
    prompt = built_prompt("pixel_spatial")
    seeds = sorted(n["inputs"]["noise_seed"]
                   for n in nodes_of(prompt, "RandomNoise"))
    assert seeds == [42, 43]


def test_duration_fps_and_seed_reach_the_graph():
    params = GenerationParams(prompt="x", mode="t2v", quality="balanced",
                              lip_sync=False, seed=7, fps=24,
                              target_seconds=6, segment_length=6.0)
    section = ADAPTER.get_section_strategy(params)[0]
    prompt = ADAPTER.map_parameters(params, section, None)
    math = nodes_of(prompt, "ComfyMathExpression")[0]["inputs"]
    assert math["values.a"] == 24.0          # FPS
    assert math["values.b"] == pytest.approx(6.041667)  # duration seconds
    # Frame count = 1 + floor(24*6/8)*8 = 145 (8k+1, matches section.frames)
    assert section.frames == 145
    noise = nodes_of(prompt, "RandomNoise")[0]["inputs"]
    assert noise["noise_seed"] == 7


def test_i2v_stages_reference_and_toggles_conditioning(reference_image):
    params = GenerationParams(prompt="a news anchor", mode="i2v",
                              quality="balanced", lip_sync=False, seed=1,
                              fps=24, reference_image=str(reference_image),
                              quality_params={"ltx_profile": "audio_first"})
    section = ADAPTER.get_section_strategy(params)[0]
    prompt = ADAPTER.map_parameters(params, section, None)
    images = [n["inputs"]["image"] for n in nodes_of(prompt, "LoadImage")]
    assert any(str(v).startswith("sc-") and v != "sc-blank.png"
               for v in images), images
    switches = {n["_meta"]["title"]: n["inputs"]["switch"]
                for n in prompt.values() if n["class_type"] == "ComfySwitchNode"}
    assert switches["T2V / I2V Latent Select"] is True       # i2v branch
    assert switches["Audio Latent — Generate / Uploaded Master"] is False
    assert switches["Final Audio — Generated / Original Uploaded"] is False


def test_quality_presets_map_to_pack_profiles():
    for quality, expected in (("draft", "fast_preview"),
                              ("balanced", "facelock"),
                              ("quality", "facelock")):
        params = GenerationParams(prompt="x", mode="t2v", quality=quality)
        assert ADAPTER._select_profile(params) == expected
    assert ADAPTER.resolve_quality(
        GenerationParams(prompt="x", mode="t2v", quality="draft"))["steps"] \
        == ADAPTER.SINGLE_STAGE_STEPS
    assert ADAPTER.resolve_quality(
        GenerationParams(prompt="x", mode="t2v", quality="balanced"))["steps"] \
        == ADAPTER.TWO_STAGE_STEPS
    assert ADAPTER.resolve_quality(GenerationParams(
        prompt="x", mode="t2v", quality="balanced",
        quality_params={"ltx_profile": "pixel_spatial"}))["steps"] \
        == ADAPTER.PIXEL_SPATIAL_STEPS


def test_unknown_profile_is_rejected():
    params = GenerationParams(prompt="x", mode="t2v", quality="balanced",
                              quality_params={"ltx_profile": "nope"})
    with pytest.raises(ValueError, match="Unknown LTX-2.5 profile"):
        ADAPTER._select_profile(params)


def test_all_ten_profiles_are_reachable():
    for profile in PROFILES:
        prompt = built_prompt(profile)
        assert nodes_of(prompt, "SaveVideo"), profile
        assert nodes_of(prompt, "UNETLoader"), profile
