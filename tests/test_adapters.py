"""Adapter registry and workflow-template tests (hermetic: no ComfyUI).

Covers, per model adapter (ltx25 / wan22 / framepack):

* ``build_adapters()`` registers exactly the three built-in engines;
* every workflow template JSON parses and contains the loader / decode /
  save nodes the engine requires — editor-format templates (the AAN LTX-2.5
  pack) are checked as UI graphs against the pinned spec fixture, API-format
  templates as prompt dicts;
* building a prompt from a typical ``GenerationParams`` yields an API-format
  dict whose nodes are well-formed (``class_type`` + non-empty ``inputs``)
  and fully substituted (no ``{{PLACEHOLDER}}`` tokens remain anywhere).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List

import pytest

from adapters import (FramePackAdapter, GenerationParams, LTX25Adapter,
                      Wan22Adapter, build_adapters)
from adapters.base import ModelAdapter, WORKFLOW_DIR
from adapters.framepack import MAX_TOTAL_SECONDS as FRAMEPACK_MAX_SECONDS
from adapters.ltx25 import PROFILES

# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def class_types(prompt: Dict[str, Any]) -> set:
    """The set of node class_type values in an API-format prompt."""
    return {node["class_type"] for node in prompt.values()
            if isinstance(node, dict)}


def iter_strings(value: Any) -> Iterator[str]:
    """Yield every string value anywhere inside a nested dict/list."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_strings(v)


def assert_nodes_wellformed(prompt: Dict[str, Any]) -> None:
    """Every node is a dict with a class_type and a non-empty inputs dict."""
    assert isinstance(prompt, dict) and prompt
    for node_id, node in prompt.items():
        assert isinstance(node, dict), f"node {node_id} is not a dict"
        assert node.get("class_type"), f"node {node_id} missing class_type"
        inputs = node.get("inputs")
        assert isinstance(inputs, dict) and inputs, \
            f"node {node_id} has no inputs"


def assert_prompt_wellformed(prompt: Dict[str, Any]) -> None:
    """A BUILT prompt: well-formed nodes and no placeholder left behind."""
    assert_nodes_wellformed(prompt)
    for text in iter_strings(prompt):
        assert "{{" not in text, f"unsubstituted placeholder in: {text!r}"


def assert_template_nodes(adapter: ModelAdapter, template: str,
                          required: set) -> set:
    """Load a template and assert the required class_types are present.

    Raw templates legitimately contain ``{{TOKENS}}``; only their structure
    is checked here — substitution is asserted on built prompts."""
    workflow = adapter.load_workflow(template)
    types = class_types(workflow)
    missing = required - types
    assert not missing, f"{template} missing nodes: {sorted(missing)}"
    assert_nodes_wellformed(workflow)
    return types


def build_all_sections(adapter: ModelAdapter, params: GenerationParams):
    """Map every section of a strategy; yields (section, workflow)."""
    for section in adapter.get_section_strategy(params):
        yield section, adapter.map_parameters(params, section, None)


def typical_params(**overrides: Any) -> GenerationParams:
    """A plain, fully-populated generation request."""
    base = dict(
        prompt="a slow pan across a quiet kitchen at dawn",
        mode="t2v",
        aspect_ratio="16:9",
        quality="balanced",
        lip_sync=True,
        seed=42,
        fps=24,
    )
    base.update(overrides)
    return GenerationParams(**base)


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #
def test_build_adapters_registers_all_three_engines():
    adapters = build_adapters()
    assert set(adapters) == {"ltx25", "wan22", "framepack"}
    assert isinstance(adapters["ltx25"], LTX25Adapter)
    assert isinstance(adapters["wan22"], Wan22Adapter)
    assert isinstance(adapters["framepack"], FramePackAdapter)


# ---------------------------------------------------------------------- #
# LTX 2.5 (AAN candidate pack — editor-format graphs)
# ---------------------------------------------------------------------- #
LTX_REQUIRED = {"UNETLoader", "CLIPLoader", "VAELoader",
                "TextGenerateLTX2Prompt", "VAEDecodeTiled", "SaveVideo"}

def ui_node_types(workflow: Dict[str, Any]) -> set:
    """Executable class types in an editor-format (UI) workflow."""
    return {n["type"] for n in workflow["nodes"]
            if n["type"] not in ("Note", "MarkdownNote", "Reroute")}

@pytest.mark.parametrize("template", sorted(PROFILES.values()))
def test_ltx25_pack_templates_parse_with_required_nodes(template):
    workflow = LTX25Adapter().load_workflow(template)
    assert isinstance(workflow.get("nodes"), list) and workflow["nodes"]
    assert isinstance(workflow.get("links"), list)
    types = ui_node_types(workflow)
    missing = LTX_REQUIRED - types
    assert not missing, f"{template} missing nodes: {sorted(missing)}"
    # Every executable type must resolve against the pinned spec fixture
    # (MarkdownNote is editor-only and exempt).
    fixture = json.loads((WORKFLOW_DIR / "aan_ltx25" /
                          "pack_object_info.json").read_text())
    unknown = types - set(fixture)
    assert not unknown, f"{template} types missing from fixture: {sorted(unknown)}"


def test_ltx25_prompt_substitutes_all_sections():
    adapter = LTX25Adapter()
    sections = list(build_all_sections(adapter, typical_params()))
    # 30s target on the facelock backend = 3 chained ~10s sections
    # (8k+1 frames each) — longer shots, fewer chain seams.
    assert len(sections) == 3
    for section, workflow in sections:
        assert section.frames % 8 == 1
        assert_prompt_wellformed(workflow)


def test_ltx25_quality_presets_select_graph_variant():
    adapter = LTX25Adapter()
    _, balanced = next(iter(build_all_sections(adapter,
                                               typical_params())))
    _, draft = next(iter(build_all_sections(adapter,
                                            typical_params(quality="draft"))))
    # Two-stage adds the latent upsampler branch; draft stays single-stage.
    assert "LatentUpscaleModelLoader" in class_types(balanced)
    assert "LatentUpscaleModelLoader" not in class_types(draft)
    assert adapter.resolve_quality(typical_params(quality="draft"))["steps"] \
        == adapter.SINGLE_STAGE_STEPS
    assert adapter.resolve_quality(typical_params())["steps"] \
        == adapter.TWO_STAGE_STEPS


def test_ltx25_i2v_stages_reference_image(reference_image):
    adapter = LTX25Adapter()
    params = typical_params(mode="i2v",
                            reference_image=str(reference_image))
    _, workflow = next(iter(build_all_sections(adapter, params)))
    assert "LoadImage" in class_types(workflow)
    staged = [s for s in iter_strings(workflow) if s.startswith("sc-")]
    assert staged, "reference image was not staged into the graph"


# ---------------------------------------------------------------------- #
# Wan 2.2
# ---------------------------------------------------------------------- #
WAN_COMMON = {"UNETLoader", "LoraLoaderModelOnly", "ModelSamplingSD3",
              "CLIPLoader", "VAELoader", "CLIPTextEncode", "KSamplerAdvanced",
              "VAEDecodeTiled", "CreateVideo", "SaveVideo"}

def test_wan22_templates_parse_with_required_nodes():
    t2v_types = assert_template_nodes(Wan22Adapter(), Wan22Adapter.T2V_TEMPLATE,
                                      WAN_COMMON)
    i2v_types = assert_template_nodes(Wan22Adapter(), Wan22Adapter.I2V_TEMPLATE,
                                      WAN_COMMON)
    assert "EmptyHunyuanLatentVideo" in t2v_types
    assert {"LoadImage", "WanImageToVideo"} <= i2v_types


def test_wan22_prompt_substitutes_all_sections():
    adapter = Wan22Adapter()
    sections = list(build_all_sections(adapter, typical_params()))
    # 30s target / 5s sections = 6 sections; 4n+1 frames at the native 16fps.
    assert len(sections) == 6
    for section, workflow in sections:
        assert section.fps == Wan22Adapter.NATIVE_FPS
        assert section.frames % 4 == 1
        assert_prompt_wellformed(workflow)


def test_wan22_i2v_uses_reference_template(reference_image):
    adapter = Wan22Adapter()
    params = typical_params(mode="i2v",
                            reference_image=str(reference_image))
    _, workflow = next(iter(build_all_sections(adapter, params)))
    assert {"LoadImage", "WanImageToVideo"} <= class_types(workflow)
    staged = [s for s in iter_strings(workflow) if s.startswith("sc-")]
    assert staged, "reference image was not staged into the graph"


# ---------------------------------------------------------------------- #
# FramePack
# ---------------------------------------------------------------------- #
FRAMEPACK_REQUIRED = {"LoadFramePackModel", "DualCLIPLoader", "VAELoader",
                      "LoadImage", "FramePackFindNearestBucket",
                      "FramePackSampler", "VAEDecodeTiled", "SaveVideo"}

def test_framepack_template_parses_with_required_nodes():
    assert_template_nodes(FramePackAdapter(), FramePackAdapter.I2V_TEMPLATE,
                          FRAMEPACK_REQUIRED)


def test_framepack_is_single_section_within_node_bounds(reference_image):
    adapter = FramePackAdapter()
    params = typical_params(mode="i2v",
                            reference_image=str(reference_image))
    sections = adapter.get_section_strategy(params)
    assert len(sections) == 1  # whole clip renders in one call
    assert sections[0].approx_seconds <= FRAMEPACK_MAX_SECONDS
    workflow = adapter.map_parameters(params, sections[0], None)
    assert_prompt_wellformed(workflow)
    assert "FramePackSampler" in class_types(workflow)


def test_framepack_requires_reference_image():
    adapter = FramePackAdapter()
    params = typical_params(mode="i2v", reference_image=None)
    section = adapter.get_section_strategy(params)[0]
    with pytest.raises(ValueError, match="reference image"):
        adapter.map_parameters(params, section, None)


# ---------------------------------------------------------------------- #
# Target duration (20s / 30s clips)
# ---------------------------------------------------------------------- #
def test_ltx25_default_30s_sections_unchanged():
    sections = LTX25Adapter().get_section_strategy(typical_params())
    assert len(sections) == 3
    total = sum(s.approx_seconds for s in sections)
    assert 29.5 <= total <= 30.5


def test_ltx25_20s_target_distributes_exactly():
    sections = LTX25Adapter().get_section_strategy(
        typical_params(target_seconds=20))
    assert len(sections) == 2
    total = sum(s.approx_seconds for s in sections)
    assert 19.5 <= total <= 20.5, f"20s target produced {total:.2f}s"
    # LTX frame counts must stay 8k+1 friendly per section.
    assert all((s.frames - 1) % 8 == 0 for s in sections)


def test_ltx25_explicit_segment_length_respected():
    # Pack profiles take an explicit segment length as-is (approximate
    # totals are the documented trade-off).
    sections = LTX25Adapter().get_section_strategy(
        typical_params(target_seconds=20, segment_length=6.0,
                       quality_params={"ltx_profile": "two_stage"}))
    assert len(sections) == 3
    # frames snap to 8k+1 (145 @ 24fps -> 6.04s), not exact 6.0
    assert all(5.9 <= s.approx_seconds <= 6.1 for s in sections)


def test_ltx25_facelock_clamps_segment_length_into_window():
    # The face-lock graph is built for ~10s shots with a 20s ceiling: the
    # UI's default 5s slider value must not double the chain seams, and a
    # shorter target must never be stretched.
    sections = LTX25Adapter().get_section_strategy(
        typical_params(target_seconds=20, segment_length=5.0))
    assert len(sections) == 2
    assert all(9.9 <= s.approx_seconds <= 10.1 for s in sections)
    # Explicit lengths inside the window pass through...
    sections = LTX25Adapter().get_section_strategy(
        typical_params(target_seconds=30, segment_length=15.0))
    assert len(sections) == 2
    assert all(14.9 <= s.approx_seconds <= 15.1 for s in sections)
    # ...above it they are capped...
    sections = LTX25Adapter().get_section_strategy(
        typical_params(target_seconds=30, segment_length=25.0))
    assert all(s.approx_seconds <= 20.1 for s in sections)
    # ...and a short target is never stretched to the floor.
    sections = LTX25Adapter().get_section_strategy(
        typical_params(target_seconds=4, segment_length=5.0))
    assert len(sections) == 1
    assert 3.9 <= sections[0].approx_seconds <= 4.1


def test_wan22_20s_target_four_exact_sections():
    sections = Wan22Adapter().get_section_strategy(
        typical_params(target_seconds=20))
    assert len(sections) == 4
    total = sum(s.approx_seconds for s in sections)
    assert 19.5 <= total <= 20.5
    assert all((s.frames - 1) % 4 == 0 for s in sections)


def test_framepack_20s_target_single_call():
    sections = FramePackAdapter().get_section_strategy(
        typical_params(mode="i2v", target_seconds=20))
    assert len(sections) == 1
    assert sections[0].approx_seconds == pytest.approx(20.0)


def test_duration_section_memory_bound():
    """Distributed sections never exceed ~1.5x the section-length bound."""
    # Pack graphs keep the 6s default bound.
    for dur, expect_sections in [(8, 1), (10, 2), (20, 3), (30, 5)]:
        sections = LTX25Adapter().get_section_strategy(
            typical_params(target_seconds=dur,
                           quality_params={"ltx_profile": "two_stage"}))
        assert len(sections) == expect_sections, (dur, len(sections))
        assert all(s.approx_seconds <= 6.0 * 1.5 + 0.1 for s in sections)
    # The face-lock backend (balanced/quality default) bounds at its own
    # 10s shot length instead.
    for dur, expect_sections in [(8, 1), (10, 1), (20, 2), (30, 3)]:
        sections = LTX25Adapter().get_section_strategy(
            typical_params(target_seconds=dur))
        assert len(sections) == expect_sections, (dur, len(sections))
        assert all(s.approx_seconds <= 10.0 * 1.5 + 0.1 for s in sections)
    # 4s (the API minimum) is a single short section.
    assert len(LTX25Adapter().get_section_strategy(
        typical_params(target_seconds=4))) == 1
