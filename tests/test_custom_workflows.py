"""Custom workflow import tests: UI->API conversion, validation-error
parsing and registry legacy-record loading. All hermetic — conversion uses an
inline object_info, so no ComfyUI is ever contacted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

import custom_workflows as cw
from custom_workflows import (CustomWorkflowError, CustomWorkflowRegistry,
                              convert_ui_to_api, detect_format,
                              parse_validation_errors)

BAR_WORKFLOW = (Path(__file__).resolve().parents[1] / "gauntlet" / "bar" /
                "lightx2v_t2v_4step_NativeComfy.json")


# ---------------------------------------------------------------------- #
# Inline node definitions (mirrors ComfyUI /object_info, minimal but with
# the real required/optional input order so widget mapping is exercised).
# ---------------------------------------------------------------------- #
def _combo(*options: str):
    return [list(options), {}]


FI = {"forceInput": True}

BAR_OBJECT_INFO: Dict[str, Any] = {
    "UNETLoader": {"input": {"required": {
        "unet_name": _combo("wan2.2_t2v_high_noise_14B_fp16.safetensors",
                            "wan2.2_t2v_low_noise_14B_fp16.safetensors"),
        "weight_dtype": _combo("default", "fp8_e4m3fn"),
    }}},
    "CLIPLoader": {"input": {"required": {
        "clip_name": _combo("umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        "type": _combo("wan", "sd3", "flux"),
    }, "optional": {"device": _combo("default", "cpu")}}},
    "VAELoader": {"input": {"required": {
        "vae_name": _combo("wan_2.1_vae.safetensors"),
    }}},
    "CLIPTextEncode": {"input": {"required": {
        "text": ["STRING", {"multiline": True}],
        "clip": ["CLIP", FI],
    }}},
    "ModelSamplingSD3": {"input": {"required": {
        "model": ["MODEL", FI],
        "shift": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}],
    }}},
    "LoraLoaderModelOnly": {"input": {"required": {
        "lora_name": _combo("high_noise_model.safetensors",
                            "low_noise_model.safetensors"),
        "strength_model": ["FLOAT", {"default": 1.0}],
        "model": ["MODEL", FI],
    }}},
    "KSamplerAdvanced": {"input": {"required": {
        "add_noise": _combo("enable", "disable"),
        "noise_seed": ["INT", {"default": 0,
                               "control_after_generate": True}],
        "steps": ["INT", {"default": 20}],
        "cfg": ["FLOAT", {"default": 8.0}],
        "sampler_name": _combo("euler", "uni_pc"),
        "scheduler": _combo("simple", "karras"),
        "start_at_step": ["INT", {"default": 0}],
        "end_at_step": ["INT", {"default": 10000}],
        "return_with_leftover_noise": _combo("enable", "disable"),
        "model": ["MODEL", FI],
        "positive": ["CONDITIONING", FI],
        "negative": ["CONDITIONING", FI],
        "latent_image": ["LATENT", FI],
    }}},
    "EmptyHunyuanLatentVideo": {"input": {"required": {
        "width": ["INT", {"default": 1280}],
        "height": ["INT", {"default": 720}],
        "length": ["INT", {"default": 81}],
        "batch_size": ["INT", {"default": 1}],
    }}},
    "VAEDecode": {"input": {"required": {
        "samples": ["LATENT", FI], "vae": ["VAE", FI],
    }}},
    "CreateVideo": {"input": {"required": {
        "images": ["IMAGE", FI],
        "frame_rate": ["INT", {"default": 16, "min": 1, "max": 512}],
    }, "optional": {"audio": ["AUDIO", FI]}}},
    "SaveVideo": {"input": {"required": {
        "filename_prefix": ["STRING", {"default": "video"}],
        "format": _combo("mp4", "webm", "gif"),
        "codec": _combo("h264", "h265", "av1"),
    }}},
}

# A minimal editor-format graph: CLIPLoader -> CLIPTextEncode, plus a Note.
MINI_UI: Dict[str, Any] = {
    "nodes": [
        {"id": 1, "type": "CLIPLoader", "mode": 0,
         "inputs": [], "outputs": [],
         "widgets_values": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                            "wan", "default"]},
        {"id": 2, "type": "CLIPTextEncode", "mode": 0,
         "inputs": [{"name": "clip", "type": "CLIP", "link": 5}],
         "outputs": [],
         "widgets_values": ["a calm sunrise"]},
        {"id": 3, "type": "Note", "mode": 0,
         "inputs": [], "outputs": [],
         "widgets_values": ["editor-only annotation"]},
    ],
    "links": [[5, 1, 0, 2, 0, "CLIP"]],
}


# ---------------------------------------------------------------------- #
# Format detection + UI -> API conversion
# ---------------------------------------------------------------------- #
def test_detect_format_classifies_ui_and_api():
    assert detect_format(MINI_UI) == "ui"
    api = {"1": {"class_type": "VAELoader", "inputs": {"vae_name": "v"}}}
    assert detect_format(api) == "api"
    with pytest.raises(CustomWorkflowError):
        detect_format({"foo": "bar"})


def test_convert_ui_to_api_inline_fixture():
    prompt = convert_ui_to_api(MINI_UI, BAR_OBJECT_INFO)
    assert set(prompt) == {"1", "2"}  # the Note is editor-only, dropped
    node1, node2 = prompt["1"], prompt["2"]
    assert node1["class_type"] == "CLIPLoader"
    assert node1["inputs"]["clip_name"] == "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    assert node1["inputs"]["type"] == "wan"
    assert node2["class_type"] == "CLIPTextEncode"
    # Wired input becomes [origin_node_id, origin_slot]; widget becomes value.
    assert node2["inputs"]["clip"] == ["1", 0]
    assert node2["inputs"]["text"] == "a calm sunrise"
    assert node2["_meta"]["title"]


def test_convert_real_bar_ui_workflow():
    ui = json.loads(BAR_WORKFLOW.read_text(encoding="utf-8"))
    assert detect_format(ui) == "ui"
    prompt = convert_ui_to_api(ui, BAR_OBJECT_INFO)

    assert prompt, "conversion produced an empty prompt"
    types = {node["class_type"] for node in prompt.values()}
    assert "Note" not in types and "MarkdownNote" not in types
    for node_id, node in prompt.items():
        assert node["class_type"] in BAR_OBJECT_INFO, node_id
        assert isinstance(node["inputs"], dict)

    # Every wired input resolved to [origin_node_id, origin_slot].
    for node in prompt.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert len(value) == 2 and value[0] in prompt

    # Widget order is honoured, and the seed-control widget ("fixed")
    # trailing noise_seed is swallowed rather than mapped onto `steps`.
    ks = next(n for n in prompt.values()
              if n["class_type"] == "KSamplerAdvanced"
              and n["inputs"]["noise_seed"] == 42)
    assert ks["inputs"]["steps"] == 4
    assert ks["inputs"]["cfg"] == 1.0
    assert ks["inputs"]["sampler_name"] == "euler"
    assert ks["inputs"]["start_at_step"] == 0
    assert ks["inputs"]["end_at_step"] == 2

    latent = next(n for n in prompt.values()
                  if n["class_type"] == "EmptyHunyuanLatentVideo")
    assert latent["inputs"] == {"width": 1280, "height": 720,
                                "length": 81, "batch_size": 1}
    lora = next(n for n in prompt.values()
                if n["class_type"] == "LoraLoaderModelOnly")
    assert lora["inputs"]["strength_model"] == pytest.approx(1.0)


def test_convert_subgraph_workflow_raises_with_api_format_guidance():
    subgraph_ui = {
        "nodes": [{"id": 1, "type": "MySubgraph", "mode": 0,
                   "inputs": [], "outputs": [], "widgets_values": []}],
        "links": [],
        "definitions": {"subgraphs": [{"id": "MySubgraph",
                                       "nodes": [], "links": []}]},
    }
    assert detect_format(subgraph_ui) == "ui"
    with pytest.raises(CustomWorkflowError) as excinfo:
        convert_ui_to_api(subgraph_ui, BAR_OBJECT_INFO)
    msg = str(excinfo.value).lower()
    assert "subgraph" in msg
    assert "api format" in msg  # guidance: re-export with Save (API format)


def test_convert_unknown_custom_node_names_the_class():
    ui = {"nodes": [{"id": 9, "type": "DefinitelyNotInstalled", "mode": 0,
                     "inputs": [], "outputs": [], "widgets_values": []}],
          "links": []}
    with pytest.raises(CustomWorkflowError, match="DefinitelyNotInstalled"):
        convert_ui_to_api(ui, BAR_OBJECT_INFO)


# ---------------------------------------------------------------------- #
# Validation-error parsing
# ---------------------------------------------------------------------- #
def test_parse_validation_errors_maps_node_errors_to_strings():
    body = json.dumps({
        "error": {"type": "prompt_outputs_failed_validation",
                  "message": "Prompt outputs failed validation"},
        "node_errors": {
            "37": {"class_type": "UNETLoader", "errors": [
                {"type": "value_not_in_list", "message": "Value not in list",
                 "details": "unet_name: 'missing.safetensors' not in list"},
            ]},
            "57": {"class_type": "KSamplerAdvanced", "errors": [
                {"type": "value_mismatch_int", "message": "Value not in range",
                 "details": "steps: 4 not in range"},
            ]},
        },
    })
    errors = parse_validation_errors(body)
    assert errors == [
        "UNETLoader #37: unet_name: 'missing.safetensors' not in list",
        "KSamplerAdvanced #57: steps: 4 not in range",
    ]


def test_parse_validation_errors_falls_back_gracefully():
    # No node_errors -> top-level error message; non-JSON -> raw text.
    errors = parse_validation_errors(
        json.dumps({"error": {"message": "prompt is empty"}}))
    assert errors == ["prompt is empty"]
    raw = parse_validation_errors("<html>Bad Gateway</html>")
    assert raw == ["<html>Bad Gateway</html>"]


# ---------------------------------------------------------------------- #
# Registry: legacy manifest records
# ---------------------------------------------------------------------- #
LEGACY_API_WORKFLOW = {
    "1": {"class_type": "VAELoader",
          "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
          "_meta": {"title": "Load VAE"}},
    "2": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "a calm sunrise", "clip": ["1", 0]},
          "_meta": {"title": "Prompt"}},
}

@pytest.fixture
def legacy_store(tmp_path: Path, monkeypatch):
    """Point the registry at a temp store with legacy manifest records
    (imported before the validated/validated_at fields existed)."""
    store = tmp_path / "custom_workflows"
    store.mkdir()
    (store / "legacy-a.json").write_text(
        json.dumps(LEGACY_API_WORKFLOW), encoding="utf-8")
    (store / "legacy-b.json").write_text(
        json.dumps({"1": {"class_type": "VAELoader",
                          "inputs": {"vae_name": "v.safetensors"}}}),
        encoding="utf-8")
    manifest = {
        # No schema at all (schema introspection must regenerate it).
        "legacy-a": {"slug": "legacy-a", "name": "Legacy A",
                     "file": "legacy-a.json", "format": "api",
                     "imported_at": "2026-01-01T00:00:00"},
        # Schema present but no validation verdict fields.
        "legacy-b": {"slug": "legacy-b", "name": "Legacy B",
                     "file": "legacy-b.json", "format": "api",
                     "imported_at": "2026-01-02T00:00:00",
                     "schema": {"fields": [], "node_count": 1,
                                "output_nodes": [], "has_image_input": False}},
    }
    (store / "manifest.json").write_text(json.dumps(manifest),
                                         encoding="utf-8")
    monkeypatch.setattr(cw, "STORE_DIR", store)
    monkeypatch.setattr(cw, "MANIFEST_PATH", store / "manifest.json")
    return store


def test_registry_loads_legacy_records_without_validation_fields(legacy_store):
    adapters: dict = {}
    registry = CustomWorkflowRegistry(adapters)
    registry.load_all()

    assert set(registry.records) == {"legacy-a", "legacy-b"}
    for slug in ("legacy-a", "legacy-b"):
        record = registry.get(slug)
        assert record["name"]
        adapter = registry.get_adapter(slug)
        assert adapter is not None
        # No verdict recorded -> "not yet validated", never "invalid".
        assert adapter.validated is None
        assert adapter.validation_errors == []

    # legacy-a had no schema: one is introspected from the workflow itself.
    adapter_a = registry.get_adapter("legacy-a")
    assert adapter_a.schema["node_count"] == 2
    assert any(f["input"] == "text" for f in adapter_a.schema["fields"])

    # Offline capability check keeps legacy workflows usable.
    status = adapter_a.detect_capabilities(object_info={}, model_files={})
    assert status.available is True
    assert "not yet validated" in status.detail


def test_registry_survives_unreadable_manifest(tmp_path: Path,
                                               monkeypatch):
    store = tmp_path / "custom_workflows"
    store.mkdir()
    (store / "manifest.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(cw, "STORE_DIR", store)
    monkeypatch.setattr(cw, "MANIFEST_PATH", store / "manifest.json")
    registry = CustomWorkflowRegistry({})
    registry.load_all()
    assert registry.records == {}


# ---------------------------------------------------------------------- #
# ComfyUI >= 0.34 schema shapes (string "COMBO", dotted dynamic inputs)
# ---------------------------------------------------------------------- #
NEW_SCHEMA_OBJECT_INFO: Dict[str, Any] = {
    # New-style string-typed combo (options live in the config dict).
    "LoadAudio": {"input": {"required": {
        "audio": ["COMBO", {"options": ["narration.mp3", "music.wav"]}],
    }}},
    "PrimitiveFloat": {"input": {"required": {
        "value": ["FLOAT", {"default": 1.0}],
    }}},
    "PrimitiveInt": {"input": {"required": {
        "value": ["INT", {"default": 0,
                          "control_after_generate": "fixed"}],
    }}},
    # Autogrow numeric inputs arrive as dotted slot names ("values.a").
    "ComfyMathExpression": {"input": {"required": {
        "expression": ["STRING", {"default": "a + b", "multiline": True}],
        "values": ["COMFY_AUTOGROW_V3",
                   {"template": {"input": {"required": {
                       "value": ["FLOAT,INT,BOOLEAN", {}],
                   }}}}],
    }}},
    # Dynamic combo with nested inputs (nested slots named "mode.seed"...)
    "Enhancer": {"input": {"required": {
        "prompt": ["STRING", {"multiline": True}],
        "mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "on", "inputs": {"required": {
                "temperature": ["FLOAT", {"default": 0.7}],
                "seed": ["INT", {"default": 0}],
            }, "optional": {
                "presence_penalty": ["FLOAT", {"default": 0.0}],
            }}},
            {"key": "off", "inputs": {}},
        ]}],
    }}},
}


def _ui_node(nid, ctype, widgets, inputs=(), title=""):
    return {"id": nid, "type": ctype, "title": title,
            "widgets_values": list(widgets),
            "inputs": [{"name": n, "link": lk} for n, lk in inputs],
            "outputs": [{"name": "OUT", "links": []}]}


def test_string_typed_combo_consumes_widget_value():
    ui = {"nodes": [_ui_node(1, "LoadAudio", ["narration.mp3"])],
          "links": []}
    api = convert_ui_to_api(ui, NEW_SCHEMA_OBJECT_INFO)
    assert api["1"]["inputs"] == {"audio": "narration.mp3"}


def test_combo_without_widget_value_is_omitted_not_none():
    # A short widget array must not inject None into a combo input
    # (validation would report "None not in list").
    ui = {"nodes": [_ui_node(1, "LoadAudio", [])], "links": []}
    api = convert_ui_to_api(ui, NEW_SCHEMA_OBJECT_INFO)
    assert "audio" not in api["1"]["inputs"]


def test_autogrow_dotted_inputs_are_emitted_under_full_names():
    # values.a wired to a Primitive, values.b left unconnected (widget).
    ui = {"nodes": [
        _ui_node(1, "ComfyMathExpression", ["a + b"],
                 inputs=[("values.a", 7)]),
        _ui_node(2, "PrimitiveFloat", [24.0]),
    ], "links": [[7, 2, 0, 1, 0, "FLOAT,INT,BOOLEAN"]]}
    api = convert_ui_to_api(ui, NEW_SCHEMA_OBJECT_INFO)
    assert api["1"]["inputs"]["expression"] == "a + b"
    assert api["1"]["inputs"]["values.a"] == 24.0  # inlined Primitive value


def test_dynamic_combo_nested_inputs_use_dotted_names():
    # Editor-canonical widget array: no slot for the wired "mode.seed".
    ui = {"nodes": [
        _ui_node(1, "Enhancer", ["hello", "on", 0.5, 0.25],
                 inputs=[("mode.seed", 7)]),
        _ui_node(2, "PrimitiveInt", [1234]),
    ], "links": [[7, 2, 0, 1, 3, "INT"]]}
    api = convert_ui_to_api(ui, NEW_SCHEMA_OBJECT_INFO)
    inp = api["1"]["inputs"]
    assert inp["prompt"] == "hello"
    assert inp["mode"] == "on"
    assert inp["mode.temperature"] == 0.5
    assert inp["mode.seed"] == 1234          # wired Primitive, not a widget
    assert inp["mode.presence_penalty"] == 0.25  # optional nested input
