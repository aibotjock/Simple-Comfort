"""Custom workflow import: drag-and-drop ComfyUI JSON as a first-class engine.

Users drop a ComfyUI workflow export into the Workflows panel. The file is
parsed, optionally converted to API format, and introspected into a *form
schema* — the UI then rebuilds its controls from whatever the graph actually
contains (prompts, seeds, samplers, model pickers, image inputs...), so the
panel adapts to any workflow instead of the workflow having to fit the app.

Three pieces live here:

* :func:`detect_format` / :func:`convert_ui_to_api` — accept both the API
  export format and the editor (UI) format; UI graphs are converted using a
  live ``/object_info`` the same way ComfyUI's own frontend does it.
* :func:`introspect` — walk the API graph and emit a schema describing every
  user-settable scalar input, grouped by role.
* :class:`CustomWorkflowAdapter` / :class:`CustomWorkflowRegistry` — wrap an
  imported workflow in the normal :class:`ModelAdapter` contract so jobs run
  through the standard :class:`GenerationManager` pipeline unchanged.
* :func:`queue_validate_prompt` — check a converted prompt against a live
  ComfyUI using validate-and-discard (queue it, immediately delete it, never
  render) so imports and generations fail honestly instead of at render time.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from adapters.base import (GenerationParams, ModelAdapter, SectionConfig,
                           ModelStatus)

log = logging.getLogger("simple_comfort.custom_workflows")

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "custom_workflows"
MANIFEST_PATH = STORE_DIR / "manifest.json"

# Connection-only input types: they never consume a widget value and are
# skipped when building the form (they are graph wiring, not user settings).
CONNECTION_TYPES = {
    "IMAGE", "LATENT", "VAE", "MODEL", "CLIP", "CONDITIONING", "MASK",
    "CONTROL_NET", "SAMPLER", "SIGMAS", "SCHEDULER", "GUIDER", "NOISE",
    "AUDIO", "WEBCAM", "UPLOAD_VIDEO", "COMBO3D",
}

# Seed widget modes the ComfyUI frontend appends after seed-like widgets.
SEED_MODES = {"fixed", "increment", "decrement", "randomize"}

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".m4v", ".gif", ".avi"}


class CustomWorkflowError(ValueError):
    """Raised when an imported workflow cannot be parsed or converted."""


# ---------------------------------------------------------------------- #
# Format handling
# ---------------------------------------------------------------------- #
def detect_format(data: Any) -> str:
    """Classify a parsed workflow JSON as 'api', 'ui' or raise."""
    if not isinstance(data, dict) or not data:
        raise CustomWorkflowError("Workflow JSON must be a non-empty object.")
    if isinstance(data.get("nodes"), list) and isinstance(data.get("links"),
                                                           list):
        return "ui"
    values = [v for k, v in data.items() if not k.startswith("_")]
    if values and all(isinstance(v, dict) and "class_type" in v
                      for v in values):
        return "api"
    raise CustomWorkflowError(
        "Unrecognised workflow format — expected a ComfyUI 'Save (API "
        "format)' JSON or an editor graph export."
    )


def convert_ui_to_api(ui: Dict[str, Any],
                      object_info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an editor-format graph to the API prompt format.

    Mirrors ComfyUI's frontend graph-to-prompt logic: widget values are mapped
    onto inputs in ``required`` then ``optional`` order, skipping inputs that
    are wired to links; seed-like widgets swallow one extra "control" value.
    Muted (mode 2) and bypassed (mode 4) nodes are dropped, Reroutes are
    followed to their origin, and Primitive nodes feed their value directly
    into the inputs they drive. Annotation nodes (Note/MarkdownNote) are
    editor-only and skipped.

    ComfyUI >= 0.34 schema shapes are handled explicitly: ``["COMBO",
    {"options": ...}]`` string-typed combos consume a widget value like
    classic list combos, and dotted slot names (``values.a`` autogrow inputs,
    ``sampling_mode.seed`` dynamic-combo nested inputs) are emitted under
    their full dotted key — the server resolves those against the dynamic
    schema via ``get_finalized_class_inputs``.
    """
    links: Dict[int, list] = {}
    for link in ui.get("links") or []:
        # link := [id, origin_id, origin_slot, target_id, target_slot, type]
        if isinstance(link, (list, tuple)) and len(link) >= 5 \
                and link[0] is not None:
            links[int(link[0])] = link
    nodes_by_id = {n.get("id"): n for n in ui.get("nodes") or []}

    # Subgraph-typed nodes reference a nested graph definition; flattening
    # them is out of scope — ask for an API-format export instead.
    subgraph_ids = {
        sg.get("id") for sg in
        ((ui.get("definitions") or {}).get("subgraphs") or [])
        if isinstance(sg, dict) and sg.get("id")
    }

    def resolve_origin(link_id: int):
        """Follow a link through Reroutes to its origin.

        Returns ("node", [node_id, slot]) for a real node, ("value", v) when
        the chain ends at a Primitive node, or None if unresolvable.
        """
        for _ in range(128):  # reroute-chain guard
            link = links.get(int(link_id))
            if link is None:
                return None
            origin = nodes_by_id.get(link[1])
            if origin is None:
                return ("node", [str(link[1]), int(link[2])])
            otype = origin.get("type", "")
            if otype == "Reroute":
                nxt = (origin.get("inputs") or [{}])[0].get("link")
                if nxt is None:
                    return None
                link_id = int(nxt)
                continue
            if otype.startswith("Primitive"):
                vals = origin.get("widgets_values") or []
                val = vals[0] if vals else None
                if isinstance(val, dict):
                    val = val.get("value")
                return ("value", val)
            return ("node", [str(origin.get("id")), int(link[2])])
        return None

    prompt: Dict[str, Any] = {}
    missing_classes: List[str] = []

    for node in ui.get("nodes") or []:
        if node.get("mode", 0) in (2, 4):  # muted / bypassed
            continue
        ctype = node.get("type", "")
        if ctype in ("Note", "MarkdownNote", "Reroute"):
            continue  # editor-only annotation / passthrough
        if ctype in subgraph_ids:
            raise CustomWorkflowError(
                "This workflow uses ComfyUI subgraphs, which Simple-Comfort "
                "cannot convert. Open it in ComfyUI and use 'Save (API "
                "format)' to export a flat workflow, then drop that file."
            )
        spec = object_info.get(ctype)
        if spec is None:
            missing_classes.append(ctype)
            continue

        # Wire up connected inputs: name -> [origin_node_id, origin_slot],
        # plus values injected by Primitive nodes.
        connected: Dict[str, list] = {}
        primitive_values: Dict[str, Any] = {}
        for slot in node.get("inputs") or []:
            link_id = slot.get("link")
            if not slot.get("name") or link_id is None:
                continue
            resolved = resolve_origin(int(link_id))
            if resolved is None:
                continue
            if resolved[0] == "value":
                primitive_values[slot["name"]] = resolved[1]
            else:
                connected[slot["name"]] = resolved[1]

        input_spec = spec.get("input", {})
        widgets = list(node.get("widgets_values") or [])
        wi = 0
        inputs: Dict[str, Any] = {}

        def next_widget() -> Any:
            nonlocal wi
            if wi < len(widgets):
                val = widgets[wi]
                wi += 1
                return val
            return None

        def emit_input(iname: str, ispec: Any) -> None:
            """Map one input spec onto widget value(s), recursing through
            dynamic combos (COMBO3D), whose nested selections flatten into
            sibling inputs in API format."""
            nonlocal inputs, wi
            if iname in inputs:
                # Already satisfied — e.g. a nested dynamic-combo recursion
                # set it while its parent input was being consumed. Re-
                # emitting would consume another widget and overwrite it.
                return
            # Expanded (dotted) slots of dynamic inputs — autogrow vars
            # ("values.a") and dynamic-combo nested inputs
            # ("sampling_mode.seed") — are keyed by their full dotted name
            # both in UI slot lists and in the submitted prompt.
            prefix = iname + "."
            for dotted in [k for k in connected if k.startswith(prefix)]:
                inputs[dotted] = connected[dotted]
            for dotted in [k for k in primitive_values
                           if k.startswith(prefix)]:
                inputs[dotted] = primitive_values.pop(dotted)
            if iname in connected:
                inputs[iname] = connected[iname]
                return
            if iname in primitive_values:
                value = primitive_values.pop(iname)
                if isinstance(ispec, list) and ispec \
                        and ispec[0] == "INT" and isinstance(value, float):
                    value = int(value)
                inputs[iname] = value
                return
            if not isinstance(ispec, list) or not ispec:
                return
            itype = ispec[0]
            cfg = ispec[1] if len(ispec) > 1 and isinstance(ispec[1], dict)\
                else {}

            # Dynamic combo: options are dicts with "key" + nested "inputs".
            if _is_dynamic_combo(itype, cfg):
                value = next_widget()
                if value is None:
                    value = _dynamic_combo_default(itype, cfg)
                inputs[iname] = value
                option = _match_dynamic_option(itype, cfg, value)
                if option:
                    nested = option.get("inputs") or {}
                    for section in ("required", "optional"):
                        for sub_name, sub_spec in \
                                (nested.get(section) or {}).items():
                            emit_input(f"{iname}.{sub_name}", sub_spec)
                return

            # Classic list combos AND the >= 0.34 string-typed "COMBO"
            # (options in the config dict) both consume one widget value.
            is_combo = isinstance(itype, list) or itype == "COMBO"
            widget_able = (itype in ("STRING", "INT", "FLOAT", "BOOLEAN")
                           or is_combo)
            if not widget_able or cfg.get("forceInput"):
                return  # connection-only input left unwired
            value = next_widget()
            if value is None:
                # Ran out of widget values — fall back to the node's
                # declared default so the graph stays queueable.
                value = _spec_default(ispec)
            if value is None:
                return  # no value and no default: leave the input unset
            inputs[iname] = value
            has_cag = bool(cfg.get("control_after_generate")
                           or cfg.get("control_before_generate"))
            if has_cag or _looks_like_seed(iname, itype):
                # Seed-like widgets are followed by a mode string in
                # widgets_values; consume it if it really is one.
                if wi < len(widgets) and widgets[wi] in SEED_MODES:
                    wi += 1

        for section in ("required", "optional"):
            for iname, ispec in (input_spec.get(section) or {}).items():
                emit_input(iname, ispec)

        prompt[str(node.get("id"))] = {
            "class_type": ctype,
            "inputs": inputs,
            "_meta": {"title": node.get("title") or ctype},
        }

    if missing_classes:
        uniq = sorted(set(missing_classes))
        raise CustomWorkflowError(
            "Workflow needs custom nodes that are not installed in ComfyUI: "
            + ", ".join(uniq)
        )
    if not prompt:
        raise CustomWorkflowError("Workflow has no active nodes.")
    return prompt


def _looks_like_seed(iname: str, itype: Any) -> bool:
    return itype == "INT" and iname.lower().endswith("seed")


def _dynamic_combo_options(itype: Any, cfg: Dict[str, Any]) -> List[dict]:
    """Return the nested option dicts of a dynamic combo, if any."""
    options = cfg.get("options")
    if isinstance(options, list) and options \
            and all(isinstance(o, dict) and "key" in o for o in options):
        return options
    return []


def _is_dynamic_combo(itype: Any, cfg: Dict[str, Any]) -> bool:
    return itype in ("COMFY_DYNAMICCOMBO_V3", "COMBO3D") \
        or bool(_dynamic_combo_options(itype, cfg))


def _match_dynamic_option(itype: Any, cfg: Dict[str, Any],
                          value: Any) -> Optional[dict]:
    for option in _dynamic_combo_options(itype, cfg):
        if option.get("key") == value:
            return option
    return None


def _dynamic_combo_default(itype: Any, cfg: Dict[str, Any]) -> Any:
    options = _dynamic_combo_options(itype, cfg)
    return options[0].get("key") if options else "auto"


def _spec_default(ispec: List[Any]) -> Any:
    """Best-effort default for an input spec ([type] or [options, cfg])."""
    itype = ispec[0] if ispec else None
    if isinstance(itype, list) and itype:
        return itype[0]
    return {"STRING": "", "INT": 0, "FLOAT": 0.0, "BOOLEAN": False}.get(itype)


# ---------------------------------------------------------------------- #
# Introspection -> form schema
# ---------------------------------------------------------------------- #
def _group_for(class_type: str, input_name: str) -> str:
    c = class_type.lower()
    n = input_name.lower()
    if "loadimage" in c or n in ("image", "choose file to upload"):
        return "images"
    if ("cliptextencode" in c or "textencode" in c or "prompt" in c
            or c.startswith("string")):
        return "prompts"
    if "loader" in c or c.startswith("load"):
        return "models"
    if ("sampler" in c or "guider" in c or "sigma" in c or "scheduler" in c
            or "latent" in c or "noise" in c or n.endswith("seed")
            or n in ("steps", "cfg", "width", "height", "length", "fps")):
        return "generation"
    return "advanced"


PROMPTISH_NAMES = {"text", "prompt", "positive", "negative", "string",
                   "caption", "conditioning", "enhanced_prompt", "idea"}


def introspect(workflow: Dict[str, Any],
               object_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a form schema describing every user-settable input in the graph.

    Works with or without a live ``object_info``; without it, kinds are
    inferred from the values already present in the API-format workflow.
    """
    fields: List[Dict[str, Any]] = []
    output_nodes: List[str] = []
    has_image_input = False

    for nid, node in sorted(workflow.items(),
                            key=lambda kv: _node_sort_key(kv[0])):
        ctype = node.get("class_type", "")
        title = (node.get("_meta") or {}).get("title") or ctype
        current = node.get("inputs") or {}
        spec = (object_info or {}).get(ctype) or {}
        input_spec = spec.get("input", {})

        if any(tag in ctype.lower() for tag in ("savevideo", "saveimage",
                                                "saveanimatedwebp", "saveaudio",
                                                "videocombine")):
            output_nodes.append(ctype)

        ordered: List[Tuple[str, List[Any], Dict[str, Any]]] = []
        if input_spec:
            for section in ("required", "optional"):
                for iname, ispec in (input_spec.get(section) or {}).items():
                    ordered.append((iname, ispec if isinstance(ispec, list)
                                    else [], current))
        else:
            for iname, val in current.items():
                if isinstance(val, list):
                    continue  # connection
                ordered.append((iname, [], current))

        for iname, ispec, _src in ordered:
            if iname == "filename_prefix" or iname.startswith("stroke_"):
                continue
            if isinstance(current.get(iname), list):
                continue  # wired to another node — not a user control
            if ispec:
                itype = ispec[0] if ispec else None
                cfg = ispec[1] if len(ispec) > 1 and isinstance(ispec[1], dict)\
                    else {}
                if cfg.get("forceInput"):
                    continue
                if _is_dynamic_combo(itype, cfg):
                    kind, options = "combo", [
                        str(o.get("key"))
                        for o in _dynamic_combo_options(itype, cfg)
                    ][:500]
                elif isinstance(itype, list):
                    kind, options = "combo", [str(o) for o in itype][:500]
                elif itype == "STRING":
                    kind, options = "text", None
                elif itype == "INT":
                    kind, options = "int", None
                elif itype == "FLOAT":
                    kind, options = "float", None
                elif itype == "BOOLEAN":
                    kind, options = "bool", None
                else:
                    continue  # connection-only type
            else:
                val = current.get(iname)
                if isinstance(val, bool):
                    kind, options = "bool", None
                elif isinstance(val, int):
                    kind, options = "int", None
                elif isinstance(val, float):
                    kind, options = "float", None
                elif isinstance(val, str):
                    kind, options = "text", None
                else:
                    continue

            group = _group_for(ctype, iname)
            if group == "images":
                kind = "image"
                has_image_input = True
            if kind == "int" and iname.lower().endswith("seed"):
                kind = "seed"

            default = current.get(iname)
            if default is None:
                default = _spec_default(ispec) if ispec else ""

            field: Dict[str, Any] = {
                "id": f"{nid}.{iname}",
                "node_id": str(nid),
                "class_type": ctype,
                "node_title": title,
                "input": iname,
                "kind": kind,
                "label": f"{title} · {iname}" if title != ctype
                         else iname,
                "default": default,
                "group": group,
            }
            if options:
                field["options"] = options
            if ispec and len(ispec) > 1 and isinstance(ispec[1], dict):
                cfg = ispec[1]
                if "min" in cfg:
                    field["min"] = cfg["min"]
                if "max" in cfg:
                    field["max"] = cfg["max"]
                if "step" in cfg:
                    field["step"] = cfg["step"]
            if kind == "text":
                field["multiline"] = (iname.lower() in PROMPTISH_NAMES
                                      or len(str(default)) > 60)
            fields.append(field)

    return {
        "fields": fields,
        "node_count": len(workflow),
        "output_nodes": sorted(set(output_nodes)),
        "has_image_input": has_image_input,
    }


def _node_sort_key(nid: str):
    try:
        return (0, int(nid))
    except (TypeError, ValueError):
        return (1, str(nid))


def apply_values(workflow: Dict[str, Any], values: Dict[str, Any],
                 ) -> Dict[str, Any]:
    """Return a copy of ``workflow`` with user form values patched in.

    Empty seed values are replaced with fresh random seeds; numeric coercion
    follows the type already present in the graph so ComfyUI validation
    accepts the result.
    """
    wf = copy.deepcopy(workflow)
    for key, value in (values or {}).items():
        if "." not in key:
            continue
        nid, iname = key.split(".", 1)
        node = wf.get(nid)
        if not isinstance(node, dict):
            continue
        inputs = node.setdefault("inputs", {})
        existing = inputs.get(iname)
        if isinstance(existing, bool) or isinstance(value, bool):
            inputs[iname] = bool(value)
        elif isinstance(existing, float) and not isinstance(existing, bool):
            inputs[iname] = float(value)
        elif isinstance(existing, int) and not isinstance(existing, bool) \
                and value != "":
            try:
                inputs[iname] = int(float(value))
            except (TypeError, ValueError):
                inputs[iname] = value
        else:
            inputs[iname] = value
    # Empty string seed -> random draw (the UI sends "" for "random").
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        for iname, val in (node.get("inputs") or {}).items():
            if iname.lower().endswith("seed") and val in ("", None):
                node["inputs"][iname] = random.randint(0, 2**32 - 1)
    return wf


def stage_image_values(values: Dict[str, Any]) -> Dict[str, Any]:
    """Decode data-URL image values into the ComfyUI input directory.

    LoadImage inputs reject absolute paths (server-side containment), so
    uploads arrive as ``data:image/...;base64,...`` strings and leave as bare
    input-directory filenames.
    """
    import base64

    inp = os.environ.get("COMFY_INPUT_DIR", "/data/comfyui/input")
    os.makedirs(inp, exist_ok=True)
    out: Dict[str, Any] = {}
    for key, value in (values or {}).items():
        if isinstance(value, str) and value.startswith("data:"):
            try:
                header, b64 = value.split(",", 1)
                ext = ".png"
                if "jpeg" in header or "jpg" in header:
                    ext = ".jpg"
                elif "webp" in header:
                    ext = ".webp"
                elif "gif" in header:
                    ext = ".gif"
                name = ("sc-" + hashlib.sha1(b64.encode()).hexdigest()[:12]
                        + ext)
                dest = os.path.join(inp, name)
                if not os.path.exists(dest):
                    with open(dest, "wb") as fh:
                        fh.write(base64.b64decode(b64))
                out[key] = name
            except Exception as exc:  # noqa: BLE001
                raise CustomWorkflowError(
                    f"Could not decode uploaded image for {key}: {exc}"
                ) from exc
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------- #
# Live queue validation (validate-and-discard — never rendered)
# ---------------------------------------------------------------------- #
def parse_validation_errors(body: str) -> List[str]:
    """Turn a ComfyUI ``/prompt`` rejection body into readable strings.

    ComfyUI answers 400 with ``{"error": ..., "node_errors": {node_id:
    {"class_type": ..., "errors": [{message, details, ...}]}}}``; the
    ``details`` text names the offending value (e.g. a model file that is
    not installed).
    """
    try:
        detail = json.loads(body)
    except json.JSONDecodeError:
        return [body[:400]]
    errors: List[str] = []
    for nid, node in (detail.get("node_errors") or {}).items():
        ctype = node.get("class_type") or "node"
        for err in node.get("errors") or []:
            msg = f"{ctype} #{nid}: {err.get('details') or err.get('message')}"
            errors.append(msg if len(msg) <= 240 else msg[:240] + "…")
    if not errors:
        top = detail.get("error")
        msg = top.get("message") if isinstance(top, dict) else None
        errors.append(str(msg or body[:240]))
    return errors


async def queue_validate_prompt(base_url: str,
                                workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ``workflow`` against a live ComfyUI without rendering it.

    Same mechanism as validate_templates.py: POST the prompt to ``/prompt``
    (ComfyUI validates the full graph server-side and answers 400 with
    per-node errors for invalid values such as missing model files), then
    immediately delete the accepted prompt id from the queue so no GPU time
    is spent.

    Returns record fields — ``{"validated": True, "validated_at": ...}`` or
    ``{"validated": False, "validated_at": ..., "validation_errors": [...]}``.
    Network failures propagate to the caller, which records them as
    "unknown" (``validated: None``), never as "invalid".
    """
    url = base_url.rstrip("/")
    outcome: Dict[str, Any] = {"validated": True, "validated_at": _now_iso()}
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(
            f"{url}/prompt",
            json={"prompt": workflow, "client_id": "simple-comfort-validate"},
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                outcome["validated"] = False
                outcome["validation_errors"] = parse_validation_errors(text)
                return outcome
            try:
                prompt_id = json.loads(text).get("prompt_id")
            except json.JSONDecodeError:
                prompt_id = None
        if prompt_id:
            # Accepted — pull it back off the queue so no GPU time is spent
            # (one retry in case the delete races the execution loop).
            for attempt in (1, 2):
                try:
                    async with session.post(f"{url}/queue",
                                            json={"delete": [prompt_id]}):
                        pass
                    break
                except aiohttp.ClientError:
                    if attempt == 2:
                        raise
    return outcome


# ---------------------------------------------------------------------- #
# Adapter
# ---------------------------------------------------------------------- #
class CustomWorkflowAdapter(ModelAdapter):
    """Wraps one imported workflow in the standard adapter contract."""

    SUPPORTED_MODES = {"t2v", "i2v"}

    def __init__(self, slug: str, name: str, workflow: Dict[str, Any],
                 schema: Dict[str, Any], source_format: str = "api",
                 validated: Optional[bool] = None,
                 validation_errors: Optional[List[str]] = None):
        self.slug = slug
        self.MODEL_KEY = f"custom:{slug}"
        self.MODEL_NAME = name
        self.workflow = workflow
        self.schema = schema
        self.source_format = source_format
        # Live-validation verdict for the workflow's DEFAULT values:
        # True/False once checked against a live ComfyUI; None = not yet
        # checked (imported while ComfyUI was down, or a legacy record).
        self.validated = validated
        self.validation_errors = validation_errors or []

    # -- detection: honest about whether the defaults actually queue ----- #
    def detect_capabilities(self, object_info, model_files) -> ModelStatus:
        status = ModelStatus(model=self.MODEL_KEY, display_name=self.MODEL_NAME)
        status.t2v_capable = True
        status.i2v_capable = bool(self.schema.get("has_image_input"))
        if self.validated is False:
            status.available = False
            status.status = "invalid_defaults"
            status.detail = ("Workflow defaults fail ComfyUI validation — "
                             "fix the highlighted values: "
                             + "; ".join(self.validation_errors[:2]))
            status.missing = list(self.validation_errors)
            return status
        status.available = True
        status.status = "available"
        status.detail = ("Imported ComfyUI workflow"
                         + (" (defaults validated)" if self.validated
                            else " (defaults not yet validated)"))
        return status

    # -- single section: the graph renders in one go --------------------- #
    def get_section_strategy(self, params: GenerationParams) -> List[SectionConfig]:
        return [SectionConfig(index=0, frames=25, fps=params.fps or 24,
                              is_first=True, use_reference=False,
                              approx_seconds=0.0)]

    def resolve_prompt(self, params: GenerationParams) -> str:
        """The prompt IS whatever the user typed into the graph's prompt
        fields — never appended lip-sync guidance."""
        if params.custom_values:
            for field in self.schema.get("fields", []):
                if field["group"] == "prompts":
                    val = params.custom_values.get(field["id"])
                    if isinstance(val, str) and val.strip():
                        return val.strip()
        return params.prompt.strip() or self.MODEL_NAME

    def resolve_resolution(self, params: GenerationParams) -> tuple:
        dims = {}
        for field in self.schema.get("fields", []):
            if field["input"] in ("width", "height"):
                dims[field["input"]] = (params.custom_values or {}).get(
                    field["id"])
        try:
            return int(dims["width"]), int(dims["height"])
        except (KeyError, TypeError, ValueError):
            return 0, 0

    def resolve_quality(self, params: GenerationParams) -> Dict[str, float]:
        steps = guidance = None
        for field in self.schema.get("fields", []):
            val = (params.custom_values or {}).get(field["id"])
            if field["input"] == "steps" and val not in (None, ""):
                steps = val
            if field["input"] == "cfg" and val not in (None, ""):
                guidance = val
        return {"steps": float(steps or 0), "guidance": float(guidance or 0)}

    def map_parameters(self, params: GenerationParams, section: SectionConfig,
                       last_frame_path: Optional[str]) -> Dict[str, Any]:
        return apply_values(self.workflow, params.custom_values)


# ---------------------------------------------------------------------- #
# Registry / persistence
# ---------------------------------------------------------------------- #
def _slugify(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "workflow"
    return base[:60]


class CustomWorkflowRegistry:
    """Stores imported workflows on disk and registers their adapters."""

    def __init__(self, adapters: Dict[str, ModelAdapter]):
        self.adapters = adapters  # the live registry dict — mutated in place
        self.records: Dict[str, Dict[str, Any]] = {}
        STORE_DIR.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> None:
        if not MANIFEST_PATH.exists():
            return
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Custom workflow manifest unreadable: %s", exc)
            return
        for slug, rec in manifest.items():
            try:
                self._register(slug, rec)
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping custom workflow %s: %s", slug, exc)

    def list(self) -> List[Dict[str, Any]]:
        return sorted(self.records.values(), key=lambda r: r["imported_at"],
                      reverse=True)

    def get(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.records.get(slug)

    def get_adapter(self, slug: str) -> Optional[CustomWorkflowAdapter]:
        rec = self.records.get(slug)
        return self.adapters.get(f"custom:{slug}") if rec else None

    async def add(self, filename: str, data: bytes,
                  object_info: Optional[Dict[str, Any]],
                  comfy_url: Optional[str] = None,
                  ) -> Dict[str, Any]:
        name = Path(filename).stem or "workflow"
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CustomWorkflowError(f"Not valid JSON: {exc}") from exc

        fmt = detect_format(parsed)
        if fmt == "ui":
            if not object_info:
                raise CustomWorkflowError(
                    "Editor-format workflows need a running ComfyUI to "
                    "convert (for node definitions). Start ComfyUI, or "
                    "re-export with 'Save (API format)'."
                )
            parsed = convert_ui_to_api(parsed, object_info)

        schema = introspect(parsed, object_info)
        if not schema["output_nodes"]:
            log.info("Imported workflow '%s' has no recognised output node",
                     name)

        # Live queue-validation of the DEFAULT values (validate-and-discard —
        # nothing renders). The import still succeeds on failure: the user
        # can fix the values in the UI, and ComfyUI may just be down.
        validation: Dict[str, Any] = {"validated": None,
                                      "validated_at": _now_iso()}
        if comfy_url:
            try:
                validation = await queue_validate_prompt(comfy_url, parsed)
            except Exception as exc:  # noqa: BLE001
                log.warning("Import-time validation unavailable for '%s': %s",
                            name, exc)

        slug = _slugify(name)
        while slug in self.records:
            slug += "-2" if not slug.endswith("2") else "x"

        wf_path = STORE_DIR / f"{slug}.json"
        wf_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        rec = {
            "slug": slug,
            "name": name,
            "file": wf_path.name,
            "format": fmt,
            "imported_at": _now_iso(),
            "schema": schema,
            "validated": validation.get("validated"),
            "validated_at": validation.get("validated_at"),
        }
        if rec["validated"] is False:
            rec["validation_errors"] = (
                validation.get("validation_errors")
                or ["ComfyUI rejected the workflow defaults."])
        self._register(slug, rec)
        self._save_manifest()
        return rec

    def remove(self, slug: str) -> bool:
        rec = self.records.pop(slug, None)
        if not rec:
            return False
        self.adapters.pop(f"custom:{slug}", None)
        (STORE_DIR / rec["file"]).unlink(missing_ok=True)
        self._save_manifest()
        return True

    # ------------------------------------------------------------------ #
    def _register(self, slug: str, rec: Dict[str, Any]) -> None:
        wf_path = STORE_DIR / rec["file"]
        workflow = json.loads(wf_path.read_text(encoding="utf-8"))
        # Legacy records predate validation and simply have no
        # "validated"/"validation_errors" keys — rec.get() keeps them
        # loading unchanged (verdict: not yet validated).
        adapter = CustomWorkflowAdapter(
            slug, rec["name"], workflow, rec.get("schema") or {},
            rec.get("format", "api"),
            validated=rec.get("validated"),
            validation_errors=rec.get("validation_errors"),
        )
        if not adapter.schema:
            adapter.schema = introspect(workflow)
        self.adapters[adapter.MODEL_KEY] = adapter
        self.records[slug] = rec

    def record_validation(self, slug: str,
                          outcome: Dict[str, Any]) -> None:
        """Persist a validation verdict onto the record, adapter, manifest."""
        rec = self.records.get(slug)
        if not rec:
            return
        rec["validated"] = outcome.get("validated")
        rec["validated_at"] = outcome.get("validated_at") or _now_iso()
        if rec["validated"] is False:
            rec["validation_errors"] = (
                outcome.get("validation_errors")
                or ["ComfyUI rejected the workflow defaults."])
        else:
            rec.pop("validation_errors", None)
        adapter = self.adapters.get(f"custom:{slug}")
        if isinstance(adapter, CustomWorkflowAdapter):
            adapter.validated = rec["validated"]
            adapter.validation_errors = rec.get("validation_errors") or []
        self._save_manifest()

    async def revalidate_stale(self, comfy_url: str) -> None:
        """(Re-)check the defaults of records without a live verdict.

        Covers legacy records imported before validation existed, records
        imported while ComfyUI was unreachable, and previously-invalid ones
        (the missing models may have been installed since). Already-valid
        records are left alone.
        """
        for slug, rec in list(self.records.items()):
            if rec.get("validated") is True:
                continue
            try:
                workflow = json.loads(
                    (STORE_DIR / rec["file"]).read_text(encoding="utf-8"))
                outcome = await queue_validate_prompt(comfy_url, workflow)
            except Exception as exc:  # noqa: BLE001
                log.warning("Revalidation skipped for '%s': %s", slug, exc)
                return  # worker unreachable — stop rather than retry-loop
            self.record_validation(slug, outcome)
            log.info("Custom workflow '%s' defaults %s", slug,
                     "validated" if outcome.get("validated") else "INVALID")

    def _save_manifest(self) -> None:
        MANIFEST_PATH.write_text(json.dumps(self.records, indent=2),
                                 encoding="utf-8")


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")
