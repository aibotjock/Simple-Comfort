#!/usr/bin/env python3
"""Regenerate the AAN LTX-2.5 pack's pinned node-spec fixture.

``workflows/aan_ltx25/pack_object_info.json`` pins the ComfyUI
``/object_info`` input specs (required/optional order included — the order
is load-bearing for widget mapping) for every node type the ten pack
workflows use. ``adapters/ltx25.py`` converts the editor-format graphs to
API prompts offline against this fixture, so the app works with ComfyUI
down and hermetic tests need no network.

Re-run this after upgrading ComfyUI (or the LTX custom nodes), then run
the pack tests and ``validate_templates.py`` — they will fail loudly if
input orders changed:

    ./venv/bin/python tools/refresh_pack_fixture.py
    ./venv/bin/python -m pytest tests/test_ltx25_pack.py -q
    ./venv/bin/python validate_templates.py

File-listing combo options (input-dir/model filenames) are stubbed so no
local filenames travel with the repo; they are never consulted for
conversion, only live ComfyUI validates actual membership.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = REPO_ROOT / "workflows" / "aan_ltx25"
# Extra editor-format workflow sets converted through the same pinned
# fixture (currently: the CCTech face-lock LTX-2.5 backend graph).
EXTRA_WORKFLOW_DIRS = [REPO_ROOT / "workflows" / "ltx25_facelock"]
ENDPOINT = "http://127.0.0.1:8188"


def pack_node_types() -> set:
    types: set = set()
    workflows = sorted(PACK_DIR.glob("AAN_LTX25_*.json"))
    for extra in EXTRA_WORKFLOW_DIRS:
        workflows += sorted(extra.glob("*.json"))
    for wf in workflows:
        for node in json.loads(wf.read_text(encoding="utf-8"))["nodes"]:
            if node["type"] not in ("Note", "MarkdownNote", "Reroute"):
                types.add(node["type"])
    return types


def sanitize(spec: list) -> list:
    """Stub file-listing combo options; keep structure and order."""
    if not isinstance(spec, list) or not spec:
        return spec
    itype = spec[0]
    if isinstance(itype, list):
        return [[], spec[1] if len(spec) > 1 else {}]
    if itype == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        cfg = dict(spec[1])
        cfg["options"] = []
        return [itype, cfg]
    return spec


def main() -> int:
    used = pack_node_types()
    with urllib.request.urlopen(f"{ENDPOINT}/object_info", timeout=30) as r:
        object_info = json.load(r)
    missing = sorted(t for t in used if t not in object_info)
    if missing:
        print("error: node types missing from live /object_info: "
              + ", ".join(missing), file=sys.stderr)
        return 1

    fixture = {}
    for ctype in sorted(used):  # top-level sort only; input order preserved
        entry = {"input": {}}
        for section in ("required", "optional"):
            raw = (object_info[ctype].get("input") or {}).get(section) or {}
            entry["input"][section] = {k: sanitize(v) for k, v in raw.items()}
        fixture[ctype] = entry

    out = PACK_DIR / "pack_object_info.json"
    out.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)} "
          f"({len(fixture)} node types from {ENDPOINT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
