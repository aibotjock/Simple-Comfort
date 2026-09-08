#!/usr/bin/env python3
"""Model setup checker: is every file the engines need actually installed?

Read-only by design: this tool NEVER downloads anything and NEVER prompts.
It only inspects the ComfyUI models tree (and, when reachable, the live
ComfyUI ``/object_info`` endpoint) and prints what is present, what is
missing, and the exact command that would install each missing file.

The REQUIRED manifest is derived from what the app actually loads:

* ``workflows/**/*.json`` — every filename fed to a loader node
  (``UNETLoader.unet_name``, ``CheckpointLoaderSimple.ckpt_name``,
  ``CLIPLoader.clip_name``, ``DualCLIPLoader.clip_name1/2``,
  ``VAELoader.vae_name``, ``LoraLoader*/lora_name``,
  ``CLIPVisionLoader.clip_name``, ``LatentUpscaleModelLoader.model_name``,
  ``LoadFramePackModel.model``, ``LTXICLoRALoaderModelOnly.lora_name``).
  Both graph shapes are scanned: API-format prompts read the named
  loader inputs directly; editor-format graphs (the AAN LTX-2.5 pack)
  scan the loader nodes' widget values for model-file extensions, since
  widget slots there are positional, not named. Template placeholders
  (``{{...}}``) are skipped. Only workflows whose filename prefix maps
  to an engine registered in ``adapters/__init__.py:build_adapters`` are
  required — unmapped workflows are reported as skipped, not required.
* ``adapters/<engine>.py`` — any quoted ``*.safetensors`` / ``*.ckpt``
  filename hardcoded in the adapter itself (directory resolved by looking
  the name up in the known model folders).

Availability authority: presence is decided on disk under
``$COMFY_PATH/models`` (recursive, alias folders included, ``.part`` /
incomplete downloads never count). When a ComfyUI endpoint from
``$COMFY_ENDPOINTS`` is reachable, its ``/object_info`` combo lists are
cross-checked per file — a file that sits on disk but is not served live
is flagged (stale ComfyUI process or a different models dir), which keeps
the tool honest on both sides without letting a live process mask a
misconfigured models root.

Exit code: 0 when every engine is READY, 1 when anything is missing
(usable as a CI/script gate), 2 on internal error.

NOTE: tools/MODELS.md is the human-readable manifest of the same data —
it MUST be updated together with this script whenever a workflow, adapter
or source entry changes.

Usage::

    ./venv/bin/python tools/check_models.py           # human table
    ./venv/bin/python tools/check_models.py --json    # machine-readable

Environment::

    COMFY_PATH       models root          (default /data/comfyui)
    COMFY_ENDPOINTS  comma-separated URLs (default http://127.0.0.1:8188)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    """Process env wins; else the repo's .env (same convention as config.py);
    else the default."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == name:
                return value.strip()
    return default


MODELS_ROOT = Path(_env("COMFY_PATH", "/data/comfyui"))
ENDPOINTS = [e.strip().rstrip("/") for e in
             _env("COMFY_ENDPOINTS", "http://127.0.0.1:8188").split(",")
             if e.strip()]

# Engines mirror adapters/__init__.py:build_adapters() — a workflow whose
# filename starts with the prefix (case-insensitive) belongs to that
# engine. The LTX-2.5 backend is the AAN candidate pack under
# workflows/aan_ltx25/.
ENGINES: List[Tuple[str, str, str]] = [
    # (workflow prefix, model key, display name)
    ("aan_ltx25", "ltx25", "LTX 2.5"),
    ("ltx25", "ltx25", "LTX 2.5"),
    ("wan22", "wan22", "Wan 2.2"),
    ("framepack", "framepack", "FramePack"),
]

# Loader node -> {input key: logical model folder} (ComfyUI folder_paths
# conventions; LoadFramePackModel reads diffusion_models — verified in
# ComfyUI-FramePackWrapper/nodes.py, folder_paths.get_full_path_or_raise).
LOADER_INPUTS: Dict[str, Dict[str, str]] = {
    "UNETLoader": {"unet_name": "diffusion_models"},
    "CheckpointLoaderSimple": {"ckpt_name": "checkpoints"},
    "CLIPLoader": {"clip_name": "text_encoders"},
    "DualCLIPLoader": {"clip_name1": "text_encoders",
                       "clip_name2": "text_encoders"},
    "VAELoader": {"vae_name": "vae"},
    "LoraLoader": {"lora_name": "loras"},
    "LoraLoaderModelOnly": {"lora_name": "loras"},
    "CLIPVisionLoader": {"clip_name": "clip_vision"},
    "LatentUpscaleModelLoader": {"model_name": "latent_upscale_models"},
    "LoadFramePackModel": {"model": "diffusion_models"},
    "LTXICLoRALoaderModelOnly": {"lora_name": "loras"},
}

# Legacy folder aliases ComfyUI still serves (same mapping as
# adapters/base.py model detection).
FOLDER_ALIASES: Dict[str, List[str]] = {
    "diffusion_models": ["diffusion_models", "unet"],
    "text_encoders": ["text_encoders", "clip"],
    "checkpoints": ["checkpoints"],
}

# Incomplete-download suffixes that must never count as present.
PARTIAL_SUFFIXES = (".part", ".incomplete", ".tmp", ".crdownload")

# Official / well-established sources. verified=False means the exact repo
# and path are certain; verified=True prints the org/repo slug but marks the
# source "verify" — no URL is invented when we are not sure.
# (repo, repo-relative path or None for repo root, verified)
SOURCES: Dict[str, Tuple[str, Optional[str], bool]] = {
    # LTX 2.5 — Lightricks official org.
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors":
        ("Lightricks/LTX-2.5", None, True),
    "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors":
        ("Lightricks/LTX-2.5", None, True),
    "gemma4_e2b_it_int8_convrot.safetensors":
        ("Lightricks/LTX-2.5", None, True),
    "ltx-2.5-video-vae-bf16.safetensors":
        ("Lightricks/LTX-2.5", None, True),
    "ltx-2.5-audio-vae-bf16.safetensors":
        ("Lightricks/LTX-2.5", None, True),
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors":
        ("Lightricks/LTX-2.5", None, True),
    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors":
        ("Lightricks/LTX-2.5", None, True),
    # Wan 2.2 — ComfyUI org repackaged split_files distribution.
    "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors":
        ("Comfy-Org/Wan_2.2_ComfyUI_repackaged",
         "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", False),
    "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors":
        ("Comfy-Org/Wan_2.2_ComfyUI_repackaged",
         "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", False),
    "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors":
        ("Comfy-Org/Wan_2.2_ComfyUI_repackaged",
         "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", False),
    "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors":
        ("Comfy-Org/Wan_2.2_ComfyUI_repackaged",
         "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", False),
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors":
        ("Comfy-Org/Wan_2.2_ComfyUI_repackaged",
         "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", False),
    "wan_2.1_vae.safetensors":
        ("Comfy-Org/Wan_2.2_ComfyUI_repackaged",
         "split_files/vae/wan_2.1_vae.safetensors", False),
    "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors":
        ("lightx2v (exact repo slug unconfirmed)", None, True),
    "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors":
        ("lightx2v (exact repo slug unconfirmed)", None, True),
    "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors":
        ("lightx2v (exact repo slug unconfirmed)", None, True),
    "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors":
        ("lightx2v (exact repo slug unconfirmed)", None, True),
    # FramePack — lllyasviel official.
    "FramePackI2V_HY_fp8_e4m3fn.safetensors":
        ("lllyasviel/FramePackI2V_HY", None, False),
    "clip_l.safetensors":
        ("comfyanonymous/flux_text_encoders", None, False),
    "llava_llama3_fp8_scaled.safetensors":
        ("lllyasviel FramePack ecosystem (exact repo unconfirmed)", None, True),
    "hunyuan_video_vae_bf16.safetensors":
        ("lllyasviel / Kijai HunyuanVideo ecosystem (exact repo unconfirmed)", None, True),
    "sigclip_vision_patch14_384.safetensors":
        ("Comfy-Org/Wan_2.2_ComfyUI_repackaged",
         "split_files/clip_vision/sigclip_vision_patch14_384.safetensors", False),
}


@dataclass
class RequiredFile:
    """One model file an engine needs."""
    name: str
    directory: str            # logical ComfyUI model folder
    engine: str               # engine display name
    origin: str               # where the requirement came from
    status: str = "MISSING"   # ok | MISSING
    path: Optional[str] = None
    size_bytes: Optional[int] = None
    live_visible: Optional[bool] = None
    partial_on_disk: Optional[str] = None  # matching .part/.incomplete file
    source_repo: str = "unknown"
    source_path: Optional[str] = None
    source_needs_verify: bool = True

    def install_command(self) -> str:
        """Exact command that would install this file (never executed)."""
        target_dir = MODELS_ROOT / "models" / self.directory
        if self.source_repo and "/" in self.source_repo and not self.source_needs_verify:
            if self.source_path:
                url = (f"https://huggingface.co/{self.source_repo}"
                       f"/resolve/main/{self.source_path}")
                return f"curl -L -o {target_dir}/{self.name} {url}"
            return f"hf download {self.source_repo} {self.name} --local-path {target_dir}"
        if self.source_repo:
            return (f"hf download <repo> {self.name} --local-path {target_dir}"
                    f"   # VERIFY exact repo/path first — candidate: {self.source_repo}")
        return f"# no known source for {self.name} — find it manually"

    def size_h(self) -> str:
        if self.size_bytes is None:
            return "-"
        n = float(self.size_bytes)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if n < 1024 or unit == "TiB":
                return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
            n /= 1024
        return "-"


@dataclass
class EngineReport:
    display: str
    key: str
    workflows: List[str] = field(default_factory=list)
    files: List[RequiredFile] = field(default_factory=list)

    @property
    def missing(self) -> List[RequiredFile]:
        return [f for f in self.files if f.status != "ok"]

    @property
    def verdict(self) -> str:
        n = len(self.missing)
        return "READY" if n == 0 else f"MISSING {n} file{'s' if n != 1 else ''}"

    @property
    def live_unseen(self) -> List[RequiredFile]:
        return [f for f in self.files
                if f.status == "ok" and f.live_visible is False]

    def to_dict(self) -> Dict:
        return {
            "engine": self.display,
            "key": self.key,
            "workflows": self.workflows,
            "verdict": self.verdict,
            "missing_count": len(self.missing),
            "files": [
                {
                    "file": f.name,
                    "directory": f.directory,
                    "status": f.status,
                    "path": f.path,
                    "size_bytes": f.size_bytes,
                    "source": f.source_repo,
                    "source_path": f.source_path,
                    "source_status": "verify" if f.source_needs_verify else "official",
                    "live_visible": f.live_visible,
                    "partial_download": f.partial_on_disk,
                    "install_command": f.install_command(),
                    "required_by": f.origin,
                }
                for f in self.files
            ],
        }


# --------------------------------------------------------------------------
# Derivation: workflows + adapters -> required files
# --------------------------------------------------------------------------

def _ui_loader_refs(graph: dict) -> List[dict]:
    """Loader pseudo-nodes from an editor-format graph.

    Widget slots are positional in editor format, so loader inputs cannot
    be read by name; instead every widget value that looks like a model
    file becomes a pseudo ``{class_type, inputs: {<first input key>: v}}``
    entry — a loader node only ever references files of its own kind, so
    the class type alone fixes the folder.
    """
    pseudo: List[dict] = []
    for node in graph.get("nodes") or []:
        ctype = node.get("type", "")
        if ctype not in LOADER_INPUTS:
            continue
        for value in node.get("widgets_values") or []:
            if isinstance(value, str) \
                    and value.endswith((".safetensors", ".ckpt", ".pt", ".bin")):
                first_key = next(iter(LOADER_INPUTS[ctype]))
                pseudo.append({"class_type": ctype,
                               "inputs": {first_key: value}})
    return pseudo


def derive_required() -> Tuple[List[EngineReport], List[str]]:
    """Scan workflows/**/*.json and adapters/*.py for loader-referenced
    files."""
    engines = {key: EngineReport(display=disp, key=key)
               for _prefix, key, disp in ENGINES}
    prefix_to_key = {prefix: key for prefix, key, _disp in ENGINES}
    unmapped: List[str] = []

    for wf in sorted((REPO_ROOT / "workflows").rglob("*.json")):
        stem = wf.stem.lower()
        key = next((k for prefix, k in prefix_to_key.items()
                    if stem.startswith(prefix)), None)
        if key is None:
            unmapped.append(str(wf.relative_to(REPO_ROOT / "workflows")))
            continue
        engines[key].workflows.append(wf.name)
        try:
            graph = json.loads(wf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SystemExit(f"error: cannot parse {wf}: {exc}")
        if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
            nodes = _ui_loader_refs(graph)
        else:
            nodes = graph.values() if isinstance(graph, dict) else graph
        for node in nodes:
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            folder_map = LOADER_INPUTS.get(node.get("class_type", ""), {})
            for input_key, folder in folder_map.items():
                value = inputs.get(input_key)
                if not isinstance(value, str) or "{{" in value:
                    continue  # template placeholder, resolved per-request
                engines[key].files.append(RequiredFile(
                    name=value, directory=folder,
                    engine=engines[key].display,
                    origin=f"{wf.name}#{node.get('class_type')}.{input_key}"))

    # Adapter-source hardcoded filenames (directory resolved later).
    adapter_re = re.compile(r"[\"']([\w./-]+\.(?:safetensors|ckpt|pt|bin))[\"']")
    for _prefix, key, _disp in ENGINES:
        adapter = REPO_ROOT / "adapters" / f"{key}.py"
        if not adapter.exists():
            continue
        for match in adapter_re.findall(adapter.read_text(encoding="utf-8")):
            engines[key].files.append(RequiredFile(
                name=match, directory="?",
                engine=engines[key].display,
                origin=f"adapters/{key}.py"))

    # Dedup (same file can be required by several workflows/inputs).
    for engine in engines.values():
        seen: Dict[str, RequiredFile] = {}
        for rf in engine.files:
            if rf.name in seen:
                seen[rf.name].origin += f", {rf.origin}"
            else:
                seen[rf.name] = rf
        engine.files = [seen[n] for n in sorted(seen, key=lambda n: seen[n].name)]
    # Several prefixes can map to one engine (aan_ltx25 + ltx25 -> ltx25):
    # report each engine once, in first-seen order.
    order: List[str] = []
    for _p, key, _d in ENGINES:
        if key not in order:
            order.append(key)
    return [engines[key] for key in order], unmapped


# --------------------------------------------------------------------------
# Availability: disk scan + live /object_info cross-check
# --------------------------------------------------------------------------

def scan_disk() -> Dict[str, Dict[str, str]]:
    """basename -> full path, per logical folder (aliases merged, recursive)."""
    found: Dict[str, Dict[str, str]] = {}
    for logical, aliases in FOLDER_ALIASES.items():
        bag: Dict[str, str] = {}
        for alias in aliases:
            base = MODELS_ROOT / "models" / alias
            if not base.is_dir():
                continue
            for root, _dirs, names in os.walk(base):
                for n in names:
                    if n not in bag and not n.endswith(PARTIAL_SUFFIXES) \
                            and not n.startswith("put_"):
                        bag[n] = str(Path(root) / n)
        found[logical] = bag
    for folder in ("vae", "loras", "clip_vision", "latent_upscale_models"):
        if folder not in found:
            found[folder] = {}
            base = MODELS_ROOT / "models" / folder
            if base.is_dir():
                for root, _dirs, names in os.walk(base):
                    for n in names:
                        if n not in found[folder] and not n.endswith(PARTIAL_SUFFIXES) \
                                and not n.startswith("put_"):
                            found[folder][n] = str(Path(root) / n)
    return found


def _combo_files(spec: List) -> List[str]:
    """Extract filenames from a combo input spec (both ComfyUI shapes)."""
    if not spec:
        return []
    # classic: [["a.safetensors", ...], {..}]; new: ["COMBO", {"options": [...]}]
    if isinstance(spec[0], list):
        return [str(v) for v in spec[0]]
    if spec[0] == "COMBO" and len(spec) > 1:
        options = spec[1].get("options", []) if isinstance(spec[1], dict) else []
        return [str(v) for v in options]
    return []


def live_object_info() -> Tuple[Optional[str], Dict[str, Dict[str, List[str]]]]:
    """Fetch per-loader file lists from the first reachable endpoint."""
    logical_dirs = {folder for m in LOADER_INPUTS.values() for folder in m.values()}
    for endpoint in ENDPOINTS:
        lists: Dict[str, List[str]] = {}
        for node in LOADER_INPUTS:
            try:
                req = urllib.request.Request(f"{endpoint}/object_info/{node}",
                                             headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    info = json.loads(resp.read().decode("utf-8"))
                combo = info[node]["input"]["required"]
                for input_key in LOADER_INPUTS[node]:
                    if input_key in combo:
                        lists.setdefault(input_key, []).extend(
                            _combo_files(combo[input_key]))
            except Exception:
                continue  # node missing on this endpoint; others may answer
        if lists:
            by_dir: Dict[str, Dict[str, List[str]]] = {d: [] for d in logical_dirs}
            for node, folder_map in LOADER_INPUTS.items():
                for input_key, folder in folder_map.items():
                    by_dir[folder] = sorted(set(by_dir[folder] + lists.get(input_key, [])))
            return endpoint, by_dir
    return None, {}


def resolve(reports: List[EngineReport], disk: Dict[str, Dict[str, str]],
            live: Dict[str, Dict[str, List[str]]]) -> None:
    """Fill status/size/source/live fields on every required file."""
    for engine in reports:
        for rf in engine.files:
            # Adapter-sourced files carry directory '?': find where it lives.
            if rf.directory == "?":
                for logical, bag in disk.items():
                    if rf.name in bag:
                        rf.directory = logical
                        break
                else:
                    rf.directory = "diffusion_models"  # best guess; stays MISSING
            bag = disk.get(rf.directory, {})
            path = bag.get(rf.name)
            if path:
                rf.status, rf.path = "ok", path
                try:
                    rf.size_bytes = os.path.getsize(path)
                except OSError:
                    pass
            else:
                # flag an interrupted download with the same stem, if any
                for suffix in PARTIAL_SUFFIXES:
                    for name in disk.get(rf.directory, {}):
                        if name == rf.name + suffix:
                            rf.partial_on_disk = name
            if rf.directory in live:
                rf.live_visible = rf.name in live[rf.directory]
            src = SOURCES.get(rf.name)
            if src:
                rf.source_repo, rf.source_path, rf.source_needs_verify = src


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def print_human(reports: List[EngineReport], unmapped: List[str],
                endpoint: Optional[str]) -> None:
    out = sys.stdout
    live_note = (endpoint + "/object_info (reachable)"
                 if endpoint else "unreachable — disk scan only")
    print("Model setup check")
    print(f"  models root : {MODELS_ROOT} (COMFY_PATH)")
    print(f"  live check  : {live_note}")
    if unmapped:
        print(f"  note        : workflows with no registered engine (skipped): "
              f"{', '.join(unmapped)}")
    print()

    for engine in reports:
        wf = ", ".join(engine.workflows) or "(none found)"
        print(f"Engine: {engine.display}   [{wf}]")
        w_dir = max([len(f.directory) for f in engine.files] + [9])
        w_name = max([len(f.name) for f in engine.files] + [4])
        print(f"  {'STATUS':7} {'SIZE':>10}  {('DIRECTORY' + ' ' * w_dir)[:w_dir]}  "
              f"{('FILE' + ' ' * w_name)[:w_name]}  SOURCE")
        for f in sorted(engine.files, key=lambda f: (f.directory, f.name)):
            src = f.source_repo + ("" if not f.source_needs_verify else " (verify)")
            print(f"  {f.status:7} {f.size_h():>10}  {f.directory:<{w_dir}}  "
                  f"{f.name:<{w_name}}  {src}")
        for f in engine.live_unseen:
            print(f"  warning: {f.name} is on disk but the live endpoint does not"
                  f" list it (ComfyUI restart or different models dir?)")
        n_ok = len(engine.files) - len(engine.missing)
        print(f"  Verdict: {engine.verdict}  ({n_ok}/{len(engine.files)} files)")
        print()

    total = sum(len(e.files) for e in reports)
    ok = total - sum(len(e.missing) for e in reports)
    ready = sum(1 for e in reports if not e.missing)
    print(f"SUMMARY: {ready}/{len(reports)} engines READY, {ok}/{total} files present")

    missing = [(e, f) for e in reports for f in e.missing]
    if missing:
        print("\nInstall commands (NOT executed — this tool is read-only):")
        for engine, f in missing:
            print(f"  # {engine.display} — {f.directory}/{f.name}"
                  + (f"  [partial download on disk: {f.partial_on_disk}]"
                     if f.partial_on_disk else ""))
            print(f"  {f.install_command()}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only check that every engine model file is installed.")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output (app error envelope)")
    args = parser.parse_args(argv)

    reports, unmapped = derive_required()
    disk = scan_disk()
    endpoint, live = live_object_info()
    resolve(reports, disk, live)

    total = sum(len(e.files) for e in reports)
    missing = sum(len(e.missing) for e in reports)
    ready = sum(1 for e in reports if not e.missing)
    all_ready = missing == 0 and total > 0

    if args.json:
        envelope = {
            "success": all_ready,
            "data": {
                "summary": {
                    "engines_ready": ready,
                    "engines_total": len(reports),
                    "files_present": total - missing,
                    "files_total": total,
                    "all_ready": all_ready,
                    "models_root": str(MODELS_ROOT),
                    "live_endpoint": endpoint,
                    "skipped_unmapped_workflows": unmapped,
                },
                "engines": [e.to_dict() for e in reports],
            },
            "error": None if all_ready else
            f"{missing} required file(s) missing across "
            f"{len(reports) - ready} engine(s)",
        }
        print(json.dumps(envelope, indent=2))
    else:
        print_human(reports, unmapped, endpoint)

    return 0 if all_ready else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover — internal error gate
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
