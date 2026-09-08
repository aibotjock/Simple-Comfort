"""One-shot validation of every engine template against the live ComfyUI.

Builds each adapter's workflow via map_parameters() exactly as a real job
would, POSTs it to /prompt (ComfyUI validates the full graph server-side),
then interrupts so no GPU time is spent. Exits non-zero on any failure.
"""
import asyncio
import json
import os
import struct
import sys
import zlib

import aiohttp

sys.path.insert(0, "/data/simple_comfort")

from adapters import build_adapters
from adapters.base import GenerationParams

COMFY = "http://127.0.0.1:8188"


def tiny_png(path):
    # 1x1 black PNG
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00")
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", idat) + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)
    return path


REF = tiny_png("/tmp/sc_ref.png")


async def validate(session, name, workflow):
    payload = {"prompt": workflow, "client_id": "simple-comfort-validate"}
    async with session.post(f"{COMFY}/prompt", json=payload) as resp:
        text = await resp.text()
        if resp.status != 200:
            try:
                detail = json.loads(text)
                err = detail.get("error", {})
                errs = "; ".join(
                    f"{n}: " + "; ".join(
                        f"{e.get('message')} ({e.get('details')})"
                        for e in (v.get("errors") or []))
                    for n, v in (detail.get("node_errors") or {}).items()
                ) or text[:400]
                return name, False, f"{err.get('message', err)} | {errs}"
            except json.JSONDecodeError:
                return name, False, text[:400]
        prompt_id = json.loads(text).get("prompt_id", "")
        # Accepted — pull it back off the queue so no GPU time is spent.
        await session.post(f"{COMFY}/queue",
                           json={"delete": [prompt_id]})
        return name, True, prompt_id


async def main():
    adapters = build_adapters()

    cases = []
    # LTX 2.5 — the AAN candidate pack: every profile in both modes, plus
    # the plain quality presets (which map onto pack profiles).
    from adapters.ltx25 import PROFILES
    for profile in sorted(PROFILES):
        for mode in ("t2v", "i2v"):
            cases.append((f"ltx25-{profile}-{mode}", "ltx25", mode,
                          {"ltx_profile": profile}))
    for quality in ("draft", "balanced", "quality"):
        cases.append((f"ltx25-preset-{quality}", "ltx25", "t2v", quality))
    cases += [
        ("wan22-t2v", "wan22", "t2v", "balanced"),
        ("wan22-i2v", "wan22", "i2v", "balanced"),
        ("framepack-i2v", "framepack", "i2v", "balanced"),
    ]

    results = []
    async with aiohttp.ClientSession() as session:
        for case in cases:
            label, key, mode, quality = case
            quality_params = {}
            if isinstance(quality, dict):
                quality_params, quality = quality, "balanced"
            adapter = adapters[key]
            params = GenerationParams(
                prompt="a calm ocean at dawn, gentle waves",
                mode=mode,
                quality=quality,
                lip_sync=False,
                reference_image=REF if mode == "i2v" else None,
                target_seconds=6,
                segment_length=6.0,
                quality_params=quality_params,
            )
            try:
                sections = adapter.get_section_strategy(params)
                wf = adapter.map_parameters(params, sections[0], None)
                results.append(await validate(session, label, wf))
            except Exception as exc:  # noqa: BLE001
                results.append((label, False, f"{type(exc).__name__}: {exc}"))

        # Split-GPU variant of the production backend (launch-split-gpu.sh):
        # the inserted MultiGPU_WorkUnits wrappers and the dual-CFG guider
        # values must validate against the live server too. 720p portrait
        # exercises the halved working grid.
        os.environ["SC_SPLIT_GPUS"] = "1"
        try:
            for mode in ("t2v", "i2v"):
                params = GenerationParams(
                    prompt="a calm ocean at dawn, gentle waves",
                    mode=mode,
                    quality="balanced",
                    lip_sync=False,
                    width=720,
                    height=1280,
                    reference_image=REF if mode == "i2v" else None,
                    target_seconds=6,
                    segment_length=6.0,
                )
                try:
                    adapter = adapters["ltx25"]
                    sections = adapter.get_section_strategy(params)
                    wf = adapter.map_parameters(params, sections[0], None)
                    results.append(await validate(
                        session, f"ltx25-facelock-splitgpu-{mode}", wf))
                except Exception as exc:  # noqa: BLE001
                    results.append((f"ltx25-facelock-splitgpu-{mode}",
                                    False, f"{type(exc).__name__}: {exc}"))
        finally:
            os.environ.pop("SC_SPLIT_GPUS", None)

        # ---- custom workflow import path ------------------------------ #
        from custom_workflows import (convert_ui_to_api, detect_format,
                                      introspect)
        async with session.get(f"{COMFY}/object_info") as resp:
            object_info = await resp.json()

        bar = "/data/simple_comfort/gauntlet/bar"
        for fname in ("lightx2v_t2v_4step_NativeComfy.json",
                      "text_to_video_wan22_14B.json",
                      "LTX-2.5_T2V_I2V_Single_Stage_Distilled.json"):
            try:
                data = json.load(open(f"{bar}/{fname}"))
                fmt = detect_format(data)
                wf = convert_ui_to_api(data, object_info) if fmt == "ui" \
                    else data
                schema = introspect(wf, object_info)
                ok_flag = bool(schema["fields"]) and bool(schema["node_count"])
                results.append((f"custom-import:{fname}", ok_flag,
                                f"format={fmt}, nodes={schema['node_count']}, "
                                f"fields={len(schema['fields'])}, "
                                f"outputs={schema['output_nodes']}"))
                if ok_flag:
                    results.append(await validate(
                        session, f"custom-queue:{fname}", wf))
            except Exception as exc:  # noqa: BLE001
                results.append((f"custom-import:{fname}", False,
                                f"{type(exc).__name__}: {exc}"))

        # Interrupt anything left over from queued validations.
        try:
            async with session.post(f"{COMFY}/interrupt"):
                pass
        except Exception:  # noqa: BLE001
            pass

    failed = 0
    for name, ok_flag, detail in results:
        mark = "PASS" if ok_flag else "FAIL"
        if not ok_flag:
            failed += 1
        print(f"[{mark}] {name}: {detail}")
    print(f"\n{len(results) - failed}/{len(results)} passed")
    sys.exit(1 if failed else 0)


asyncio.run(main())
