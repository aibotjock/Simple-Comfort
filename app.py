"""Simple-Comfort — FastAPI application entry point.

A deliberately simple local WebUI that wraps ComfyUI for generating 30-second
AI video clips. Run with::

    python app.py

The server binds to localhost:PORT (default 7788). It starts cleanly even when
ComfyUI is not running (the UI then shows a disconnected status).
"""
from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import (FastAPI, File, Form, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from adapters import GenerationParams, build_adapters
from comfy_client import WorkerPool
from config import configure_logging, ensure_directories, settings
from generation_manager import GenerationManager
from model_detector import ModelDetector
from project_manager import ProjectManager

configure_logging()
log = logging.getLogger("simple_comfort.app")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = settings.OUTPUT_DIR / "uploads"


class AppState:
    pool: WorkerPool
    adapters: dict
    detector: ModelDetector
    projects: ProjectManager
    manager: GenerationManager


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    state.pool = WorkerPool(settings.endpoints, timeout=settings.COMFY_TIMEOUT)
    state.adapters = build_adapters()
    state.detector = ModelDetector(state.pool, state.adapters,
                                   cache_ttl=settings.MODEL_CACHE_TTL)
    state.projects = ProjectManager(settings.OUTPUT_DIR)
    state.manager = GenerationManager(
        state.pool, state.adapters, state.projects,
        max_concurrent=settings.MAX_CONCURRENT_JOBS,
        timeout=settings.COMFY_TIMEOUT,
    )

    log.info("Simple-Comfort starting on http://%s:%s", settings.HOST, settings.PORT)
    log.info("ComfyUI endpoints: %s", ", ".join(settings.endpoints))
    try:
        await state.pool.refresh_health()
    except Exception as exc:  # noqa: BLE001
        log.warning("Initial health check failed: %s", exc)

    yield

    log.info("Shutting down...")
    await state.pool.close()


app = FastAPI(title="Simple-Comfort", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------- #
# Static mounts
# ---------------------------------------------------------------------- #
STATIC_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(settings.OUTPUT_DIR)),
          name="outputs")


# ---------------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------------- #
@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"error": "index.html missing"}, status_code=500)
    return FileResponse(str(index_file))


@app.get("/api/status")
async def api_status():
    await state.pool.refresh_health()
    workers = state.pool.status()
    connected = any(w["alive"] for w in workers)
    return {
        "connected": connected,
        "workers": workers,
        "multi_gpu": state.pool.is_multi,
        "ffmpeg": _ffmpeg_ok(),
    }


@app.get("/api/models")
async def api_models():
    statuses = await state.detector.detect_all()
    return {
        "models": [s.to_dict() for s in statuses],
        "multi_gpu": state.pool.is_multi,
        "workers": state.pool.status(),
    }


@app.get("/api/clips")
async def api_clips(project_id: Optional[str] = None):
    clips = state.projects.list_clips(project_id)
    return {
        "clips": [c.to_public_dict(settings.OUTPUT_DIR) for c in clips]
    }


@app.get("/api/clip/{clip_id}")
async def api_clip(clip_id: str):
    clip = state.projects.get_clip(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    return clip.to_public_dict(settings.OUTPUT_DIR)


@app.get("/api/last-clip")
async def api_last_clip():
    clip = state.projects.get_last_clip()
    if not clip:
        return {"clip": None}
    return {"clip": clip.to_public_dict(settings.OUTPUT_DIR)}


def _ffmpeg_ok() -> bool:
    import ffmpeg_utils
    return ffmpeg_utils.ffmpeg_available()


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@app.post("/api/generate")
async def api_generate(
    prompt: str = Form(""),
    model: str = Form(...),
    mode: str = Form(...),
    aspect_ratio: str = Form("16:9"),
    quality: str = Form("balanced"),
    lip_sync: str = Form("true"),
    seed: Optional[str] = Form(None),
    fps: Optional[str] = Form(None),
    steps: Optional[str] = Form(None),
    guidance: Optional[str] = Form(None),
    width: Optional[str] = Form(None),
    height: Optional[str] = Form(None),
    segment_length: Optional[str] = Form(None),
    motion_strength: Optional[str] = Form(None),
    worker_index: Optional[str] = Form(None),
    continue_from: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    # Validate model + mode.
    if model not in state.adapters:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'")
    if mode not in ("t2v", "i2v"):
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'")

    adapter = state.adapters[model]

    # Capability guard.
    status = await state.detector.get_status(model)
    if status and not status.available:
        raise HTTPException(
            status_code=409,
            detail=f"{adapter.MODEL_NAME} is not ready: {status.detail}",
        )
    if status:
        if mode == "t2v" and not status.t2v_capable:
            raise HTTPException(status_code=409,
                                detail=f"{adapter.MODEL_NAME} does not support Text-to-Video.")
        if mode == "i2v" and not status.i2v_capable:
            raise HTTPException(status_code=409,
                                detail=f"{adapter.MODEL_NAME} does not support Image-to-Video.")

    # Continuity handling.
    parent_clip_id = None
    reference_image_path: Optional[str] = None
    if continue_from:
        parent = state.projects.get_clip(continue_from)
        if not parent:
            raise HTTPException(status_code=404, detail="Parent clip not found")
        parent_clip_id = parent.clip_id
        project_id = parent.project_id
        # Auto-use the parent's last frame as the reference.
        if parent.last_frame.exists():
            reference_image_path = str(parent.last_frame)
            mode = "i2v"

    # Handle uploaded reference image (overrides continuity frame if provided).
    if image is not None and image.filename:
        ext = Path(image.filename).suffix or ".png"
        dest = UPLOAD_DIR / f"ref-{secrets.token_hex(6)}{ext}"
        async with aiofiles.open(dest, "wb") as fh:
            await fh.write(await image.read())
        reference_image_path = str(dest)

    if mode == "i2v" and not reference_image_path:
        raise HTTPException(status_code=400,
                            detail="Image-to-Video requires a reference image.")

    if not prompt.strip() and mode == "t2v":
        raise HTTPException(status_code=400, detail="A prompt is required.")

    # Ensure a project exists.
    if not project_id:
        project_id = state.projects.create_project()

    params = GenerationParams(
        prompt=prompt,
        mode=mode,
        aspect_ratio=aspect_ratio,
        quality=quality,
        lip_sync=str(lip_sync).lower() in ("true", "1", "on", "yes"),
        seed=_parse_int(seed),
        fps=_parse_int(fps) or 24,
        steps=_parse_int(steps),
        guidance=_parse_float(guidance),
        width=_parse_int(width),
        height=_parse_int(height),
        segment_length=_parse_float(segment_length),
        motion_strength=_parse_float(motion_strength),
        reference_image=reference_image_path,
    )

    extra_metadata = {
        "model_key": model,
        "checkpoint_info": {"adapter": adapter.MODEL_NAME},
        "generation_settings": {
            "quality": quality,
            "worker_index": _parse_int(worker_index),
        },
    }

    job = state.manager.submit(
        params, project_id, parent_clip_id,
        _parse_int(worker_index), extra_metadata,
    )
    return {"job_id": job.job_id, "project_id": project_id,
            "status": job.status}


@app.post("/api/cancel/{job_id}")
async def api_cancel(job_id: str):
    ok = await state.manager.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404,
                            detail="Job not found or already finished")
    return {"cancelled": True, "job_id": job_id}


@app.get("/api/job/{job_id}")
async def api_job(job_id: str):
    job = state.manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.public()


@app.websocket("/ws/progress/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = state.manager.get_job(job_id)
    if not job:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    queue = state.manager.subscribe(job_id)
    try:
        # Send the current state immediately.
        await websocket.send_json(job.public())
        if job.status in ("completed", "failed", "cancelled"):
            await websocket.close()
            return

        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                # Heartbeat to keep the connection alive.
                await websocket.send_json({"heartbeat": True})
                continue
            await websocket.send_json(payload)
            if payload.get("status") in ("completed", "failed", "cancelled"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.debug("WS error for %s: %s", job_id, exc)
    finally:
        state.manager.unsubscribe(job_id, queue)
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
