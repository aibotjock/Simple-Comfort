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
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import (Body, FastAPI, File, Form, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from adapters import GenerationParams, build_adapters
from avatar_manager import AvatarManager
from custom_workflows import (CustomWorkflowError, CustomWorkflowRegistry,
                              apply_values, queue_validate_prompt,
                              stage_image_values)
from broll_manager import (BRollCategory, BRollRequest, list_categories,
                           prepare_generation_params)
from comfy_client import WorkerPool
from config import configure_logging, ensure_directories, settings
from export_manager import ExportManager, ExportType
from generation_manager import GenerationManager
from model_detector import ModelDetector
from project_manager import ProjectManager
from sequence_manager import SequenceManager
from voice_manager import VoiceManager

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
    avatars: AvatarManager
    voices: VoiceManager
    sequences: SequenceManager
    exports: ExportManager
    custom_workflows: CustomWorkflowRegistry


state = AppState()


def _port_in_use(host: str, port: int) -> bool:
    """Return True if another process already holds ``host:port``."""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((probe_host, port))
            return True
        except (ConnectionRefusedError, OSError):
            return False


def ok(data=None):
    """Standard success envelope."""
    return {"success": True, "data": data, "error": None}


def err(message: str, status_code: int = 400):
    """Standard error envelope as a JSONResponse."""
    return JSONResponse(
        {"success": False, "data": None, "error": message},
        status_code=status_code,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    state.pool = WorkerPool(settings.endpoints, timeout=settings.COMFY_TIMEOUT)
    state.adapters = build_adapters()
    # Imported custom workflows register themselves as adapters so jobs run
    # through the normal GenerationManager pipeline.
    state.custom_workflows = CustomWorkflowRegistry(state.adapters)
    state.custom_workflows.load_all()
    state.detector = ModelDetector(state.pool, state.adapters,
                                   cache_ttl=settings.MODEL_CACHE_TTL)
    state.projects = ProjectManager(settings.OUTPUT_DIR)
    state.manager = GenerationManager(
        state.pool, state.adapters, state.projects,
        max_concurrent=settings.MAX_CONCURRENT_JOBS,
        timeout=settings.COMFY_TIMEOUT,
    )
    state.avatars = AvatarManager(settings.OUTPUT_DIR)
    state.voices = VoiceManager(settings.OUTPUT_DIR, projects=state.projects)
    state.sequences = SequenceManager(settings.OUTPUT_DIR,
                                      projects=state.projects)
    state.exports = ExportManager(settings.OUTPUT_DIR, state.projects)

    log.info("Simple-Comfort starting on http://%s:%s", settings.HOST, settings.PORT)
    log.info("ComfyUI endpoints: %s", ", ".join(settings.endpoints))
    try:
        await state.pool.refresh_health()
    except Exception as exc:  # noqa: BLE001
        log.warning("Initial health check failed: %s", exc)

    # Legacy imports predate import-time validation; re-check their
    # defaults against a live worker (validate-and-discard, never rendered)
    # so /api/models never claims more than we actually know.
    alive = next((w for w in state.pool.workers if w.alive), None)
    if alive:
        try:
            await state.custom_workflows.revalidate_stale(
                alive.client.base_url)
        except Exception as exc:  # noqa: BLE001
            log.warning("Custom workflow revalidation failed: %s", exc)

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
    # Bulletproof: always return valid JSON with HTTP 200, never a 500 HTML
    # page. The front-end treats connected:false gracefully as "Disconnected".
    try:
        pool = getattr(state, "pool", None)
        if pool is None:
            return {
                "connected": False,
                "workers": [],
                "multi_gpu": False,
                "ffmpeg": _ffmpeg_ok(),
                "error": "worker pool not initialized",
            }
        await pool.refresh_health()
        workers = pool.status()
        connected = any(w["alive"] for w in workers)
        return {
            "connected": connected,
            "workers": workers,
            "multi_gpu": pool.is_multi,
            "ffmpeg": _ffmpeg_ok(),
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("api_status failed")
        return {
            "connected": False,
            "workers": [],
            "multi_gpu": False,
            "ffmpeg": False,
            "error": str(exc),
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


# ---------------------------------------------------------------------- #
# Custom workflow import (drag & drop ComfyUI JSON)
# ---------------------------------------------------------------------- #
async def _live_worker():
    """Return the first alive worker after a health refresh (or None)."""
    try:
        await state.pool.refresh_health()
        for worker in state.pool.workers:
            if worker.alive:
                return worker
    except Exception as exc:  # noqa: BLE001
        log.debug("worker health refresh failed: %s", exc)
    return None


async def _live_object_info():
    """Fetch /object_info from the first alive worker (or None)."""
    worker = await _live_worker()
    if worker is None:
        return None
    try:
        return await worker.client.get_object_info()
    except Exception as exc:  # noqa: BLE001
        log.debug("object_info fetch failed: %s", exc)
        return None


@app.get("/api/custom-workflows")
async def api_custom_workflows_list():
    return {"workflows": state.custom_workflows.list()}


@app.post("/api/custom-workflows")
async def api_custom_workflows_add(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        return err("Workflow file too large (max 8MB).")
    try:
        worker = await _live_worker()
        object_info = None
        if worker:
            try:
                object_info = await worker.client.get_object_info()
            except Exception as exc:  # noqa: BLE001
                log.debug("object_info fetch failed: %s", exc)
        # Import queue-validates the defaults when a worker is reachable;
        # the import still succeeds (with validated:false recorded) if the
        # defaults are broken or ComfyUI is down.
        record = await state.custom_workflows.add(
            file.filename or "workflow.json", data, object_info,
            comfy_url=worker.client.base_url if worker else None,
        )
    except CustomWorkflowError as exc:
        return err(str(exc))
    # The new engine (and its validation verdict) must show up in
    # /api/models immediately, not after the detector's cache TTL.
    state.detector.invalidate()
    return ok(record)


@app.delete("/api/custom-workflows/{slug}")
async def api_custom_workflows_delete(slug: str):
    if not state.custom_workflows.remove(slug):
        return err(f"Unknown workflow '{slug}'", status_code=404)
    state.detector.invalidate()
    return ok()


@app.post("/api/custom-workflows/{slug}/generate")
async def api_custom_generate(slug: str, body: dict = Body(...)):
    record = state.custom_workflows.get(slug)
    if not record:
        return err(f"Unknown workflow '{slug}'", status_code=404)

    values = body.get("values") or {}
    try:
        values = stage_image_values(values)
    except CustomWorkflowError as exc:
        return err(str(exc))

    # Honest failure up front: ask ComfyUI to validate the exact prompt this
    # job would submit (validate-and-discard — an accepted prompt is deleted
    # from the queue immediately and never rendered). ComfyUI's own error
    # text (naming e.g. missing model files) goes straight to the envelope.
    adapter = state.custom_workflows.get_adapter(slug)
    worker = await _live_worker()
    if worker is None or adapter is None:
        return err("No ComfyUI endpoint is reachable. Start ComfyUI and "
                   "check COMFY_ENDPOINTS.", status_code=503)
    try:
        outcome = await queue_validate_prompt(
            worker.client.base_url,
            apply_values(adapter.workflow, values),
        )
    except Exception as exc:  # noqa: BLE001
        return err(f"Could not validate the workflow against ComfyUI: {exc}",
                   status_code=503)
    if outcome.get("validated") is False:
        return err("ComfyUI rejected this workflow: "
                   + " | ".join(outcome.get("validation_errors") or []),
                   status_code=400)
    if record.get("validated") is not True:
        # First live check of this record's defaults — persist the verdict.
        try:
            defaults = await queue_validate_prompt(
                worker.client.base_url, adapter.workflow)
            state.custom_workflows.record_validation(slug, defaults)
        except Exception as exc:  # noqa: BLE001
            log.debug("Default revalidation skipped: %s", exc)

    schema = record.get("schema") or {}
    mode = "i2v" if schema.get("has_image_input") else "t2v"

    project_id = body.get("project_id") or state.projects.create_project()
    params = GenerationParams(
        prompt="",  # the adapter pulls it from the graph's prompt fields
        mode=mode,
        custom_values=values,
    )
    extra_metadata = {
        "model_key": f"custom:{slug}",
        "checkpoint_info": {"adapter": "Custom workflow",
                            "workflow": record["name"]},
        "generation_settings": {
            "workflow": record["name"],
            "worker_index": body.get("worker_index"),
        },
    }
    job = state.manager.submit(
        params, project_id, None, body.get("worker_index"), extra_metadata,
    )
    return {"job_id": job.job_id, "project_id": project_id,
            "status": job.status}


def _ffmpeg_ok() -> bool:
    try:
        import ffmpeg_utils
        return bool(ffmpeg_utils.ffmpeg_available())
    except Exception:  # noqa: BLE001
        return False


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
    duration: Optional[str] = Form(None),
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
    # Professional-quality panel (all optional).
    denoise: Optional[str] = Form(None),
    interpolate_fps: Optional[str] = Form(None),
    camera_motion: Optional[str] = Form(None),
    face_stability: Optional[str] = Form(None),
    upscale: Optional[str] = Form(None),
    anti_drift: Optional[str] = Form(None),
    avatar_id: Optional[str] = Form(None),
    voice_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    # Validate model + mode.
    if model not in state.adapters:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'")
    if mode not in ("t2v", "i2v"):
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'")
    target_seconds = _parse_float(duration)
    if target_seconds is not None and not 4 <= target_seconds <= 120:
        raise HTTPException(
            status_code=400,
            detail="duration must be between 4 and 120 seconds",
        )

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

    # Assemble optional professional-quality parameters.
    def _truthy(v: Optional[str]) -> bool:
        return str(v).lower() in ("true", "1", "on", "yes") if v else False

    quality_params: dict = {}
    if _truthy(denoise):
        quality_params["denoise"] = True
    interp = _parse_int(interpolate_fps)
    if interp:
        quality_params["interpolate_fps"] = interp
    cam = _parse_float(camera_motion)
    if cam is not None:
        quality_params["camera_motion"] = cam
    face = _parse_float(face_stability)
    if face is not None:
        quality_params["face_stability"] = face
    ups = _parse_float(upscale)
    if ups is not None:
        quality_params["upscale"] = ups
    anti = _parse_float(anti_drift)
    if anti is not None:
        quality_params["anti_drift"] = anti

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
        target_seconds=target_seconds,
        motion_strength=_parse_float(motion_strength),
        reference_image=reference_image_path,
        quality_params=quality_params,
    )

    extra_metadata = {
        "model_key": model,
        "checkpoint_info": {"adapter": adapter.MODEL_NAME},
        "avatar_id": avatar_id or None,
        "voice_id": voice_id or None,
        "generation_settings": {
            "quality": quality,
            "target_seconds": target_seconds,
            "worker_index": _parse_int(worker_index),
            "quality_params": quality_params,
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


# ---------------------------------------------------------------------- #
# Avatar library
# ---------------------------------------------------------------------- #
@app.get("/api/avatars")
async def api_avatars():
    return ok({"avatars": state.avatars.list_avatars(),
               "max": __import__("avatar_manager").MAX_AVATARS})


@app.get("/api/avatars/{avatar_id}")
async def api_avatar_get(avatar_id: str):
    avatar = state.avatars.get_avatar(avatar_id)
    if not avatar:
        return err("Avatar not found", 404)
    return ok(avatar.to_public_dict(settings.OUTPUT_DIR))


@app.post("/api/avatars")
async def api_avatar_create(
    name: str = Form(""),
    notes: str = Form(""),
    voice_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    if not image or not image.filename:
        return err("An avatar image is required.")
    try:
        data = await image.read()
        avatar = state.avatars.create_avatar(
            name=name, image_bytes=data, filename=image.filename,
            notes=notes, voice_id=voice_id or None,
        )
    except ValueError as exc:
        return err(str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Avatar creation failed")
        return err(f"Failed to create avatar: {exc}", 500)
    return ok(avatar.to_public_dict(settings.OUTPUT_DIR))


@app.patch("/api/avatars/{avatar_id}")
async def api_avatar_update(
    avatar_id: str,
    payload: dict = Body(default={}),
):
    avatar = state.avatars.update_avatar(
        avatar_id,
        name=payload.get("name"),
        notes=payload.get("notes"),
        voice_id=payload.get("voice_id"),
    )
    if not avatar:
        return err("Avatar not found", 404)
    return ok(avatar.to_public_dict(settings.OUTPUT_DIR))


@app.post("/api/avatars/{avatar_id}/image")
async def api_avatar_replace_image(avatar_id: str, image: UploadFile = File(...)):
    if not image or not image.filename:
        return err("An image file is required.")
    try:
        data = await image.read()
        avatar = state.avatars.replace_image(avatar_id, data, image.filename)
    except ValueError as exc:
        return err(str(exc))
    if not avatar:
        return err("Avatar not found", 404)
    return ok(avatar.to_public_dict(settings.OUTPUT_DIR))


@app.delete("/api/avatars/{avatar_id}")
async def api_avatar_delete(avatar_id: str):
    if not state.avatars.delete_avatar(avatar_id):
        return err("Avatar not found", 404)
    return ok({"deleted": avatar_id})


# ---------------------------------------------------------------------- #
# Voice library
# ---------------------------------------------------------------------- #
@app.get("/api/voices")
async def api_voices():
    return ok({"voices": state.voices.list_voices()})


@app.get("/api/voices/{voice_id}")
async def api_voice_get(voice_id: str):
    voice = state.voices.get_voice(voice_id)
    if not voice:
        return err("Voice not found", 404)
    return ok(voice.to_public_dict(settings.OUTPUT_DIR))


@app.post("/api/voices")
async def api_voice_create(
    name: str = Form(""),
    notes: str = Form(""),
    avatar_id: Optional[str] = Form(None),
    sample: UploadFile = File(...),
):
    if not sample or not sample.filename:
        return err("A voice sample file is required.")
    try:
        data = await sample.read()
        voice = state.voices.create_voice(
            name=name, audio_bytes=data, filename=sample.filename,
            notes=notes, avatar_id=avatar_id or None,
        )
    except ValueError as exc:
        return err(str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("Voice creation failed")
        return err(f"Failed to create voice: {exc}", 500)
    # If linked to an avatar, mirror the link on the avatar record.
    if avatar_id:
        state.avatars.assign_voice(avatar_id, voice.id)
    return ok(voice.to_public_dict(settings.OUTPUT_DIR))


@app.patch("/api/voices/{voice_id}")
async def api_voice_update(voice_id: str, payload: dict = Body(default={})):
    voice = state.voices.update_voice(
        voice_id, name=payload.get("name"), notes=payload.get("notes"),
    )
    if not voice:
        return err("Voice not found", 404)
    return ok(voice.to_public_dict(settings.OUTPUT_DIR))


@app.post("/api/voices/{voice_id}/assign-avatar")
async def api_voice_assign_avatar(voice_id: str, payload: dict = Body(default={})):
    avatar_id = payload.get("avatar_id")
    voice = state.voices.assign_to_avatar(voice_id, avatar_id)
    if not voice:
        return err("Voice not found", 404)
    if avatar_id:
        state.avatars.assign_voice(avatar_id, voice_id)
    return ok(voice.to_public_dict(settings.OUTPUT_DIR))


@app.delete("/api/voices/{voice_id}")
async def api_voice_delete(voice_id: str):
    if not state.voices.delete_voice(voice_id):
        return err("Voice not found", 404)
    return ok({"deleted": voice_id})


@app.post("/api/clips/{clip_id}/narration")
async def api_clip_narration(clip_id: str, payload: dict = Body(default={})):
    voice_id = payload.get("voice_id")
    if not voice_id:
        return err("voice_id is required.")
    try:
        result = state.voices.assign_narration_to_clip(clip_id, voice_id)
    except ValueError as exc:
        return err(str(exc), 404)
    except Exception as exc:  # noqa: BLE001
        log.exception("Narration assignment failed")
        return err(f"Failed to assign narration: {exc}", 500)
    return ok(result)


# ---------------------------------------------------------------------- #
# B-roll
# ---------------------------------------------------------------------- #
@app.get("/api/broll/categories")
async def api_broll_categories():
    return ok({"categories": list_categories()})


@app.get("/api/broll/{project_id}")
async def api_broll_list(project_id: str):
    clips = state.projects.get_clips_for_project(project_id)
    broll = [c.to_public_dict(settings.OUTPUT_DIR) for c in clips
             if c.clip_type == "broll"]
    return ok({"broll": broll})


@app.post("/api/broll/generate")
async def api_broll_generate(payload: dict = Body(...)):
    model = payload.get("model")
    if model not in state.adapters:
        return err(f"Unknown model '{model}'")
    adapter = state.adapters[model]

    status = await state.detector.get_status(model)
    if status and not status.available:
        return err(f"{adapter.MODEL_NAME} is not ready: {status.detail}", 409)
    if status and not status.t2v_capable:
        return err(f"{adapter.MODEL_NAME} does not support Text-to-Video.", 409)

    try:
        category = BRollCategory(str(payload.get("category", "")).lower())
    except ValueError:
        return err(f"Unknown b-roll category '{payload.get('category')}'")

    subject = (payload.get("subject") or "").strip()
    if not subject:
        return err("A b-roll subject / description is required.")

    project_id = payload.get("project_id") or state.projects.create_project()
    request = BRollRequest(
        project_id=project_id,
        category=category,
        subject=subject,
        aspect_ratio=payload.get("aspect_ratio", "16:9"),
        quality=payload.get("quality", "balanced"),
        variants=int(payload.get("variants", 1) or 1),
        seed=_parse_int(str(payload.get("seed"))) if payload.get("seed") else None,
        model=model,
    )
    param_sets = prepare_generation_params(request)

    jobs = []
    for ps in param_sets:
        params = GenerationParams(
            prompt=ps["prompt"],
            mode="t2v",
            aspect_ratio=ps["aspect_ratio"],
            quality=ps["quality"],
            lip_sync=False,
            seed=ps["seed"],
            target_seconds=ps["target_seconds"],
        )
        extra_metadata = {
            "model_key": model,
            "checkpoint_info": {"adapter": adapter.MODEL_NAME},
            "clip_type": "broll",
            "broll_category": ps["broll_category"],
            "variant_index": ps["variant_index"],
            "generation_settings": {"quality": ps["quality"], "broll": True},
        }
        job = state.manager.submit(params, project_id, None, None,
                                   extra_metadata)
        jobs.append({"job_id": job.job_id, "variant_index": ps["variant_index"]})

    return ok({"project_id": project_id, "jobs": jobs})


# ---------------------------------------------------------------------- #
# Clip annealing / bridge clips
# ---------------------------------------------------------------------- #
@app.post("/api/clips/{clip_id}/anneal-to-next")
async def api_clip_anneal(clip_id: str, payload: dict = Body(default={})):
    """Generate a short bridge clip continuing from ``clip_id``'s last frame.

    The bridge conditions on the source clip's final frame (I2V) so it can be
    inserted between this clip and the next for a smooth transition.
    """
    clip = state.projects.get_clip(clip_id)
    if not clip:
        return err("Source clip not found", 404)
    if not clip.last_frame.exists():
        return err("Source clip has no last frame to continue from.", 409)

    model = payload.get("model") or clip.metadata.get("model_key")
    if model not in state.adapters:
        return err(f"Unknown model '{model}'")
    adapter = state.adapters[model]

    status = await state.detector.get_status(model)
    if status and not status.available:
        return err(f"{adapter.MODEL_NAME} is not ready: {status.detail}", 409)
    if status and not status.i2v_capable:
        return err(f"{adapter.MODEL_NAME} does not support Image-to-Video.", 409)

    bridge_seconds = float(payload.get("bridge_seconds", 3.0) or 3.0)
    prompt = (payload.get("prompt") or clip.metadata.get("prompt") or "").strip()

    params = GenerationParams(
        prompt=prompt,
        mode="i2v",
        aspect_ratio=clip.metadata.get("aspect_ratio", "16:9"),
        quality=payload.get("quality", clip.metadata.get("quality", "balanced")),
        lip_sync=bool(clip.metadata.get("lip_sync", False)),
        reference_image=str(clip.last_frame),
        target_seconds=bridge_seconds,
    )
    extra_metadata = {
        "model_key": model,
        "checkpoint_info": {"adapter": adapter.MODEL_NAME},
        "clip_type": "bridge",
        "generation_settings": {"bridge": True},
    }
    job = state.manager.submit(params, clip.project_id, clip.clip_id, None,
                               extra_metadata)
    state.projects.save_anneal_metadata(clip_id, {
        "bridge_job_id": job.job_id,
        "next_clip_id": payload.get("next_clip_id"),
        "bridge_seconds": bridge_seconds,
    })
    return ok({"job_id": job.job_id, "project_id": clip.project_id,
               "source_clip_id": clip.clip_id})


# ---------------------------------------------------------------------- #
# Sequence
# ---------------------------------------------------------------------- #
@app.get("/api/sequence/{project_id}")
async def api_sequence_get(project_id: str):
    return ok({"sequence": state.sequences.get_sequence(project_id)})


@app.post("/api/sequence/{project_id}")
async def api_sequence_add(project_id: str, payload: dict = Body(...)):
    clip_id = payload.get("clip_id")
    if not clip_id:
        return err("clip_id is required.")
    clip = state.projects.get_clip(clip_id)
    item_type = payload.get("type") or (clip.clip_type if clip else "clip")
    label = payload.get("label", "")
    seq = state.sequences.add_to_sequence(project_id, clip_id, item_type, label)
    return ok({"sequence": seq})


@app.delete("/api/sequence/{project_id}/{item_id}")
async def api_sequence_remove(project_id: str, item_id: str):
    seq = state.sequences.remove_from_sequence(project_id, item_id)
    return ok({"sequence": seq})


@app.post("/api/sequence/{project_id}/move")
async def api_sequence_move(project_id: str, payload: dict = Body(...)):
    item_id = payload.get("item_id")
    direction = payload.get("direction", "up")
    if not item_id:
        return err("item_id is required.")
    if direction not in ("up", "down"):
        return err("direction must be 'up' or 'down'.")
    seq = state.sequences.move_item(project_id, item_id, direction)
    return ok({"sequence": seq})


@app.post("/api/sequence/{project_id}/reorder")
async def api_sequence_reorder(project_id: str, payload: dict = Body(...)):
    order = payload.get("order") or payload.get("item_ids")
    if not isinstance(order, list):
        return err("'order' must be a list of item ids.")
    seq = state.sequences.reorder(project_id, order)
    return ok({"sequence": seq})


# ---------------------------------------------------------------------- #
# Export / handoff
# ---------------------------------------------------------------------- #
@app.post("/api/export")
async def api_export(payload: dict = Body(...)):
    project_id = payload.get("project_id")
    if not project_id:
        return err("project_id is required.")
    clip_ids = payload.get("clip_ids")
    if clip_ids is not None and not isinstance(clip_ids, list):
        return err("clip_ids must be a list.")
    try:
        export_type = ExportType(str(payload.get("export_type",
                                                 "project_package")).lower())
    except ValueError:
        return err(f"Unknown export type '{payload.get('export_type')}'")

    try:
        zip_path = state.exports.create_export(
            project_id, clip_ids, export_type,
        )
    except ValueError as exc:
        return err(str(exc), 409)
    except Exception as exc:  # noqa: BLE001
        log.exception("Export failed")
        return err(f"Export failed: {exc}", 500)

    def _iter_file(path: Path):
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    headers = {
        "Content-Disposition": f'attachment; filename="{zip_path.name}"'
    }
    return StreamingResponse(_iter_file(zip_path),
                             media_type="application/zip", headers=headers)


def main() -> None:
    # Section 5: refuse to start a second instance on the same port. If the
    # port is already serving, assume Simple-Comfort is running and exit cleanly
    # (the desktop launcher will simply open the existing tab).
    if _port_in_use(settings.HOST, settings.PORT):
        print(
            f"Simple-Comfort is already running on "
            f"http://{settings.HOST}:{settings.PORT} — opening the existing "
            f"instance instead of starting a new one.",
            flush=True,
        )
        sys.exit(0)

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
