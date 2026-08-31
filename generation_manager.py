"""Generation orchestration.

Coordinates the full multi-section 30-second generation for any adapter:

1. Determine the adapter's section strategy.
2. Render each section on a ComfyUI worker (sequentially), extracting the last
   frame between sections to condition the next one.
3. Concatenate all sections with FFmpeg into ``final.mp4``.
4. Persist frames, thumbnail and metadata via :class:`ProjectManager`.

Progress is published to a per-job dict for WebSocket / polling consumers.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import ffmpeg_utils
from adapters import GenerationParams, ModelAdapter
from comfy_client import (ComfyConnectionError, ComfyExecutionError,
                          WorkerPool, Worker)
from project_manager import ClipRecord, ProjectManager

log = logging.getLogger("simple_comfort.generation")


@dataclass
class GenerationJob:
    job_id: str
    model: str
    mode: str
    status: str = "queued"  # queued|running|completed|failed|cancelled
    progress: float = 0.0
    current_section: int = 0
    total_sections: int = 1
    section_fraction: float = 0.0
    eta: Optional[float] = None
    error: Optional[str] = None
    clip_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    started_at: float = 0.0
    message: str = ""
    preview_b64: Optional[str] = None

    def public(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model": self.model,
            "mode": self.mode,
            "status": self.status,
            "progress": round(self.progress, 4),
            "current_section": self.current_section,
            "total_sections": self.total_sections,
            "section_fraction": round(self.section_fraction, 4),
            "eta": round(self.eta, 1) if self.eta else None,
            "error": self.error,
            "clip_id": self.clip_id,
            "result": self.result,
            "message": self.message,
            "preview": self.preview_b64,
        }


class GenerationManager:
    """Async job queue that runs generations across the worker pool."""

    def __init__(self, pool: WorkerPool, adapters: Dict[str, ModelAdapter],
                 projects: ProjectManager, max_concurrent: int = 2,
                 timeout: int = 300):
        self.pool = pool
        self.adapters = adapters
        self.projects = projects
        self.timeout = timeout
        self.jobs: Dict[str, GenerationJob] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(max_concurrent)
        # progress subscribers: job_id -> list[asyncio.Queue]
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(job_id)
        if subs and q in subs:
            subs.remove(q)

    async def _publish(self, job: GenerationJob) -> None:
        payload = job.public()
        for q in list(self._subscribers.get(job.job_id, [])):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        return self.jobs.get(job_id)

    def submit(self, params: GenerationParams, project_id: str,
               parent_clip_id: Optional[str], worker_index: Optional[int],
               extra_metadata: Dict[str, Any]) -> GenerationJob:
        job_id = "job-" + uuid.uuid4().hex[:12]
        job = GenerationJob(
            job_id=job_id,
            model=params.__dict__.get("_model", ""),
            mode=params.mode,
            status="queued",
        )
        self.jobs[job_id] = job
        self._cancel_events[job_id] = asyncio.Event()
        task = asyncio.create_task(
            self._run_job(job, params, project_id, parent_clip_id,
                          worker_index, extra_metadata)
        )
        self._tasks[job_id] = task
        return job

    async def cancel(self, job_id: str) -> bool:
        event = self._cancel_events.get(job_id)
        job = self.jobs.get(job_id)
        if not event or not job:
            return False
        if job.status in ("completed", "failed", "cancelled"):
            return False
        event.set()
        job.message = "Cancelling..."
        await self._publish(job)
        return True

    # ------------------------------------------------------------------ #
    # Core execution
    # ------------------------------------------------------------------ #
    async def _run_job(self, job: GenerationJob, params: GenerationParams,
                       project_id: str, parent_clip_id: Optional[str],
                       worker_index: Optional[int],
                       extra_metadata: Dict[str, Any]) -> None:
        async with self._sem:
            cancel_event = self._cancel_events[job.job_id]
            adapter = self.adapters[extra_metadata["model_key"]]
            job.model = adapter.MODEL_NAME
            job.started_at = time.time()

            worker: Optional[Worker] = None
            try:
                await self.pool.refresh_health()
                if not self.pool.any_alive():
                    raise ComfyConnectionError(
                        "No ComfyUI endpoint is reachable. Start ComfyUI and "
                        "check COMFY_ENDPOINTS."
                    )
                if not ffmpeg_utils.ffmpeg_available():
                    raise RuntimeError(
                        "FFmpeg is required but was not found on PATH."
                    )

                params.resolved_prompt = adapter.resolve_prompt(params)
                sections = adapter.get_section_strategy(params)
                job.total_sections = len(sections)
                job.status = "running"
                job.message = "Acquiring worker..."
                await self._publish(job)

                worker = await self.pool.acquire(preferred_index=worker_index)
                log.info("Job %s assigned to %s", job.job_id, worker.label)

                clip_type = extra_metadata.get("clip_type", "clip")
                if clip_type in ("broll", "bridge"):
                    clip = self.projects.create_typed_clip(project_id, clip_type)
                else:
                    clip = self.projects.create_clip(project_id)
                job.clip_id = clip.clip_id

                section_files: List[Path] = []
                last_frame_path: Optional[str] = None

                for section in sections:
                    if cancel_event.is_set():
                        raise asyncio.CancelledError()

                    job.current_section = section.index + 1
                    job.section_fraction = 0.0
                    job.message = (f"Rendering section {section.index + 1}/"
                                   f"{len(sections)}")
                    await self._publish(job)

                    section_path = await self._render_section(
                        job, adapter, params, section, worker,
                        last_frame_path, clip, cancel_event
                    )
                    section_files.append(section_path)

                    # Extract the last frame to condition the next section.
                    if section.index < len(sections) - 1:
                        lf = clip.sections_dir / f"section_{section.index:02d}_last.png"
                        await ffmpeg_utils.extract_last_frame(section_path, lf)
                        last_frame_path = str(lf)

                    # Update overall progress after each section.
                    job.progress = (section.index + 1) / len(sections)
                    self._update_eta(job)
                    await self._publish(job)

                # Concatenate all sections.
                job.message = "Concatenating sections..."
                await self._publish(job)
                await ffmpeg_utils.concat_videos(
                    section_files, clip.final_video, fps=params.fps
                )

                # Optional professional post-processing (denoise / interpolation).
                await self._post_process(job, clip, params)

                # Post-processing: frames, thumbnail, metadata.
                await self._finalize_clip(clip, params, adapter, sections,
                                          parent_clip_id, extra_metadata)

                job.result = clip.to_public_dict(self.projects.output_dir)
                job.status = "completed"
                job.progress = 1.0
                job.message = "Done"
                await self._publish(job)
                log.info("Job %s completed -> %s", job.job_id, clip.clip_id)

            except asyncio.CancelledError:
                job.status = "cancelled"
                job.message = "Cancelled"
                job.error = "Generation cancelled by user."
                await self._publish(job)
                log.info("Job %s cancelled", job.job_id)
            except (ComfyConnectionError, ComfyExecutionError) as exc:
                job.status = "failed"
                job.error = str(exc)
                job.message = "Failed"
                await self._publish(job)
                log.error("Job %s failed: %s", job.job_id, exc)
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "Failed"
                await self._publish(job)
                log.exception("Job %s crashed", job.job_id)
            finally:
                if worker is not None:
                    await self.pool.release(worker)
                # Give WS consumers a moment to drain, then clean cancel event.
                self._cancel_events.pop(job.job_id, None)

    async def _render_section(self, job: GenerationJob, adapter: ModelAdapter,
                              params: GenerationParams, section,
                              worker: Worker, last_frame_path: Optional[str],
                              clip: ClipRecord,
                              cancel_event: asyncio.Event) -> Path:
        """Render a single section on a worker and download the result."""
        workflow = adapter.map_parameters(params, section, last_frame_path)

        async def progress_cb(event: Dict[str, Any]) -> None:
            if event.get("type") == "step":
                job.section_fraction = event.get("fraction", 0.0)
                # Blend into overall progress.
                base = section.index / job.total_sections
                job.progress = base + job.section_fraction / job.total_sections
                self._update_eta(job)
                await self._publish(job)
            elif event.get("type") == "preview":
                img = event.get("image_bytes")
                if img:
                    job.preview_b64 = ("data:image/jpeg;base64,"
                                       + base64.b64encode(img).decode("ascii"))
                    await self._publish(job)

        client = worker.client
        prompt_id = await client.queue_prompt(workflow)
        try:
            outputs = await asyncio.wait_for(
                client.poll_until_done(prompt_id, progress_cb, cancel_event),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as exc:
            await client.interrupt()
            raise ComfyExecutionError(
                f"Section {section.index + 1} timed out after {self.timeout}s"
            ) from exc

        files = client.collect_output_files(outputs)
        if not files:
            raise ComfyExecutionError(
                f"Section {section.index + 1} produced no video output. "
                f"Check that the workflow's save node emits a video."
            )

        # Prefer video/gif outputs; fall back to first file.
        media = files[0]
        data = await client.get_image(
            media["filename"], media["subfolder"], media["type"]
        )
        section_path = clip.sections_dir / f"section_{section.index:02d}.mp4"
        section_path.write_bytes(data)
        return section_path

    async def _post_process(self, job: GenerationJob, clip: ClipRecord,
                            params: GenerationParams) -> None:
        """Apply optional professional post-processing to the final video.

        Controlled by ``params.quality_params``:
        * ``denoise`` (truthy)          -> hqdn3d spatial/temporal denoise.
        * ``interpolate_fps`` (int/bool) -> minterpolate up to the target fps.
        """
        qp = params.quality_params or {}
        if not qp:
            return

        # Denoise first, then interpolate (order matters for quality).
        if qp.get("denoise"):
            job.message = "Denoising..."
            await self._publish(job)
            tmp = clip.root_path / "_denoise.mp4"
            try:
                strength = qp.get("denoise_strength") or "2:1:2:3"
                await ffmpeg_utils.denoise_video(
                    clip.final_video, tmp, strength=str(strength)
                )
                tmp.replace(clip.final_video)
                log.info("Job %s denoise applied", job.job_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("Denoise failed for %s: %s", job.job_id, exc)
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

        interp = qp.get("interpolate_fps")
        if interp:
            # Allow interp to be a bool (use 2x current fps) or an explicit fps.
            if isinstance(interp, bool):
                target_fps = int((params.fps or 24) * 2)
            else:
                try:
                    target_fps = int(interp)
                except (TypeError, ValueError):
                    target_fps = int((params.fps or 24) * 2)
            job.message = f"Interpolating to {target_fps}fps..."
            await self._publish(job)
            tmp = clip.root_path / "_interp.mp4"
            try:
                await ffmpeg_utils.interpolate_fps(
                    clip.final_video, tmp, target_fps
                )
                tmp.replace(clip.final_video)
                log.info("Job %s interpolated to %dfps", job.job_id, target_fps)
            except Exception as exc:  # noqa: BLE001
                log.warning("Interpolation failed for %s: %s", job.job_id, exc)
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

    async def _finalize_clip(self, clip: ClipRecord, params: GenerationParams,
                             adapter: ModelAdapter, sections,
                             parent_clip_id: Optional[str],
                             extra_metadata: Dict[str, Any]) -> None:
        # Extract first + last frames and thumbnail from the final video.
        try:
            await ffmpeg_utils.extract_frame(clip.final_video, 0, clip.first_frame)
        except Exception as exc:  # noqa: BLE001
            log.warning("first_frame extraction failed: %s", exc)
        try:
            await ffmpeg_utils.extract_last_frame(clip.final_video, clip.last_frame)
        except Exception as exc:  # noqa: BLE001
            log.warning("last_frame extraction failed: %s", exc)
        try:
            await ffmpeg_utils.create_thumbnail(clip.final_video, clip.thumbnail)
        except Exception as exc:  # noqa: BLE001
            log.warning("thumbnail creation failed: %s", exc)

        probe = {}
        try:
            probe = await ffmpeg_utils.probe_video(clip.final_video)
        except Exception as exc:  # noqa: BLE001
            log.warning("probe failed: %s", exc)

        width, height = adapter.resolve_resolution(params)
        quality = adapter.resolve_quality(params)
        metadata = {
            "model": adapter.MODEL_NAME,
            "model_key": adapter.MODEL_KEY,
            "mode": params.mode,
            "prompt": params.prompt,
            "resolved_prompt": params.resolved_prompt,
            "negative_prompt": params.negative_prompt,
            "seed": params.seed,
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
            "fps": probe.get("fps") or params.fps,
            "duration": probe.get("duration"),
            "codec": probe.get("codec"),
            "aspect_ratio": params.aspect_ratio,
            "quality": params.quality,
            "steps": int(quality["steps"]),
            "guidance": float(quality["guidance"]),
            "lip_sync": params.lip_sync,
            "motion_strength": params.motion_strength,
            "reference_image_info": (
                {"source": params.reference_image} if params.reference_image else None
            ),
            "checkpoint_info": extra_metadata.get("checkpoint_info"),
            "num_sections": len(sections),
            "parent_clip_id": parent_clip_id,
            "generation_settings": extra_metadata.get("generation_settings", {}),
            # Typed-clip + library linkage (b-roll / bridge / avatar / voice).
            "type": extra_metadata.get("clip_type", "clip"),
            "broll_category": extra_metadata.get("broll_category"),
            "avatar_id": extra_metadata.get("avatar_id"),
            "voice_id": extra_metadata.get("voice_id"),
            "variant_index": extra_metadata.get("variant_index"),
            "target_seconds": params.target_seconds,
            "quality_params": params.quality_params or {},
        }
        self.projects.save_clip_metadata(clip, metadata)

    def _update_eta(self, job: GenerationJob) -> None:
        elapsed = time.time() - job.started_at
        if job.progress > 0.01:
            total_est = elapsed / job.progress
            job.eta = max(total_est - elapsed, 0.0)
