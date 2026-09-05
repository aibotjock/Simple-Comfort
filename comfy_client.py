"""ComfyUI HTTP + WebSocket client and a simple worker pool.

Each :class:`ComfyUIClient` talks to one ComfyUI endpoint. The
:class:`WorkerPool` distributes work across all configured endpoints using a
least-busy strategy, treating each endpoint as an independent GPU worker.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

log = logging.getLogger("simple_comfort.comfy")

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class ComfyConnectionError(RuntimeError):
    """Raised when a ComfyUI endpoint cannot be reached."""


class ComfyExecutionError(RuntimeError):
    """Raised when ComfyUI reports an execution error for a prompt."""


class ComfyUIClient:
    """HTTP + WebSocket client for a single ComfyUI endpoint."""

    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Stable client id so ComfyUI associates our WS with our prompts.
        self.client_id = uuid.uuid4().hex
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=10)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def ws_url(self) -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://", 1)[-1]
        return f"{scheme}://{host}/ws?clientId={self.client_id}"

    # ------------------------------------------------------------------ #
    # Basic HTTP calls
    # ------------------------------------------------------------------ #
    async def is_alive(self) -> bool:
        """Return True if the endpoint responds to a stats query."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/system_stats",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except Exception as exc:  # noqa: BLE001
            log.debug("Endpoint %s not alive: %s", self.base_url, exc)
            return False

    async def get_object_info(self) -> Dict[str, Any]:
        """Return /object_info (all available node classes)."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/object_info",
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientError as exc:
            raise ComfyConnectionError(
                f"Could not fetch object_info from {self.base_url}: {exc}"
            ) from exc

    async def get_models(self, folder: str) -> List[str]:
        """Return the list of files for a given model folder (e.g. 'checkpoints')."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/models/{folder}",
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return []
                return await resp.json()
        except aiohttp.ClientError:
            return []

    async def queue_prompt(self, workflow_json: Dict[str, Any]) -> str:
        """Queue a workflow (API-format dict); return its prompt_id."""
        session = await self._get_session()
        payload = {"prompt": workflow_json, "client_id": self.client_id}
        try:
            async with session.post(
                f"{self.base_url}/prompt", json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise ComfyExecutionError(
                        f"ComfyUI rejected the workflow (HTTP {resp.status}): "
                        f"{text[:500]}"
                    )
                data = json.loads(text)
                prompt_id = data.get("prompt_id")
                if not prompt_id:
                    raise ComfyExecutionError(
                        f"ComfyUI did not return a prompt_id: {text[:300]}"
                    )
                return prompt_id
        except aiohttp.ClientError as exc:
            raise ComfyConnectionError(
                f"Failed to queue prompt on {self.base_url}: {exc}"
            ) from exc

    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/history/{prompt_id}",
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_image(self, filename: str, subfolder: str = "",
                        type_: str = "output") -> bytes:
        """Download a produced image/video file from ComfyUI."""
        session = await self._get_session()
        params = {"filename": filename, "subfolder": subfolder, "type": type_}
        async with session.get(
            f"{self.base_url}/view", params=params,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def interrupt(self) -> None:
        """Interrupt the currently executing prompt."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/interrupt",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                await resp.read()
        except aiohttp.ClientError as exc:
            log.warning("Interrupt failed on %s: %s", self.base_url, exc)

    # ------------------------------------------------------------------ #
    # Execution + progress streaming
    # ------------------------------------------------------------------ #
    async def poll_until_done(
        self,
        prompt_id: str,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """Listen on the WebSocket until the prompt finishes.

        Forwards ComfyUI execution events (progress, preview images) to
        ``progress_cb``. Returns the history outputs for ``prompt_id``.

        Three concurrent watchers race so nothing can hang forever: the WS
        event stream, the caller's cancel event (which otherwise can't fire
        while no WS traffic arrives), and a periodic history poll that
        catches prompts whose completion event was missed or that were
        removed from the server queue entirely.
        """
        session = await self._get_session()
        try:
            ws = await session.ws_connect(
                self.ws_url, heartbeat=20, timeout=aiohttp.ClientTimeout(total=30)
            )
        except aiohttp.ClientError as exc:
            raise ComfyConnectionError(
                f"Could not open WebSocket to {self.base_url}: {exc}"
            ) from exc

        async def consume_ws() -> bool:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    done = await self._handle_ws_event(
                        event, prompt_id, progress_cb
                    )
                    if done:
                        return True
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Binary payload = preview image (8-byte header then
                    # PNG/JPEG).
                    if progress_cb and len(msg.data) > 8:
                        await progress_cb({
                            "type": "preview",
                            "image_bytes": msg.data[8:],
                        })
                elif msg.type in (aiohttp.WSMsgType.CLOSED,
                                  aiohttp.WSMsgType.ERROR):
                    return False
            return False

        async def history_poll() -> bool:
            # Fallback: the prompt may finish (or vanish) with no WS event
            # ever reaching us — poll its history entry periodically.
            while True:
                await asyncio.sleep(5)
                try:
                    history = await self.get_history(prompt_id)
                    if prompt_id in history:
                        entry = history[prompt_id]
                        status = (entry.get("status") or {})
                        # Present but still running -> keep waiting.
                        if status.get("completed") or \
                                status.get("status_str") in ("error", "cancelled"):
                            return True
                except Exception:  # noqa: BLE001
                    pass

        try:
            reader = asyncio.create_task(consume_ws())
            watchers = [reader, asyncio.create_task(history_poll())]
            cancel_waiter = None
            if cancel_event is not None:
                cancel_waiter = asyncio.create_task(cancel_event.wait())
                watchers.append(cancel_waiter)

            done, pending = await asyncio.wait(
                watchers, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if cancel_waiter in done and cancel_event.is_set():
                await self.interrupt()
                raise asyncio.CancelledError()
        finally:
            await ws.close()

        history = await self.get_history(prompt_id)
        entry = history.get(prompt_id, {})
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            messages = status.get("messages", [])
            raise ComfyExecutionError(
                f"ComfyUI execution error: {messages}"
            )
        return entry.get("outputs", {})

    async def _handle_ws_event(
        self,
        event: Dict[str, Any],
        prompt_id: str,
        progress_cb: Optional[ProgressCallback],
    ) -> bool:
        """Process one WS event. Return True when this prompt is complete."""
        etype = event.get("type")
        data = event.get("data", {})

        if etype == "progress":
            if progress_cb:
                value = data.get("value", 0)
                maximum = data.get("max", 1) or 1
                await progress_cb({
                    "type": "step",
                    "value": value,
                    "max": maximum,
                    "fraction": value / maximum,
                })
        elif etype == "executing":
            node = data.get("node")
            if node is None and data.get("prompt_id") == prompt_id:
                # node == None signals this prompt is finished.
                return True
        elif etype == "execution_error":
            if data.get("prompt_id") == prompt_id:
                raise ComfyExecutionError(
                    f"Node error: {data.get('exception_message', 'unknown')}"
                )
        elif etype == "execution_cached":
            pass
        return False

    def collect_output_files(self, outputs: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract produced media file references from a history outputs dict."""
        files: List[Dict[str, str]] = []
        for node_output in outputs.values():
            for key in ("gifs", "videos", "images"):
                for item in node_output.get(key, []) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        files.append({
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        })
        return files


# ---------------------------------------------------------------------- #
# Worker pool
# ---------------------------------------------------------------------- #
@dataclass
class Worker:
    index: int
    client: ComfyUIClient
    busy: bool = False
    alive: bool = False
    label: str = ""
    jobs_completed: int = 0


class WorkerPool:
    """Manages one :class:`ComfyUIClient` per configured endpoint."""

    def __init__(self, endpoints: List[str], timeout: int = 300):
        self.workers: List[Worker] = []
        for i, ep in enumerate(endpoints):
            self.workers.append(
                Worker(index=i, client=ComfyUIClient(ep, timeout=timeout),
                       label=f"GPU {i}")
            )
        self._lock = asyncio.Lock()
        # Round-robin cursor: which alive worker to hand out next when no
        # preference is given. Ensures jobs strictly alternate across GPUs.
        self._next_worker_index = 0

    @property
    def is_multi(self) -> bool:
        return len(self.workers) > 1

    async def refresh_health(self) -> None:
        """Ping all endpoints and update their alive flag."""
        results = await asyncio.gather(
            *(w.client.is_alive() for w in self.workers),
            return_exceptions=True,
        )
        for worker, res in zip(self.workers, results):
            worker.alive = res is True

    def any_alive(self) -> bool:
        return any(w.alive for w in self.workers)

    async def acquire(self, preferred_index: Optional[int] = None,
                      wait: bool = True) -> Worker:
        """Acquire a free, alive worker using strict round-robin.

        When ``preferred_index`` is None, jobs are handed out in strict
        round-robin order across the alive workers (job 1 -> GPU 0,
        job 2 -> GPU 1, job 3 -> GPU 0, ...). If the round-robin target is
        busy, any other free alive worker is used as a fallback rather than
        blocking. When ``preferred_index`` is given, that worker is used if
        free, otherwise another free alive worker is chosen.
        """
        while True:
            async with self._lock:
                free = [w for w in self.workers if w.alive and not w.busy]
                if free:
                    worker = None
                    if preferred_index is not None:
                        # Honour the caller's preference when possible.
                        worker = next(
                            (w for w in free if w.index == preferred_index),
                            None,
                        )
                    else:
                        # Strict round-robin over the alive workers.
                        alive = [w for w in self.workers if w.alive]
                        if alive:
                            target = self._next_worker_index % len(alive)
                            desired = alive[target]
                            # Advance the cursor so the next job goes elsewhere.
                            self._next_worker_index = (
                                self._next_worker_index + 1
                            ) % len(alive)
                            if not desired.busy:
                                worker = desired
                    # Fallback: any free alive worker if the target was busy.
                    if worker is None:
                        worker = free[0]
                    worker.busy = True
                    return worker
            if not wait:
                raise ComfyConnectionError("No available ComfyUI worker")
            await asyncio.sleep(0.5)

    async def release(self, worker: Worker) -> None:
        async with self._lock:
            worker.busy = False
            worker.jobs_completed += 1

    def status(self) -> List[Dict[str, Any]]:
        return [
            {
                "index": w.index,
                "label": w.label,
                "endpoint": w.client.base_url,
                "alive": w.alive,
                "busy": w.busy,
                "jobs_completed": w.jobs_completed,
            }
            for w in self.workers
        ]

    async def close(self) -> None:
        await asyncio.gather(*(w.client.close() for w in self.workers),
                             return_exceptions=True)
