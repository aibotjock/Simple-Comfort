"""Model / capability detection.

Queries a ComfyUI endpoint's ``/object_info`` and ``/models/*`` endpoints and,
for each registered adapter, determines whether all required checkpoints, VAEs,
text encoders and custom nodes are present. Results are cached for a short TTL.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from comfy_client import ComfyUIClient, WorkerPool

log = logging.getLogger("simple_comfort.detector")


@dataclass
class ModelStatus:
    """Detection result for a single model adapter."""

    model: str
    display_name: str
    status: str = "unknown"  # available | missing_checkpoint | missing_vae |
    #                          missing_text_encoder | missing_nodes | unreachable
    available: bool = False
    t2v_capable: bool = False
    i2v_capable: bool = False
    missing: List[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelDetector:
    """Detects installed models across the worker pool with caching."""

    def __init__(self, pool: WorkerPool, adapters: Dict, cache_ttl: int = 60):
        self.pool = pool
        self.adapters = adapters  # name -> ModelAdapter instance
        self.cache_ttl = cache_ttl
        self._cache: Optional[List[ModelStatus]] = None
        self._cache_time: float = 0.0

    def invalidate(self) -> None:
        self._cache = None
        self._cache_time = 0.0

    async def detect_all(self, force: bool = False) -> List[ModelStatus]:
        now = time.time()
        if (not force and self._cache is not None
                and now - self._cache_time < self.cache_ttl):
            return self._cache

        await self.pool.refresh_health()
        results: List[ModelStatus] = []

        # Use the first alive worker as the detection source of truth.
        alive_worker = next((w for w in self.pool.workers if w.alive), None)

        if alive_worker is None:
            for name, adapter in self.adapters.items():
                results.append(ModelStatus(
                    model=name,
                    display_name=adapter.MODEL_NAME,
                    status="unreachable",
                    detail="No ComfyUI endpoint is reachable.",
                ))
            self._cache = results
            self._cache_time = now
            return results

        client = alive_worker.client
        try:
            object_info = await client.get_object_info()
        except Exception as exc:  # noqa: BLE001
            log.warning("object_info fetch failed: %s", exc)
            object_info = {}

        # Pre-fetch common model folders once.
        folders = ["checkpoints", "vae", "text_encoders", "clip", "unet",
                   "diffusion_models", "loras"]
        model_files: Dict[str, List[str]] = {}
        for folder in folders:
            model_files[folder] = await client.get_models(folder)

        for name, adapter in self.adapters.items():
            try:
                status = adapter.detect_capabilities(object_info, model_files)
                status.model = name
                status.display_name = adapter.MODEL_NAME
            except Exception as exc:  # noqa: BLE001
                log.exception("Detection failed for %s", name)
                status = ModelStatus(
                    model=name,
                    display_name=adapter.MODEL_NAME,
                    status="unknown",
                    detail=f"Detection error: {exc}",
                )
            results.append(status)

        self._cache = results
        self._cache_time = now
        return results

    async def get_status(self, model: str) -> Optional[ModelStatus]:
        for status in await self.detect_all():
            if status.model == model:
                return status
        return None
