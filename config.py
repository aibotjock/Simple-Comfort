"""Application configuration.

All settings are configurable via environment variables (or a local ``.env``
file). Simple-Comfort is intentionally local-only: no cloud calls, no API keys,
no telemetry.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Comma-separated list of ComfyUI endpoints. Each endpoint is treated as an
    # independent worker (multi-GPU support).
    COMFY_ENDPOINTS: str = "http://127.0.0.1:8188"

    # Web server bind port (always bound to localhost).
    PORT: int = 7788
    HOST: str = "127.0.0.1"

    # Base directory for all generated output.
    OUTPUT_DIR: Path = Path("outputs")

    # Logging level (DEBUG, INFO, WARNING, ERROR).
    LOG_LEVEL: str = "INFO"

    # Maximum number of generation jobs running concurrently.
    MAX_CONCURRENT_JOBS: int = 2

    # Timeout (seconds) when talking to a single ComfyUI section render.
    COMFY_TIMEOUT: int = 300

    # Seconds to cache model-detection results.
    MODEL_CACHE_TTL: int = 60

    @field_validator("OUTPUT_DIR", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        return Path(v)

    @property
    def endpoints(self) -> List[str]:
        """Parsed, de-duplicated list of ComfyUI endpoint URLs."""
        seen: List[str] = []
        for raw in self.COMFY_ENDPOINTS.split(","):
            url = raw.strip().rstrip("/")
            if url and url not in seen:
                seen.append(url)
        return seen

    @property
    def projects_dir(self) -> Path:
        return self.OUTPUT_DIR / "projects"

    @property
    def is_multi_gpu(self) -> bool:
        return len(self.endpoints) > 1


settings = Settings()


def configure_logging() -> None:
    """Configure root logging based on ``LOG_LEVEL``."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party loggers unless we are debugging.
    if level > logging.DEBUG:
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)


def ensure_directories() -> None:
    """Create runtime output directories if they do not yet exist."""
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
