"""Model adapters registry."""
from __future__ import annotations

from typing import Dict

from .base import (ModelAdapter, GenerationParams, SectionConfig,
                   ModelCapabilities)
from .ltx23 import LTX23Adapter
from .wan22 import Wan22Adapter
from .framepack import FramePackAdapter


def build_adapters() -> Dict[str, ModelAdapter]:
    """Instantiate every adapter, keyed by its model key."""
    adapters = [LTX23Adapter(), Wan22Adapter(), FramePackAdapter()]
    return {a.MODEL_KEY: a for a in adapters}


__all__ = [
    "ModelAdapter",
    "GenerationParams",
    "SectionConfig",
    "ModelCapabilities",
    "LTX23Adapter",
    "Wan22Adapter",
    "FramePackAdapter",
    "build_adapters",
]
