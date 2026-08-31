"""B-roll generation helpers.

B-roll clips are short (8-10 second) supporting shots generated with the same
ComfyUI job pipeline as normal clips, but with a category-driven prompt and a
shorter target duration. Prompt composition is fully deterministic — no LLM is
involved — so the same request always yields the same prompt text.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

log = logging.getLogger("simple_comfort.broll")

# Default target duration for a b-roll shot, in seconds.
BROLL_TARGET_SECONDS = 9.0
BROLL_MIN_SECONDS = 8.0
BROLL_MAX_SECONDS = 10.0


class BRollCategory(str, Enum):
    """Supported b-roll categories."""

    CINEMATIC = "cinematic"
    EXPLANATORY = "explanatory"
    PRODUCT = "product"
    ENVIRONMENT = "environment"
    TECHNOLOGY = "technology"
    ABSTRACT = "abstract"
    TRANSITION = "transition"


# Deterministic prompt prefixes appended in front of the user's subject text.
CATEGORY_PROMPT_PREFIXES: Dict[BRollCategory, str] = {
    BRollCategory.CINEMATIC: (
        "cinematic b-roll footage, shallow depth of field, dramatic lighting, "
        "smooth slow camera movement, film grain, professional color grading"
    ),
    BRollCategory.EXPLANATORY: (
        "clean explanatory b-roll, bright even lighting, clear focus, "
        "informative composition, steady framing, neutral background"
    ),
    BRollCategory.PRODUCT: (
        "product b-roll shot, studio lighting, glossy surfaces, "
        "macro detail, slow rotating camera, crisp reflections, clean backdrop"
    ),
    BRollCategory.ENVIRONMENT: (
        "establishing environment b-roll, wide angle, natural lighting, "
        "atmospheric depth, gentle parallax camera motion"
    ),
    BRollCategory.TECHNOLOGY: (
        "high-tech b-roll, sleek modern surfaces, glowing accents, "
        "cool color palette, precise camera glide, futuristic mood"
    ),
    BRollCategory.ABSTRACT: (
        "abstract motion b-roll, flowing shapes, soft gradients, "
        "bokeh particles, dreamy slow movement, artistic composition"
    ),
    BRollCategory.TRANSITION: (
        "transition b-roll, motion blur, sweeping camera whip, "
        "light streaks, dynamic movement designed to bridge two scenes"
    ),
}

# Shared quality guidance that keeps b-roll free of talking-head artifacts.
BROLL_SUFFIX = (
    "no people talking to camera, no subtitles, no text overlays, "
    "high detail, sharp focus, stable footage"
)


@dataclass
class BRollRequest:
    """A request to generate one or more b-roll variants."""

    project_id: str
    category: BRollCategory
    subject: str
    aspect_ratio: str = "16:9"
    quality: str = "balanced"
    variants: int = 1
    seed: Optional[int] = None
    model: str = ""
    target_seconds: float = BROLL_TARGET_SECONDS


def _coerce_category(category) -> BRollCategory:
    if isinstance(category, BRollCategory):
        return category
    try:
        return BRollCategory(str(category).lower())
    except ValueError as exc:
        valid = ", ".join(c.value for c in BRollCategory)
        raise ValueError(
            f"Unknown b-roll category '{category}'. Valid: {valid}"
        ) from exc


def compose_broll_prompt(category, subject: str) -> str:
    """Deterministically build a b-roll prompt from a category + subject.

    The output is ``"{category prefix}, {subject}, {shared suffix}"`` with
    empty parts removed. No randomness or model calls are involved.
    """
    cat = _coerce_category(category)
    prefix = CATEGORY_PROMPT_PREFIXES[cat]
    subject = (subject or "").strip().rstrip(".,")
    parts = [prefix]
    if subject:
        parts.append(subject)
    parts.append(BROLL_SUFFIX)
    return ", ".join(p for p in parts if p)


def clamp_target_seconds(seconds: float) -> float:
    """Clamp a requested duration into the allowed b-roll range."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return BROLL_TARGET_SECONDS
    return max(BROLL_MIN_SECONDS, min(BROLL_MAX_SECONDS, value))


def prepare_generation_params(request: BRollRequest) -> List[Dict]:
    """Return one param dict per variant, ready to hand to the generator.

    Each dict contains the resolved prompt, the seed (base, base+1, base+2 for
    up to three variants), the target duration and metadata marking the clip as
    b-roll. The caller maps these onto :class:`GenerationParams` and submits a
    normal generation job.
    """
    cat = _coerce_category(request.category)
    variants = max(1, min(3, int(request.variants or 1)))
    prompt = compose_broll_prompt(cat, request.subject)
    target = clamp_target_seconds(request.target_seconds)

    base_seed = request.seed
    param_sets: List[Dict] = []
    for i in range(variants):
        seed: Optional[int] = None
        if base_seed is not None:
            seed = int(base_seed) + i
        param_sets.append({
            "prompt": prompt,
            "mode": "t2v",
            "aspect_ratio": request.aspect_ratio,
            "quality": request.quality,
            "seed": seed,
            "target_seconds": target,
            "variant_index": i,
            "clip_type": "broll",
            "broll_category": cat.value,
            "model": request.model,
            "lip_sync": False,
        })
    log.info(
        "Prepared %d b-roll variant(s) for project %s (category=%s)",
        len(param_sets), request.project_id, cat.value,
    )
    return param_sets


def list_categories() -> List[Dict]:
    """Return category metadata for the UI (value + human label + prefix)."""
    labels = {
        BRollCategory.CINEMATIC: "Cinematic",
        BRollCategory.EXPLANATORY: "Explanatory",
        BRollCategory.PRODUCT: "Product",
        BRollCategory.ENVIRONMENT: "Environment",
        BRollCategory.TECHNOLOGY: "Technology",
        BRollCategory.ABSTRACT: "Abstract",
        BRollCategory.TRANSITION: "Transition",
    }
    return [
        {
            "value": cat.value,
            "label": labels[cat],
            "prompt_prefix": CATEGORY_PROMPT_PREFIXES[cat],
        }
        for cat in BRollCategory
    ]
