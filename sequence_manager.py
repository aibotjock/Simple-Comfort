"""Sequence (timeline) management.

A sequence is an ordered list of clips / b-roll / bridge items belonging to a
project. It is persisted as ``outputs/projects/{project_id}/sequence.json`` and
mirrors the order the user arranges items in the Sequence panel.
"""
from __future__ import annotations

import json
import logging
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("simple_comfort.sequence")


@dataclass
class SequenceItem:
    """One entry in a project's sequence."""

    id: str
    type: str  # clip | broll | bridge
    clip_id: str
    label: str = ""
    thumbnail_path: str = ""
    order_index: int = 0

    def to_public_dict(self, output_root: Path) -> Dict:
        def rel(p: str) -> Optional[str]:
            if not p:
                return None
            try:
                resolved = Path(p).resolve()
                if not resolved.exists():
                    return None
                return "/outputs/" + str(
                    resolved.relative_to(Path(output_root).resolve())
                ).replace("\\", "/")
            except (ValueError, OSError):
                return None

        return {
            "id": self.id,
            "type": self.type,
            "clip_id": self.clip_id,
            "label": self.label,
            "thumbnail_url": rel(self.thumbnail_path),
            "order_index": self.order_index,
        }


class SequenceManager:
    """Read and mutate the per-project sequence file."""

    def __init__(self, output_dir: Path, projects=None):
        self.output_dir = Path(output_dir)
        self.projects_dir = self.output_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.projects = projects

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _sequence_path(self, project_id: str) -> Path:
        return self.projects_dir / project_id / "sequence.json"

    def _load_items(self, project_id: str) -> List[SequenceItem]:
        path = self._sequence_path(project_id)
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load sequence for %s: %s", project_id, exc)
            return []
        items = [
            SequenceItem(
                id=d.get("id", "seq-" + secrets.token_hex(3)),
                type=d.get("type", "clip"),
                clip_id=d.get("clip_id", ""),
                label=d.get("label", ""),
                thumbnail_path=d.get("thumbnail_path", ""),
                order_index=d.get("order_index", 0),
            )
            for d in data.get("items", [])
        ]
        items.sort(key=lambda it: it.order_index)
        return items

    def _save_items(self, project_id: str, items: List[SequenceItem]) -> None:
        for idx, item in enumerate(items):
            item.order_index = idx
        proj_dir = self.projects_dir / project_id
        proj_dir.mkdir(parents=True, exist_ok=True)
        payload = {"project_id": project_id,
                   "items": [asdict(it) for it in items]}
        with self._sequence_path(project_id).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        log.info("Saved sequence for %s (%d items)", project_id, len(items))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_sequence(self, project_id: str) -> List[Dict]:
        items = self._load_items(project_id)
        return [it.to_public_dict(self.output_dir) for it in items]

    def save_sequence(self, project_id: str, items: List[SequenceItem]) -> None:
        self._save_items(project_id, items)

    def _thumbnail_for_clip(self, clip_id: str) -> str:
        if self.projects is None:
            return ""
        clip = self.projects.get_clip(clip_id)
        if clip and clip.thumbnail.exists():
            return str(clip.thumbnail)
        return ""

    def add_to_sequence(self, project_id: str, clip_id: str,
                        item_type: str = "clip",
                        label: str = "") -> List[Dict]:
        items = self._load_items(project_id)
        new_item = SequenceItem(
            id="seq-" + secrets.token_hex(4),
            type=item_type,
            clip_id=clip_id,
            label=label,
            thumbnail_path=self._thumbnail_for_clip(clip_id),
            order_index=len(items),
        )
        items.append(new_item)
        self._save_items(project_id, items)
        return self.get_sequence(project_id)

    def remove_from_sequence(self, project_id: str, item_id: str) -> List[Dict]:
        items = [it for it in self._load_items(project_id) if it.id != item_id]
        self._save_items(project_id, items)
        return self.get_sequence(project_id)

    def move_item(self, project_id: str, item_id: str,
                  direction: str) -> List[Dict]:
        """Move an item one step ``up`` or ``down`` in the sequence."""
        items = self._load_items(project_id)
        idx = next((i for i, it in enumerate(items) if it.id == item_id), None)
        if idx is None:
            return self.get_sequence(project_id)
        if direction == "up" and idx > 0:
            items[idx - 1], items[idx] = items[idx], items[idx - 1]
        elif direction == "down" and idx < len(items) - 1:
            items[idx + 1], items[idx] = items[idx], items[idx + 1]
        self._save_items(project_id, items)
        return self.get_sequence(project_id)

    def reorder(self, project_id: str, ordered_item_ids: List[str]) -> List[Dict]:
        """Reorder the sequence to match ``ordered_item_ids`` exactly.

        Any items not present in the provided list are appended at the end in
        their existing relative order.
        """
        items = self._load_items(project_id)
        by_id = {it.id: it for it in items}
        new_order: List[SequenceItem] = []
        for iid in ordered_item_ids:
            if iid in by_id:
                new_order.append(by_id.pop(iid))
        # Append any leftovers not referenced in the request.
        for it in items:
            if it.id in by_id:
                new_order.append(it)
        self._save_items(project_id, new_order)
        return self.get_sequence(project_id)
