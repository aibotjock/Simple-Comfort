"""Project / clip persistence.

Everything is stored on disk as plain files — no database. Layout::

    outputs/projects/{PROJECT-ID}/clip-001/
        final.mp4
        metadata.json
        first_frame.png
        last_frame.png
        thumbnail.jpg
        sections/
"""
from __future__ import annotations

import json
import logging
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("simple_comfort.projects")


@dataclass
class ClipRecord:
    """A single generated 30-second clip and all its file paths."""

    clip_id: str
    project_id: str
    clip_number: int
    root: str  # directory path (string for JSON friendliness)
    created_at: str = ""
    metadata: Dict = field(default_factory=dict)

    # --- derived paths -------------------------------------------------- #
    @property
    def root_path(self) -> Path:
        return Path(self.root)

    @property
    def final_video(self) -> Path:
        return self.root_path / "final.mp4"

    @property
    def metadata_path(self) -> Path:
        return self.root_path / "metadata.json"

    @property
    def first_frame(self) -> Path:
        return self.root_path / "first_frame.png"

    @property
    def last_frame(self) -> Path:
        return self.root_path / "last_frame.png"

    @property
    def thumbnail(self) -> Path:
        return self.root_path / "thumbnail.jpg"

    @property
    def sections_dir(self) -> Path:
        return self.root_path / "sections"

    def ensure_dirs(self) -> None:
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.sections_dir.mkdir(parents=True, exist_ok=True)

    def to_public_dict(self, output_root: Path) -> Dict:
        """Serialisable view with web-relative URLs under /outputs."""
        def rel(p: Path) -> Optional[str]:
            try:
                return "/outputs/" + str(p.relative_to(output_root)).replace("\\", "/")
            except ValueError:
                return None

        return {
            "clip_id": self.clip_id,
            "project_id": self.project_id,
            "clip_number": self.clip_number,
            "created_at": self.created_at,
            "video_url": rel(self.final_video) if self.final_video.exists() else None,
            "thumbnail_url": rel(self.thumbnail) if self.thumbnail.exists() else None,
            "first_frame_url": rel(self.first_frame) if self.first_frame.exists() else None,
            "last_frame_url": rel(self.last_frame) if self.last_frame.exists() else None,
            "root": self.root,
            "metadata": self.metadata,
        }


class ProjectManager:
    """Creates and looks up projects and clips on disk."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.projects_dir = self.output_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Creation
    # ------------------------------------------------------------------ #
    def create_project(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d")
        token = secrets.token_hex(3)
        project_id = f"proj-{stamp}-{token}"
        (self.projects_dir / project_id).mkdir(parents=True, exist_ok=True)
        log.info("Created project %s", project_id)
        return project_id

    def _next_clip_number(self, project_id: str) -> int:
        proj_dir = self.projects_dir / project_id
        if not proj_dir.exists():
            return 1
        existing = [
            int(p.name.split("-")[1])
            for p in proj_dir.glob("clip-*")
            if p.is_dir() and p.name.split("-")[-1].isdigit()
        ]
        return (max(existing) + 1) if existing else 1

    def create_clip(self, project_id: str,
                    clip_number: Optional[int] = None) -> ClipRecord:
        if clip_number is None:
            clip_number = self._next_clip_number(project_id)
        clip_dir = self.projects_dir / project_id / f"clip-{clip_number:03d}"
        clip_id = f"{project_id}__clip-{clip_number:03d}"
        record = ClipRecord(
            clip_id=clip_id,
            project_id=project_id,
            clip_number=clip_number,
            root=str(clip_dir),
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        record.ensure_dirs()
        return record

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def save_clip_metadata(self, clip: ClipRecord, metadata: Dict) -> None:
        clip.metadata = metadata
        payload = {
            "clip_id": clip.clip_id,
            "project_id": clip.project_id,
            "clip_number": clip.clip_number,
            "created_at": clip.created_at,
            **metadata,
        }
        clip.ensure_dirs()
        with clip.metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        log.info("Saved metadata for %s", clip.clip_id)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def _load_clip_dir(self, clip_dir: Path) -> Optional[ClipRecord]:
        meta_path = clip_dir / "metadata.json"
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
        record = ClipRecord(
            clip_id=data.get("clip_id", clip_dir.name),
            project_id=data.get("project_id", clip_dir.parent.name),
            clip_number=data.get("clip_number", 0),
            root=str(clip_dir),
            created_at=data.get("created_at", ""),
            metadata={k: v for k, v in data.items()
                      if k not in {"clip_id", "project_id", "clip_number",
                                   "created_at"}},
        )
        return record

    def get_clip(self, clip_id: str) -> Optional[ClipRecord]:
        # clip_id format: {project_id}__clip-NNN
        if "__" in clip_id:
            project_id, clip_name = clip_id.split("__", 1)
            clip_dir = self.projects_dir / project_id / clip_name
            return self._load_clip_dir(clip_dir)
        # Fall back to scanning.
        for record in self.list_clips():
            if record.clip_id == clip_id:
                return record
        return None

    def list_clips(self, project_id: Optional[str] = None) -> List[ClipRecord]:
        records: List[ClipRecord] = []
        if project_id:
            search_dirs = [self.projects_dir / project_id]
        else:
            search_dirs = [p for p in self.projects_dir.iterdir() if p.is_dir()]

        for proj_dir in search_dirs:
            if not proj_dir.exists():
                continue
            for clip_dir in proj_dir.glob("clip-*"):
                if clip_dir.is_dir():
                    rec = self._load_clip_dir(clip_dir)
                    if rec:
                        records.append(rec)

        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    def get_last_clip(self) -> Optional[ClipRecord]:
        clips = self.list_clips()
        return clips[0] if clips else None
