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

    @property
    def narration_path(self) -> Optional[Path]:
        """Return the narration audio file path if recorded in metadata."""
        rel = self.metadata.get("narration_path")
        if rel:
            return Path(rel)
        return None

    @property
    def clip_type(self) -> str:
        return self.metadata.get("type", "clip")

    def ensure_dirs(self) -> None:
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.sections_dir.mkdir(parents=True, exist_ok=True)

    def to_public_dict(self, output_root: Path) -> Dict:
        """Serialisable view with web-relative URLs under /outputs."""
        def rel(p: Optional[Path]) -> Optional[str]:
            if p is None:
                return None
            try:
                return "/outputs/" + str(Path(p).resolve().relative_to(
                    Path(output_root).resolve())).replace("\\", "/")
            except (ValueError, OSError):
                return None

        narration = self.narration_path
        narration_url = None
        if narration and narration.exists():
            narration_url = rel(narration)

        return {
            "clip_id": self.clip_id,
            "project_id": self.project_id,
            "clip_number": self.clip_number,
            "created_at": self.created_at,
            "type": self.clip_type,
            "video_url": rel(self.final_video) if self.final_video.exists() else None,
            "thumbnail_url": rel(self.thumbnail) if self.thumbnail.exists() else None,
            "first_frame_url": rel(self.first_frame) if self.first_frame.exists() else None,
            "last_frame_url": rel(self.last_frame) if self.last_frame.exists() else None,
            "narration_url": narration_url,
            "avatar_id": self.metadata.get("avatar_id"),
            "voice_id": self.metadata.get("voice_id"),
            "broll_category": self.metadata.get("broll_category"),
            "parent_clip_id": self.metadata.get("parent_clip_id"),
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

    def create_typed_clip(self, project_id: str, clip_type: str) -> ClipRecord:
        """Create a b-roll or bridge clip stored under its own subdirectory.

        ``clip_type`` is one of ``broll`` or ``bridge``. The clip id encodes the
        subpath as ``{project_id}__{clip_type}__{token}`` so :meth:`get_clip`
        can resolve it.
        """
        if clip_type not in ("broll", "bridge"):
            return self.create_clip(project_id)
        token = secrets.token_hex(3)
        sub_name = f"{clip_type}-{token}"
        clip_dir = self.projects_dir / project_id / clip_type / sub_name
        clip_id = f"{project_id}__{clip_type}__{sub_name}"
        record = ClipRecord(
            clip_id=clip_id,
            project_id=project_id,
            clip_number=0,
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
        # clip_id formats:
        #   {project_id}__clip-NNN                 (normal clip)
        #   {project_id}__{broll|bridge}__{name}   (typed clip)
        if "__" in clip_id:
            parts = clip_id.split("__")
            if len(parts) == 2:
                clip_dir = self.projects_dir / parts[0] / parts[1]
            elif len(parts) >= 3:
                clip_dir = self.projects_dir / parts[0] / parts[1] / parts[2]
            else:
                clip_dir = self.projects_dir / clip_id
            rec = self._load_clip_dir(clip_dir)
            if rec:
                return rec
        # Fall back to scanning everything.
        for record in self._scan_all_clips():
            if record.clip_id == clip_id:
                return record
        return None

    def get_clips_for_project(self, project_id: str) -> List[ClipRecord]:
        """Return all clips for a project: normal clips, b-roll, and bridges."""
        proj_dir = self.projects_dir / project_id
        records: List[ClipRecord] = []
        if not proj_dir.exists():
            return records
        for clip_dir in proj_dir.glob("clip-*"):
            if clip_dir.is_dir():
                rec = self._load_clip_dir(clip_dir)
                if rec:
                    records.append(rec)
        for sub in ("broll", "bridge"):
            sub_dir = proj_dir / sub
            if sub_dir.exists():
                for clip_dir in sub_dir.iterdir():
                    if clip_dir.is_dir():
                        rec = self._load_clip_dir(clip_dir)
                        if rec:
                            records.append(rec)
        records.sort(key=lambda r: r.created_at)
        return records

    def _scan_all_clips(self) -> List[ClipRecord]:
        records: List[ClipRecord] = []
        for proj_dir in self.projects_dir.iterdir():
            if proj_dir.is_dir():
                records.extend(self.get_clips_for_project(proj_dir.name))
        return records

    def merge_clip_metadata(self, clip_id: str, updates: Dict) -> Optional[ClipRecord]:
        """Merge ``updates`` into an existing clip's metadata.json."""
        clip = self.get_clip(clip_id)
        if not clip:
            return None
        merged = {**clip.metadata, **updates}
        self.save_clip_metadata(clip, merged)
        return clip

    def save_anneal_metadata(self, clip_id: str, anneal_data: Dict) -> Optional[ClipRecord]:
        """Store annealing / continuity data on a clip's metadata."""
        return self.merge_clip_metadata(clip_id, {"anneal_metadata": anneal_data})

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
