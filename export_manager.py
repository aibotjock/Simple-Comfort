"""Export / handoff packaging.

Bundles the artefacts of a project into a single ``.zip`` the user can hand off
to a lip-sync tool or archive. Three export types are supported:

* ``VIDEO_ONLY``      — just the final MP4s of the requested clips.
* ``LIPSYNC_PACKAGE`` — final MP4s + last frames + any narration audio +
  a manifest, laid out for a lip-sync pipeline.
* ``PROJECT_PACKAGE`` — everything: videos, frames, narration, per-clip
  metadata and a top-level manifest.
"""
from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("simple_comfort.export")


class ExportType(str, Enum):
    VIDEO_ONLY = "video_only"
    LIPSYNC_PACKAGE = "lipsync_package"
    PROJECT_PACKAGE = "project_package"


def _coerce_type(export_type) -> ExportType:
    if isinstance(export_type, ExportType):
        return export_type
    try:
        return ExportType(str(export_type).lower())
    except ValueError as exc:
        valid = ", ".join(t.value for t in ExportType)
        raise ValueError(
            f"Unknown export type '{export_type}'. Valid: {valid}"
        ) from exc


class ExportManager:
    """Create export zip archives from project clips."""

    def __init__(self, output_dir: Path, projects):
        self.output_dir = Path(output_dir)
        self.projects = projects
        self.exports_dir = self.output_dir / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def _gather_clips(self, project_id: str,
                      clip_ids: Optional[List[str]]) -> List:
        if clip_ids:
            clips = []
            for cid in clip_ids:
                clip = self.projects.get_clip(cid)
                if clip:
                    clips.append(clip)
            return clips
        return self.projects.get_clips_for_project(project_id)

    def create_export(self, project_id: str,
                      clip_ids: Optional[List[str]],
                      export_type,
                      output_dir: Optional[Path] = None) -> Path:
        """Build a zip archive and return its path.

        Raises ``ValueError`` if no clips with a final video are found.
        """
        etype = _coerce_type(export_type)
        clips = self._gather_clips(project_id, clip_ids)
        clips = [c for c in clips if c.final_video.exists()]
        if not clips:
            raise ValueError(
                "No finished clips with video were found for this export."
            )

        dest_dir = Path(output_dir) if output_dir else self.exports_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        zip_path = dest_dir / f"simple-comfort-export-{timestamp}.zip"

        manifest: Dict = {
            "project_id": project_id,
            "export_type": etype.value,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "clip_count": len(clips),
            "clips": [],
        }

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, clip in enumerate(clips, start=1):
                base = f"clip-{idx:02d}_{clip.clip_id}"
                clip_entry: Dict = {
                    "clip_id": clip.clip_id,
                    "type": clip.clip_type,
                    "folder": base,
                    "files": [],
                }

                # Final video — always included.
                video_arc = f"{base}/final.mp4"
                zf.write(clip.final_video, video_arc)
                clip_entry["files"].append(video_arc)

                if etype in (ExportType.LIPSYNC_PACKAGE,
                             ExportType.PROJECT_PACKAGE):
                    if clip.last_frame.exists():
                        arc = f"{base}/last_frame.png"
                        zf.write(clip.last_frame, arc)
                        clip_entry["files"].append(arc)
                    if clip.first_frame.exists():
                        arc = f"{base}/first_frame.png"
                        zf.write(clip.first_frame, arc)
                        clip_entry["files"].append(arc)
                    narration = clip.narration_path
                    if narration and Path(narration).exists():
                        arc = f"{base}/{Path(narration).name}"
                        zf.write(narration, arc)
                        clip_entry["files"].append(arc)
                        clip_entry["narration"] = arc

                if etype == ExportType.PROJECT_PACKAGE:
                    if clip.thumbnail.exists():
                        arc = f"{base}/thumbnail.jpg"
                        zf.write(clip.thumbnail, arc)
                        clip_entry["files"].append(arc)
                    if clip.metadata_path.exists():
                        arc = f"{base}/metadata.json"
                        zf.write(clip.metadata_path, arc)
                        clip_entry["files"].append(arc)

                clip_entry["metadata"] = clip.metadata
                manifest["clips"].append(clip_entry)

            # Include the project sequence if present (project package only).
            if etype == ExportType.PROJECT_PACKAGE:
                seq_path = (self.output_dir / "projects" / project_id
                            / "sequence.json")
                if seq_path.exists():
                    zf.write(seq_path, "sequence.json")
                    manifest["sequence_included"] = True

            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        log.info(
            "Created export %s (%s, %d clips)",
            zip_path.name, etype.value, len(clips),
        )
        return zip_path
