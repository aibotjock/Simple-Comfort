"""Voice library management.

Stores reusable voice samples on disk under ``outputs/voices/{voice_id}/``.
Each voice has an audio sample (wav / mp3 / m4a), an optional linked avatar and
free-form notes. A voice can also be attached to a specific clip as its
narration track, which copies the audio into the clip directory and records the
``narration_path`` + ``voice_id`` in the clip metadata.
"""
from __future__ import annotations

import json
import logging
import secrets
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("simple_comfort.voices")

# Accepted audio containers -> known magic byte signatures. m4a uses the ISO
# base media format ("ftyp" box) which appears at offset 4.
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a"}
_MAGIC_SIGNATURES = {
    ".wav": [b"RIFF"],
    ".mp3": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"],
    ".m4a": [b"ftyp"],  # checked at offset 4
}


@dataclass
class Voice:
    """A reusable voice sample."""

    id: str
    name: str = ""
    sample_path: str = ""
    sample_filename: str = ""
    avatar_id: Optional[str] = None
    notes: str = ""
    created_at: str = ""
    duration_seconds: Optional[float] = None

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
            "name": self.name,
            "sample_url": rel(self.sample_path),
            "sample_filename": self.sample_filename,
            "avatar_id": self.avatar_id,
            "notes": self.notes,
            "created_at": self.created_at,
            "duration_seconds": self.duration_seconds,
        }


class VoiceManager:
    """Create, read, update and delete voice samples stored on disk."""

    def __init__(self, output_dir: Path, projects=None):
        self.output_dir = Path(output_dir)
        self.voices_dir = self.output_dir / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        # Optional ProjectManager, used for narration assignment.
        self.projects = projects

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _voice_dir(self, voice_id: str) -> Path:
        return self.voices_dir / voice_id

    def _meta_path(self, voice_id: str) -> Path:
        return self._voice_dir(voice_id) / "voice.json"

    def _load(self, voice_id: str) -> Optional[Voice]:
        meta = self._meta_path(voice_id)
        if not meta.exists():
            return None
        try:
            with meta.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load voice %s: %s", voice_id, exc)
            return None
        return Voice(
            id=data.get("id", voice_id),
            name=data.get("name", ""),
            sample_path=data.get("sample_path", ""),
            sample_filename=data.get("sample_filename", ""),
            avatar_id=data.get("avatar_id"),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", ""),
            duration_seconds=data.get("duration_seconds"),
        )

    def _save(self, voice: Voice) -> None:
        self._voice_dir(voice.id).mkdir(parents=True, exist_ok=True)
        with self._meta_path(voice.id).open("w", encoding="utf-8") as fh:
            json.dump(asdict(voice), fh, indent=2)
        log.info("Saved voice %s", voice.id)

    @staticmethod
    def _validate_audio(ext: str, data: bytes) -> None:
        ext = ext.lower()
        if ext not in ALLOWED_AUDIO_EXTS:
            raise ValueError(
                f"Unsupported audio type '{ext}'. Allowed: "
                + ", ".join(sorted(ALLOWED_AUDIO_EXTS))
            )
        if len(data) < 12:
            raise ValueError("Audio file is too small or empty.")
        signatures = _MAGIC_SIGNATURES.get(ext, [])
        if not signatures:
            return
        ok = False
        for sig in signatures:
            if ext == ".m4a":
                if data[4:8] == sig:
                    ok = True
                    break
            elif data.startswith(sig):
                ok = True
                break
        # WAV also contains "WAVE" at offset 8.
        if ext == ".wav" and data[8:12] != b"WAVE":
            ok = False
        if not ok:
            raise ValueError(
                f"File does not appear to be a valid {ext} audio file."
            )

    @staticmethod
    def _probe_duration(path: Path) -> Optional[float]:
        try:
            import ffmpeg_utils  # local import to avoid hard dependency
            import subprocess
            import shutil as _sh
            ffprobe = _sh.which("ffprobe")
            if not ffprobe:
                return None
            out = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries",
                 "format=duration", "-of",
                 "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=20,
            )
            if out.returncode == 0 and out.stdout.strip():
                return round(float(out.stdout.strip()), 2)
        except Exception as exc:  # noqa: BLE001
            log.debug("Voice duration probe failed: %s", exc)
        return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list_voices(self) -> List[Dict]:
        voices: List[Voice] = []
        for vdir in sorted(self.voices_dir.iterdir()):
            if vdir.is_dir():
                voice = self._load(vdir.name)
                if voice:
                    voices.append(voice)
        voices.sort(key=lambda v: v.created_at, reverse=True)
        return [v.to_public_dict(self.output_dir) for v in voices]

    def get_voice(self, voice_id: str) -> Optional[Voice]:
        return self._load(voice_id)

    def create_voice(self, name: str, audio_bytes: bytes, filename: str,
                     notes: str = "",
                     avatar_id: Optional[str] = None) -> Voice:
        ext = Path(filename).suffix.lower()
        self._validate_audio(ext, audio_bytes)
        voice_id = "voice-" + secrets.token_hex(4)
        vdir = self._voice_dir(voice_id)
        vdir.mkdir(parents=True, exist_ok=True)
        sample_path = vdir / f"sample{ext}"
        sample_path.write_bytes(audio_bytes)
        voice = Voice(
            id=voice_id,
            name=name.strip() or "Untitled Voice",
            sample_path=str(sample_path),
            sample_filename=filename,
            avatar_id=avatar_id,
            notes=notes.strip(),
            created_at=datetime.now().isoformat(timespec="seconds"),
            duration_seconds=self._probe_duration(sample_path),
        )
        self._save(voice)
        return voice

    def update_voice(self, voice_id: str, name: Optional[str] = None,
                     notes: Optional[str] = None) -> Optional[Voice]:
        voice = self._load(voice_id)
        if not voice:
            return None
        if name is not None:
            voice.name = name.strip() or voice.name
        if notes is not None:
            voice.notes = notes.strip()
        self._save(voice)
        return voice

    def assign_to_avatar(self, voice_id: str,
                         avatar_id: Optional[str]) -> Optional[Voice]:
        voice = self._load(voice_id)
        if not voice:
            return None
        voice.avatar_id = avatar_id or None
        self._save(voice)
        return voice

    def delete_voice(self, voice_id: str) -> bool:
        vdir = self._voice_dir(voice_id)
        if not vdir.exists():
            return False
        try:
            shutil.rmtree(vdir)
        except OSError as exc:
            log.error("Failed to delete voice %s: %s", voice_id, exc)
            return False
        log.info("Deleted voice %s", voice_id)
        return True

    def assign_narration_to_clip(self, clip_id: str, voice_id: str) -> Dict:
        """Copy a voice sample into a clip's directory as its narration track.

        Updates the clip metadata with ``narration_path`` and ``voice_id`` via
        the project manager. Returns a dict describing the result.
        """
        if self.projects is None:
            raise RuntimeError("VoiceManager has no ProjectManager attached.")
        voice = self._load(voice_id)
        if not voice:
            raise ValueError(f"Voice '{voice_id}' not found.")
        clip = self.projects.get_clip(clip_id)
        if not clip:
            raise ValueError(f"Clip '{clip_id}' not found.")
        src = Path(voice.sample_path)
        if not src.exists():
            raise ValueError("Voice sample file is missing on disk.")
        ext = src.suffix.lower()
        clip.ensure_dirs()
        dest = clip.root_path / f"narration{ext}"
        shutil.copyfile(src, dest)
        self.projects.merge_clip_metadata(clip_id, {
            "narration_path": str(dest),
            "voice_id": voice_id,
        })
        log.info("Assigned voice %s narration to clip %s", voice_id, clip_id)
        return {
            "clip_id": clip_id,
            "voice_id": voice_id,
            "narration_path": str(dest),
        }
