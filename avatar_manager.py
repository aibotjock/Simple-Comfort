"""Avatar library management.

Stores up to :data:`MAX_AVATARS` reusable character avatars on disk under
``outputs/avatars/{avatar_id}/``. Each avatar has a full-resolution portrait
image plus a 256x256 thumbnail, an optional linked voice and free-form notes.

No database is used — an ``avatar.json`` file in each avatar directory is the
source of truth. The library always presents exactly ``MAX_AVATARS`` slots so
the UI can render placeholders for the empty ones.
"""
from __future__ import annotations

import json
import logging
import secrets
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:  # noqa: BLE001 - Pillow is optional at import time
    _PIL_AVAILABLE = False

log = logging.getLogger("simple_comfort.avatars")

MAX_AVATARS = 3
THUMBNAIL_SIZE = (256, 256)
# Portrait target aspect (width:height). Images wider than this are centre
# cropped so avatars line up nicely in the UI.
PORTRAIT_ASPECT = (4, 5)
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class Avatar:
    """A reusable character avatar."""

    id: str
    name: str = ""
    image_path: str = ""
    thumbnail_path: str = ""
    voice_id: Optional[str] = None
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_public_dict(self, output_root: Path) -> Dict:
        """Return a JSON-serialisable view with ``/outputs`` relative URLs."""
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
            "image_url": rel(self.image_path),
            "thumbnail_url": rel(self.thumbnail_path),
            "voice_id": self.voice_id,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "placeholder": False,
        }


class AvatarManager:
    """Create, read, update and delete avatars stored on disk."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.avatars_dir = self.output_dir / "avatars"
        self.avatars_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _avatar_dir(self, avatar_id: str) -> Path:
        return self.avatars_dir / avatar_id

    def _meta_path(self, avatar_id: str) -> Path:
        return self._avatar_dir(avatar_id) / "avatar.json"

    def _load(self, avatar_id: str) -> Optional[Avatar]:
        meta = self._meta_path(avatar_id)
        if not meta.exists():
            return None
        try:
            with meta.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load avatar %s: %s", avatar_id, exc)
            return None
        return Avatar(
            id=data.get("id", avatar_id),
            name=data.get("name", ""),
            image_path=data.get("image_path", ""),
            thumbnail_path=data.get("thumbnail_path", ""),
            voice_id=data.get("voice_id"),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def _save(self, avatar: Avatar) -> None:
        avatar.updated_at = datetime.now().isoformat(timespec="seconds")
        self._avatar_dir(avatar.id).mkdir(parents=True, exist_ok=True)
        with self._meta_path(avatar.id).open("w", encoding="utf-8") as fh:
            json.dump(asdict(avatar), fh, indent=2)
        log.info("Saved avatar %s", avatar.id)

    def _process_image(self, avatar_id: str, image_bytes: bytes,
                       filename: str) -> tuple:
        """Persist the full portrait + a square thumbnail.

        Returns ``(image_path, thumbnail_path)`` as strings. Falls back to
        storing the raw bytes if Pillow is unavailable.
        """
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTS:
            raise ValueError(
                f"Unsupported image type '{ext}'. Allowed: "
                + ", ".join(sorted(ALLOWED_IMAGE_EXTS))
            )
        adir = self._avatar_dir(avatar_id)
        adir.mkdir(parents=True, exist_ok=True)

        image_path = adir / f"portrait{ext}"
        thumb_path = adir / "thumbnail.png"

        if not _PIL_AVAILABLE:
            image_path.write_bytes(image_bytes)
            # Without Pillow we cannot resize; reuse the portrait as thumbnail.
            thumb_path = adir / f"thumbnail{ext}"
            thumb_path.write_bytes(image_bytes)
            return str(image_path), str(thumb_path)

        import io

        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            cropped = self._crop_to_aspect(img, PORTRAIT_ASPECT)
            cropped.save(image_path, quality=92)

            thumb = self._crop_to_aspect(img, (1, 1))
            thumb = thumb.resize(THUMBNAIL_SIZE, Image.LANCZOS)
            thumb.save(thumb_path)

        return str(image_path), str(thumb_path)

    @staticmethod
    def _crop_to_aspect(img, aspect: tuple):
        """Centre-crop a PIL image to the requested width:height aspect."""
        target_w, target_h = aspect
        w, h = img.size
        target_ratio = target_w / target_h
        current_ratio = w / h
        if abs(current_ratio - target_ratio) < 1e-3:
            return img
        if current_ratio > target_ratio:
            # Too wide -> crop width.
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            return img.crop((left, 0, left + new_w, h))
        # Too tall -> crop height.
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list_avatars(self) -> List[Dict]:
        """Return exactly ``MAX_AVATARS`` slots (filled + placeholders)."""
        existing = []
        for adir in sorted(self.avatars_dir.iterdir()):
            if adir.is_dir():
                avatar = self._load(adir.name)
                if avatar:
                    existing.append(avatar)
        existing.sort(key=lambda a: a.created_at)

        slots: List[Dict] = []
        for i in range(MAX_AVATARS):
            if i < len(existing):
                slots.append(existing[i].to_public_dict(self.output_dir))
            else:
                slots.append({
                    "id": None,
                    "name": "",
                    "image_url": None,
                    "thumbnail_url": None,
                    "voice_id": None,
                    "notes": "",
                    "created_at": "",
                    "updated_at": "",
                    "placeholder": True,
                    "slot_index": i,
                })
        return slots

    def count(self) -> int:
        return sum(
            1 for adir in self.avatars_dir.iterdir()
            if adir.is_dir() and self._meta_path(adir.name).exists()
        )

    def get_avatar(self, avatar_id: str) -> Optional[Avatar]:
        return self._load(avatar_id)

    def create_avatar(self, name: str, image_bytes: bytes, filename: str,
                      notes: str = "", voice_id: Optional[str] = None) -> Avatar:
        if self.count() >= MAX_AVATARS:
            raise ValueError(
                f"Avatar library is full (max {MAX_AVATARS}). "
                "Delete an avatar before adding another."
            )
        avatar_id = "avatar-" + secrets.token_hex(4)
        image_path, thumb_path = self._process_image(
            avatar_id, image_bytes, filename
        )
        now = datetime.now().isoformat(timespec="seconds")
        avatar = Avatar(
            id=avatar_id,
            name=name.strip() or "Untitled Avatar",
            image_path=image_path,
            thumbnail_path=thumb_path,
            voice_id=voice_id,
            notes=notes.strip(),
            created_at=now,
            updated_at=now,
        )
        self._save(avatar)
        return avatar

    def update_avatar(self, avatar_id: str, name: Optional[str] = None,
                      notes: Optional[str] = None,
                      voice_id: Optional[str] = None) -> Optional[Avatar]:
        avatar = self._load(avatar_id)
        if not avatar:
            return None
        if name is not None:
            avatar.name = name.strip() or avatar.name
        if notes is not None:
            avatar.notes = notes.strip()
        if voice_id is not None:
            avatar.voice_id = voice_id or None
        self._save(avatar)
        return avatar

    def replace_image(self, avatar_id: str, image_bytes: bytes,
                      filename: str) -> Optional[Avatar]:
        avatar = self._load(avatar_id)
        if not avatar:
            return None
        # Remove old image files (keep metadata + directory).
        for old in (avatar.image_path, avatar.thumbnail_path):
            try:
                if old and Path(old).exists():
                    Path(old).unlink()
            except OSError:
                pass
        image_path, thumb_path = self._process_image(
            avatar_id, image_bytes, filename
        )
        avatar.image_path = image_path
        avatar.thumbnail_path = thumb_path
        self._save(avatar)
        return avatar

    def assign_voice(self, avatar_id: str,
                     voice_id: Optional[str]) -> Optional[Avatar]:
        avatar = self._load(avatar_id)
        if not avatar:
            return None
        avatar.voice_id = voice_id or None
        self._save(avatar)
        return avatar

    def delete_avatar(self, avatar_id: str) -> bool:
        adir = self._avatar_dir(avatar_id)
        if not adir.exists():
            return False
        try:
            shutil.rmtree(adir)
        except OSError as exc:
            log.error("Failed to delete avatar %s: %s", avatar_id, exc)
            return False
        log.info("Deleted avatar %s", avatar_id)
        return True
