"""FFmpeg finish tests: audio-preserving assembly + exact-resolution landing.

Runs real ffmpeg/ffprobe on tiny synthetic clips (no network, no GPU).
These pin the two behaviours the LTX-2.5 joint A/V backend depends on:
concatenated sections must keep their audio, and a requested 720p/1080p
must land at exactly those dimensions without aspect distortion.

Async helpers are driven through asyncio.run() — this suite deliberately
avoids any pytest-asyncio dependency, matching the rest of the repo.
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from ffmpeg_utils import concat_videos, probe_video, scale_video

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not on PATH",
)


def run(coro):
    return asyncio.run(coro)


def make_clip(path: Path, seconds: float = 0.4, audio: bool = True,
              size: str = "640x352") -> Path:
    """Render a tiny synthetic mp4 (lavfi test source, optional sine tone)."""
    args = ["ffmpeg", "-y",
            "-f", "lavfi", "-i",
            f"testsrc=duration={seconds}:size={size}:rate=24"]
    if audio:
        args += ["-f", "lavfi", "-i",
                 f"sine=frequency=440:duration={seconds}"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    else:
        args += ["-an"]
    args += [str(path)]
    async def _render() -> int:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL)
        return await proc.wait()

    assert asyncio.run(_render()) == 0
    return path


def test_probe_reports_audio_presence(tmp_path):
    with_audio = make_clip(tmp_path / "a.mp4")
    silent = make_clip(tmp_path / "b.mp4", audio=False)
    assert run(probe_video(with_audio))["has_audio"] is True
    assert run(probe_video(silent))["has_audio"] is False


def test_concat_preserves_audio_when_all_sections_have_it(tmp_path):
    parts = [make_clip(tmp_path / f"p{i}.mp4") for i in range(3)]
    out = run(concat_videos(parts, tmp_path / "joined.mp4", fps=24))
    info = run(probe_video(out))
    assert info["has_audio"] is True
    assert info["duration"] == pytest.approx(1.2, abs=0.15)


def test_concat_drops_audio_when_any_section_lacks_it(tmp_path):
    parts = [make_clip(tmp_path / "p0.mp4"),
             make_clip(tmp_path / "p1.mp4", audio=False)]
    out = run(concat_videos(parts, tmp_path / "joined.mp4", fps=24))
    assert run(probe_video(out))["has_audio"] is False


def test_scale_lands_exact_target_dimensions(tmp_path):
    # A 640x352 render (the 720p working grid) must finish at exactly
    # 1280x720 — cover-scaled then centre-cropped, never squeezed.
    src = make_clip(tmp_path / "in.mp4", audio=True, size="640x352")
    out = run(scale_video(src, tmp_path / "out.mp4", 1280, 720))
    info = run(probe_video(out))
    assert (info["width"], info["height"]) == (1280, 720)
    assert info["has_audio"] is True  # audio survives the finish


def test_scale_handles_silent_input(tmp_path):
    src = make_clip(tmp_path / "in.mp4", audio=False, size="704x1280")
    out = run(scale_video(src, tmp_path / "out.mp4", 720, 1280))
    info = run(probe_video(out))
    assert (info["width"], info["height"]) == (720, 1280)
    assert info["has_audio"] is False


def test_scale_noop_dims_still_work(tmp_path):
    src = make_clip(tmp_path / "in.mp4", size="544x960")
    out = run(scale_video(src, tmp_path / "out.mp4", 544, 960))
    info = run(probe_video(out))
    assert (info["width"], info["height"]) == (544, 960)
