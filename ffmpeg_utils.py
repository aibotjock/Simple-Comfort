"""FFmpeg / FFprobe helpers.

Every function shells out to ``ffmpeg`` / ``ffprobe`` and logs the exact
command at DEBUG level. Functions raise :class:`FFmpegError` on failure with a
meaningful message.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("simple_comfort.ffmpeg")


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe invocation fails."""


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(
            f"'{name}' was not found on PATH. Please install FFmpeg "
            f"(https://ffmpeg.org/) to enable video processing."
        )
    return path


async def _run(cmd: List[str]) -> str:
    """Run a subprocess command asynchronously, returning combined stdout."""
    log.debug("ffmpeg cmd: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode("utf-8", "replace").strip()
        raise FFmpegError(
            f"Command failed ({proc.returncode}): {' '.join(cmd[:2])} ...\n{err[-800:]}"
        )
    return stdout.decode("utf-8", "replace")


async def probe_video(video_path: Path) -> Dict:
    """Return duration, fps, width, height, codec for a video file."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FFmpegError(f"Video not found: {video_path}")

    cmd = [
        _binary("ffprobe"),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    raw = await _run(cmd)
    data = json.loads(raw)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise FFmpegError(f"No video stream found in {video_path}")

    # fps is expressed as a fraction like "24/1".
    fps = 0.0
    rate = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate")
    if rate and "/" in rate:
        num, den = rate.split("/")
        den_f = float(den)
        fps = float(num) / den_f if den_f else 0.0

    duration = 0.0
    try:
        duration = float(data.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "duration": duration,
        "fps": round(fps, 3),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name", "unknown"),
        "nb_frames": int(video_stream.get("nb_frames", 0) or 0),
    }


async def concat_videos(paths: List[Path], output_path: Path,
                        fps: Optional[int] = None) -> Path:
    """Concatenate multiple video files into a single output.

    Uses the ffmpeg concat demuxer. Re-encodes to guarantee a constant frame
    rate (required for downstream lip-sync tooling) and uniform codec.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise FFmpegError("concat_videos called with no input paths")
    for p in paths:
        if not p.exists():
            raise FFmpegError(f"Section file missing for concat: {p}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(paths) == 1 and fps is None:
        # Fast path: nothing to concatenate.
        shutil.copyfile(paths[0], output_path)
        return output_path

    list_file = output_path.parent / f".concat_{output_path.stem}.txt"
    with list_file.open("w", encoding="utf-8") as fh:
        for p in paths:
            # Escape single quotes for the concat demuxer syntax.
            safe = str(p.resolve()).replace("'", "'\\''")
            fh.write(f"file '{safe}'\n")

    cmd = [
        _binary("ffmpeg"), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
    ]
    # Re-encode for a clean, constant-frame-rate output.
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if fps:
        cmd += ["-r", str(fps), "-vsync", "cfr"]
    cmd += ["-an", str(output_path)]

    try:
        await _run(cmd)
    finally:
        list_file.unlink(missing_ok=True)

    if not output_path.exists():
        raise FFmpegError("Concat completed but output file is missing")
    return output_path


async def still_to_video(image_path: Path, output_path: Path,
                         fps: int = 24, duration: float = 2.0) -> Path:
    """Promote a still image to a short playable mp4 (1/fps * duration frames).

    Used when a custom workflow's only output is an image — the clip library
    and browser player expect a video file.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    if not image_path.exists():
        raise FFmpegError(f"Still image missing: {image_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(duration * fps))
    cmd = [
        _binary("ffmpeg"), "-y",
        "-loop", "1", "-i", str(image_path),
        "-frames:v", str(frames), "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        str(output_path),
    ]
    await _run(cmd)
    if not output_path.exists():
        raise FFmpegError("Still-to-video conversion produced no output")
    return output_path


async def extract_frame(video_path: Path, frame_number: int,
                        output_path: Path) -> Path:
    """Extract a specific 0-based frame index to a PNG."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _binary("ffmpeg"), "-y",
        "-i", str(video_path),
        "-vf", f"select=eq(n\\,{max(frame_number, 0)})",
        "-vframes", "1",
        str(output_path),
    ]
    await _run(cmd)
    if not output_path.exists():
        raise FFmpegError(f"Failed to extract frame {frame_number}")
    return output_path


async def extract_last_frame(video_path: Path, output_path: Path) -> Path:
    """Extract the final frame of a video to a PNG."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Seek near the end and grab the last decoded frame.
    cmd = [
        _binary("ffmpeg"), "-y",
        "-sseof", "-1",
        "-i", str(video_path),
        "-update", "1",
        "-vf", "select='eq(n\\,0)+1'",
        "-q:v", "2",
        str(output_path),
    ]
    try:
        await _run(cmd)
    except FFmpegError:
        # Fallback: probe frame count and extract by index.
        info = await probe_video(video_path)
        total = info.get("nb_frames", 0)
        if total <= 0:
            fps = info.get("fps", 24) or 24
            total = int(info.get("duration", 0) * fps)
        return await extract_frame(video_path, max(total - 1, 0), output_path)

    if not output_path.exists():
        raise FFmpegError("Failed to extract last frame")
    return output_path


async def create_thumbnail(video_path: Path, output_path: Path,
                           width: int = 320) -> Path:
    """Create a JPEG thumbnail from the first frame of the video."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _binary("ffmpeg"), "-y",
        "-i", str(video_path),
        "-vf", f"thumbnail,scale={width}:-1",
        "-frames:v", "1",
        str(output_path),
    ]
    await _run(cmd)
    if not output_path.exists():
        raise FFmpegError("Failed to create thumbnail")
    return output_path


async def interpolate_fps(input_path: Path, output_path: Path,
                          target_fps: int) -> Path:
    """Motion-interpolate a video up to ``target_fps`` using minterpolate.

    Uses ffmpeg's ``minterpolate`` filter (motion-compensated frame
    interpolation) to synthesise intermediate frames for smoother motion.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FFmpegError(f"Video not found for interpolation: {input_path}")
    if target_fps <= 0:
        raise FFmpegError("target_fps must be a positive integer")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vf = (
        f"minterpolate=fps={int(target_fps)}:mi_mode=mci:"
        "mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    )
    cmd = [
        _binary("ffmpeg"), "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an", str(output_path),
    ]
    await _run(cmd)
    if not output_path.exists():
        raise FFmpegError("Frame interpolation completed but output is missing")
    return output_path


async def denoise_video(input_path: Path, output_path: Path,
                        strength: str = "2:1:2:3") -> Path:
    """Apply high-quality temporal/spatial denoising via ``hqdn3d``.

    ``strength`` is the hqdn3d parameter string
    (luma_spatial:chroma_spatial:luma_tmp:chroma_tmp).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FFmpegError(f"Video not found for denoise: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _binary("ffmpeg"), "-y",
        "-i", str(input_path),
        "-vf", f"hqdn3d={strength}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an", str(output_path),
    ]
    await _run(cmd)
    if not output_path.exists():
        raise FFmpegError("Denoise completed but output is missing")
    return output_path


def ffmpeg_available() -> bool:
    """Return True if both ffmpeg and ffprobe are on PATH."""
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
