"""Tests for GenerationManager._cleanup_empty_clip.

A failed/cancelled job must not leave a ghost clip directory behind — unless
the directory holds any real content, in which case it is never removed.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from comfy_client import WorkerPool
from generation_manager import GenerationJob, GenerationManager
from project_manager import ProjectManager


@pytest.fixture
def manager(temp_output_dir: Path) -> GenerationManager:
    return GenerationManager(
        pool=WorkerPool([]),          # never used by cleanup logic
        adapters={},
        projects=ProjectManager(temp_output_dir),
    )


@pytest.fixture
def clip(temp_output_dir: Path):
    pm = ProjectManager(temp_output_dir)
    return pm.create_clip(pm.create_project())


def make_job(status: str) -> GenerationJob:
    job = GenerationJob(job_id="job-test", model="ltx25", mode="t2v")
    job.status = status
    return job


def test_failed_job_with_zero_sections_removes_empty_dir(manager, clip):
    # create_clip() left only an empty sections/ subdir.
    assert clip.sections_dir.is_dir()
    assert not any(p.is_file() for p in clip.root_path.rglob("*"))
    manager._cleanup_empty_clip(make_job("failed"), clip)
    assert not Path(clip.root).exists()


def test_failed_job_with_produced_section_keeps_dir(manager, clip):
    section = clip.sections_dir / "section_00.mp4"
    section.write_bytes(b"partial render")
    manager._cleanup_empty_clip(make_job("failed"), clip)
    assert Path(clip.root).is_dir()
    assert section.exists()


@pytest.mark.parametrize("sentinel", ["final.mp4", "metadata.json"])
def test_dirs_with_final_video_or_metadata_never_removed(manager, clip,
                                                          sentinel):
    (clip.root_path / sentinel).write_bytes(b"keep me")
    manager._cleanup_empty_clip(make_job("failed"), clip)
    assert (clip.root_path / sentinel).exists()


def test_completed_jobs_are_never_cleaned_up(manager, clip):
    manager._cleanup_empty_clip(make_job("completed"), clip)
    assert Path(clip.root).is_dir()


def test_missing_directory_is_ignored(manager, clip):
    shutil.rmtree(clip.root)
    manager._cleanup_empty_clip(make_job("failed"), clip)  # must not raise
    assert not Path(clip.root).exists()
