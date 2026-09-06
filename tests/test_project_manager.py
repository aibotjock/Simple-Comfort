"""Project / clip persistence tests against a temporary OUTPUT_DIR."""
from __future__ import annotations

import re
from pathlib import Path

from project_manager import ProjectManager

PROJECT_ID_RE = re.compile(r"^proj-\d{8}-[0-9a-f]{6}$")


def make_manager(temp_output_dir: Path) -> ProjectManager:
    return ProjectManager(temp_output_dir)


def make_clip_with_metadata(pm: ProjectManager, project_id: str):
    clip = pm.create_clip(project_id)
    pm.save_clip_metadata(clip, {"model": "LTX 2.5", "prompt": "a kitchen"})
    return clip


# ---------------------------------------------------------------------- #
# Creation
# ---------------------------------------------------------------------- #
def test_create_project_makes_directory(temp_output_dir):
    pm = make_manager(temp_output_dir)
    project_id = pm.create_project()
    assert PROJECT_ID_RE.match(project_id)
    assert (temp_output_dir / "projects" / project_id).is_dir()


def test_create_clip_numbers_sequentially_and_lays_out_dirs(temp_output_dir):
    pm = make_manager(temp_output_dir)
    project_id = pm.create_project()
    first = pm.create_clip(project_id)
    second = pm.create_clip(project_id)

    assert first.clip_number == 1
    assert second.clip_number == 2
    assert first.clip_id == f"{project_id}__clip-001"
    assert second.clip_id == f"{project_id}__clip-002"
    for clip in (first, second):
        assert clip.root_path.is_dir()
        assert clip.sections_dir.is_dir()
        assert clip.final_video == clip.root_path / "final.mp4"
        assert clip.metadata_path == clip.root_path / "metadata.json"


# ---------------------------------------------------------------------- #
# Lookup
# ---------------------------------------------------------------------- #
def test_list_get_and_last_clip_roundtrip(temp_output_dir):
    pm = make_manager(temp_output_dir)
    project_id = pm.create_project()
    clip = make_clip_with_metadata(pm, project_id)

    listed = pm.list_clips()
    assert [c.clip_id for c in listed] == [clip.clip_id]

    fetched = pm.get_clip(clip.clip_id)
    assert fetched is not None
    assert fetched.clip_id == clip.clip_id
    assert fetched.metadata["model"] == "LTX 2.5"

    assert pm.get_clip("does-not-exist") is None

    last = pm.get_last_clip()
    assert last is not None and last.clip_id == clip.clip_id


def test_list_clips_sorts_newest_first(temp_output_dir):
    pm = make_manager(temp_output_dir)
    project_id = pm.create_project()
    older = make_clip_with_metadata(pm, project_id)
    newer = make_clip_with_metadata(pm, project_id)
    # created_at has second resolution; nudge to guarantee ordering.
    newer.created_at = "2999-01-01T00:00:00"
    older.created_at = "2000-01-01T00:00:00"

    assert [c.clip_id for c in pm.list_clips()] == \
        [newer.clip_id, older.clip_id]
    assert pm.get_last_clip().clip_id == newer.clip_id


def test_last_clip_none_when_no_clips(temp_output_dir):
    pm = make_manager(temp_output_dir)
    pm.create_project()
    assert pm.get_last_clip() is None
    assert pm.list_clips() == []


# ---------------------------------------------------------------------- #
# Public serialisation
# ---------------------------------------------------------------------- #
def test_public_dict_url_shape(temp_output_dir):
    pm = make_manager(temp_output_dir)
    project_id = pm.create_project()
    clip = make_clip_with_metadata(pm, project_id)

    # Nothing rendered yet: every media URL is null.
    public = clip.to_public_dict(temp_output_dir)
    assert public["clip_id"] == clip.clip_id
    assert public["video_url"] is None
    assert public["thumbnail_url"] is None
    assert public["narration_url"] is None

    # Once files exist they are exposed as /outputs/<relative-path> URLs.
    clip.final_video.write_bytes(b"fake-mp4")
    clip.thumbnail.write_bytes(b"fake-jpg")
    public = clip.to_public_dict(temp_output_dir)
    expected_prefix = "/outputs/projects/" + project_id + "/clip-001/"
    assert public["video_url"] == expected_prefix + "final.mp4"
    assert public["thumbnail_url"] == expected_prefix + "thumbnail.jpg"
    # A file outside OUTPUT_DIR never leaks a URL.
    assert public["first_frame_url"] is None
