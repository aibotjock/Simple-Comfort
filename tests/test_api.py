"""In-process API tests via fastapi.testclient.TestClient.

No live app process and no ComfyUI: tests/conftest.py pins ``OUTPUT_DIR`` to a
temp dir and ``COMFY_ENDPOINTS`` to an unreachable endpoint BEFORE ``app`` is
imported (config reads the environment at import time), so the API is
exercised exactly as it behaves with ComfyUI down.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from config import settings
from project_manager import ProjectManager

import app as app_module


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as test_client:
        yield test_client


# ---------------------------------------------------------------------- #
# Status
# ---------------------------------------------------------------------- #
def test_status_reports_offline_shape(client):
    # Hermeticity guard: the app under test is pointed at the endpoint
    # pinned in tests/conftest.py (unreachable by construction), never at a
    # live ComfyUI — so "connected" below is deterministic.
    assert settings.endpoints == ["http://127.0.0.1:9"]

    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    # Bulletproof shape: connected / workers / multi_gpu / ffmpeg always set.
    assert set(body) >= {"connected", "workers", "multi_gpu", "ffmpeg"}
    assert isinstance(body["connected"], bool)
    assert isinstance(body["multi_gpu"], bool)
    assert isinstance(body["ffmpeg"], bool)
    # Hermetic endpoint is unreachable by construction.
    assert body["connected"] is False
    assert body["multi_gpu"] is False
    assert isinstance(body["workers"], list) and body["workers"]
    for worker in body["workers"]:
        assert set(worker) >= {"index", "label", "endpoint", "alive", "busy",
                               "jobs_completed"}
        assert worker["alive"] is False


# ---------------------------------------------------------------------- #
# Clips
# ---------------------------------------------------------------------- #
def test_clips_listing_roundtrip(client):
    # Fresh temp OUTPUT_DIR -> empty listing.
    resp = client.get("/api/clips")
    assert resp.status_code == 200
    assert resp.json() == {"clips": []}

    # A clip persisted through the app's own ProjectManager shows up.
    projects = ProjectManager(settings.OUTPUT_DIR)
    clip = projects.create_clip(projects.create_project())
    projects.save_clip_metadata(clip, {"model": "LTX 2.5"})

    resp = client.get("/api/clips")
    assert resp.status_code == 200
    clips = resp.json()["clips"]
    assert [c["clip_id"] for c in clips] == [clip.clip_id]
    assert clips[0]["video_url"] is None  # nothing rendered yet


def test_unknown_clip_returns_404(client):
    resp = client.get("/api/clip/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------- #
# Generation
# ---------------------------------------------------------------------- #
def test_generate_with_unknown_model_returns_400(client):
    resp = client.post("/api/generate", data={
        "prompt": "a slow pan across a quiet kitchen",
        "model": "definitely-not-a-model",
        "mode": "t2v",
    })
    assert resp.status_code == 400
    assert "Unknown model" in resp.json()["detail"]


def test_generate_rejects_invalid_mode(client):
    resp = client.post("/api/generate", data={
        "prompt": "a slow pan",
        "model": "ltx25",
        "mode": "sideways",
    })
    assert resp.status_code == 400
    assert "Invalid mode" in resp.json()["detail"]


def test_generate_with_comfyui_down_reports_not_ready(client):
    """With no reachable ComfyUI the capability guard must answer 409, never
    submit a doomed job."""
    resp = client.post("/api/generate", data={
        "prompt": "a slow pan across a quiet kitchen",
        "model": "ltx25",
        "mode": "t2v",
    })
    assert resp.status_code == 409
    assert "not ready" in resp.json()["detail"]


# ---------------------------------------------------------------------- #
# Custom workflows
# ---------------------------------------------------------------------- #
def test_custom_workflows_list_envelope(client):
    resp = client.get("/api/custom-workflows")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"workflows"}
    assert isinstance(body["workflows"], list)
    for record in body["workflows"]:
        assert record["slug"] and record["name"]
        assert "validated" in record


def test_generate_duration_param_accepted_and_validated(client):
    """duration parses to target_seconds; bad values 400 before ComfyUI is
    consulted (workers are unreachable by construction in this suite)."""
    # Valid duration passes request validation and reaches the worker
    # stage, which is offline here -> 409, not 400.
    r = client.post("/api/generate", data={
        "prompt": "a quiet harbor at dawn", "model": "ltx25", "mode": "t2v",
        "duration": "20",
    })
    assert r.status_code == 409, r.text
    # Out-of-range duration is rejected outright.
    r = client.post("/api/generate", data={
        "prompt": "x", "model": "ltx25", "mode": "t2v", "duration": "3",
    })
    assert r.status_code == 400
