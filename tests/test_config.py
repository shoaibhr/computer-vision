from pathlib import Path

import pytest

from zonewatch.config import load_settings
from zonewatch.zones import Zone

BASE_ENV = {"SOURCE": "rtsp://cam/stream", "ZONES": "door:1,1,50,50"}


def test_minimal_env():
    settings = load_settings(env=BASE_ENV)
    assert settings.source == "rtsp://cam/stream"
    assert settings.zones == [Zone("door", 1, 1, 50, 50)]
    assert settings.backend == "opencv"
    assert settings.confidence == 0.5
    assert settings.headless is False


def test_webcam_index_becomes_int():
    settings = load_settings(env={**BASE_ENV, "SOURCE": "0"})
    assert settings.source == 0


def test_legacy_rtsp_url_and_roi_vars():
    env = {
        "RTSP_URL": "rtsp://legacy/stream",
        "ROI_X1": "118", "ROI_Y1": "6", "ROI_X2": "218", "ROI_Y2": "206",
    }
    settings = load_settings(env=env)
    assert settings.source == "rtsp://legacy/stream"
    assert settings.zones == [Zone("roi", 118, 6, 218, 206)]


def test_all_options_from_env():
    env = {
        **BASE_ENV,
        "BACKEND": "ultralytics",
        "MODEL": "yolov8s.pt",
        "CONFIDENCE": "0.7",
        "DETECT_EVERY": "3",
        "ENTER_FRAMES": "2",
        "EXIT_FRAMES": "10",
        "WEBHOOK_URL": "https://example.com/hook",
        "SNAPSHOT_DIR": "snaps",
        "COOLDOWN": "60",
        "HEADLESS": "true",
        "LOG_LEVEL": "debug",
    }
    settings = load_settings(env=env)
    assert settings.backend == "ultralytics"
    assert settings.model == "yolov8s.pt"
    assert settings.confidence == 0.7
    assert settings.detect_every == 3
    assert settings.enter_frames == 2
    assert settings.exit_frames == 10
    assert settings.webhook_url == "https://example.com/hook"
    assert settings.snapshot_dir == Path("snaps")
    assert settings.cooldown == 60.0
    assert settings.headless is True
    assert settings.log_level == "DEBUG"


def test_overrides_beat_env():
    settings = load_settings(env=BASE_ENV, confidence=0.9, headless=True)
    assert settings.confidence == 0.9
    assert settings.headless is True


def test_none_overrides_ignored():
    settings = load_settings(env=BASE_ENV, confidence=None)
    assert settings.confidence == 0.5


def test_missing_source_rejected():
    with pytest.raises(ValueError, match="video source"):
        load_settings(env={"ZONES": "a:1,1,2,2"})


def test_missing_zones_rejected():
    with pytest.raises(ValueError, match="zone"):
        load_settings(env={"SOURCE": "rtsp://cam"})


def test_bad_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        load_settings(env={**BASE_ENV, "BACKEND": "tensorflow"})


def test_bad_confidence_rejected():
    with pytest.raises(ValueError, match="confidence"):
        load_settings(env={**BASE_ENV, "CONFIDENCE": "1.5"})


def test_duplicate_zone_names_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        load_settings(env={**BASE_ENV, "ZONES": "a:1,1,2,2;a:3,3,4,4"})
