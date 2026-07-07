import pytest

from zonewatch.config import parse_source, save_zones_to_env
from zonewatch.zones import Zone, parse_zones, zones_to_spec


class TestParseSource:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0", 0),
            ("3", 3),
            (" 1 ", 1),
            ("/dev/video0", 0),
            ("/dev/video12", 12),
            ("auto", "auto"),
            ("AUTO", "auto"),
            ("rtsp://user:pass@host:554/stream", "rtsp://user:pass@host:554/stream"),
            ("http://cam.local/mjpeg", "http://cam.local/mjpeg"),
            ("clip.mp4", "clip.mp4"),
            # GStreamer pipelines pass through untouched
            ("v4l2src ! videoconvert ! appsink", "v4l2src ! videoconvert ! appsink"),
        ],
    )
    def test_parsing(self, raw, expected):
        assert parse_source(raw) == expected

    def test_non_video_dev_path_untouched(self):
        assert parse_source("/dev/ttyUSB0") == "/dev/ttyUSB0"


class TestZonesSpecRoundtrip:
    def test_roundtrip(self):
        zones = [Zone("entrance", 118, 6, 218, 206), Zone("dock", 300, 50, 500, 400)]
        assert parse_zones(zones_to_spec(zones)) == zones


class TestSaveZonesToEnv:
    ZONES = [Zone("door", 1, 2, 30, 40)]

    def test_creates_file(self, tmp_path):
        env = tmp_path / ".env"
        spec = save_zones_to_env(self.ZONES, env)
        assert spec == "door:1,2,30,40"
        assert env.read_text() == "ZONES=door:1,2,30,40\n"

    def test_replaces_existing_zones_line_preserving_rest(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("SOURCE=rtsp://cam\nZONES=old:0,0,1,1\nCOOLDOWN=60\n")
        save_zones_to_env(self.ZONES, env)
        assert env.read_text() == "SOURCE=rtsp://cam\nZONES=door:1,2,30,40\nCOOLDOWN=60\n"

    def test_appends_when_no_zones_line(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("SOURCE=0\n")
        save_zones_to_env(self.ZONES, env)
        assert env.read_text() == "SOURCE=0\nZONES=door:1,2,30,40\n"
