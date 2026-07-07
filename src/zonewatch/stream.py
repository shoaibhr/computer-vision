"""Video source handling: any camera or stream, with automatic reconnection.

Supported sources:
- camera index (``0``, ``1``, ...) — USB webcams, built-in laptop cameras
- Linux device paths (``/dev/video0``) — normalized to an index by the config layer
- ``"auto"`` — probe indexes and use the first camera that delivers frames
- RTSP / HTTP(S) / MJPEG URLs — network cameras
- GStreamer pipeline strings
- video file paths

RTSP feeds drop all the time (network blips, camera reboots). Instead of
crashing, VideoStream reconnects with exponential backoff.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

AUTO_SOURCE = "auto"


class StreamUnavailable(RuntimeError):
    """Raised when the source cannot be (re)opened after all retries."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    width: int
    height: int
    fps: float


def _quiet_opencv() -> None:
    """Silence OpenCV's stderr noise while probing indexes that don't exist."""
    try:
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


def _open_capture(cv2, source):
    """Open a cv2.VideoCapture with the best API for the source and platform."""
    if isinstance(source, int) and sys.platform == "win32":
        # MSMF is notoriously slow to open USB cameras on Windows
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://")):
        # Prefer TCP transport: avoids smeared/corrupted frames on flaky WiFi
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    return cv2.VideoCapture(source)


def probe_cameras(max_index: int = 10) -> list[CameraInfo]:
    """Scan camera indexes 0..max_index-1 and return the ones that deliver frames."""
    import cv2

    _quiet_opencv()
    cameras = []
    for index in range(max_index):
        cap = _open_capture(cv2, index)
        try:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cameras.append(
                        CameraInfo(
                            index=index,
                            width=frame.shape[1],
                            height=frame.shape[0],
                            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0),
                        )
                    )
        finally:
            cap.release()
    return cameras


def resolve_auto_source() -> int:
    """Return the index of the first working camera, or raise StreamUnavailable."""
    cameras = probe_cameras()
    if not cameras:
        raise StreamUnavailable(
            "No working camera found (probed indexes 0-9). Plug in a camera or "
            "pass an explicit --source."
        )
    cam = cameras[0]
    logger.info("Auto-selected camera %d (%dx%d)", cam.index, cam.width, cam.height)
    return cam.index


class VideoStream:
    def __init__(
        self,
        source: str | int,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
        max_backoff: float = 60.0,
    ):
        import cv2

        self._cv2 = cv2
        self.source = resolve_auto_source() if source == AUTO_SOURCE else source
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self._cap: cv2.VideoCapture | None = None
        self._open()

    def _open(self) -> None:
        backoff = self.initial_backoff
        for attempt in range(1, self.max_retries + 1):
            self.release()
            cap = _open_capture(self._cv2, self.source)
            if cap.isOpened():
                self._cap = cap
                logger.info("Stream opened: %s", self.source)
                return
            cap.release()
            logger.warning(
                "Could not open stream (attempt %d/%d); retrying in %.0fs",
                attempt, self.max_retries, backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff)
        raise StreamUnavailable(f"Failed to open video source after {self.max_retries} attempts: "
                                f"{self.source}")

    def read(self):
        """Return the next frame, transparently reconnecting on failure.

        Returns None only when the source is a finite file that has ended.
        """
        assert self._cap is not None
        ret, frame = self._cap.read()
        if ret:
            return frame
        if isinstance(self.source, str) and not self.source.lower().startswith("rtsp"):
            # Local video files end; that's not an error.
            frame_count = self._cap.get(self._cv2.CAP_PROP_FRAME_COUNT)
            if frame_count > 0:
                logger.info("End of video file reached")
                return None
        logger.warning("Frame read failed; reconnecting to %s", self.source)
        self._open()
        ret, frame = self._cap.read()
        if not ret:
            raise StreamUnavailable(f"Stream reconnected but still yields no frames: "
                                    f"{self.source}")
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
