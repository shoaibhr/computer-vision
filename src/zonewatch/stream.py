"""Video stream wrapper with automatic reconnection.

RTSP feeds drop all the time (network blips, camera reboots). Instead of
crashing, VideoStream reconnects with exponential backoff.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class StreamUnavailable(RuntimeError):
    """Raised when the source cannot be (re)opened after all retries."""


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
        self.source = source
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self._cap: cv2.VideoCapture | None = None
        self._open()

    def _open(self) -> None:
        backoff = self.initial_backoff
        for attempt in range(1, self.max_retries + 1):
            self.release()
            cap = self._cv2.VideoCapture(self.source)
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
