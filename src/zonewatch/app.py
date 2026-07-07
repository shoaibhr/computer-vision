"""Main application loop: capture -> detect -> track zones -> notify -> render."""

from __future__ import annotations

import logging
import signal
import time

from .config import Settings
from .detectors import create_detector
from .events import Cooldown, ZoneMonitor
from .notifiers import LogNotifier, NotifierHub, SnapshotNotifier, WebhookNotifier
from .stream import VideoStream

logger = logging.getLogger(__name__)

_GREEN = (0, 255, 0)
_BLUE = (255, 128, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)


def _build_hub(settings: Settings) -> NotifierHub:
    notifiers: list = [LogNotifier()]
    if settings.webhook_url:
        notifiers.append(WebhookNotifier(settings.webhook_url))
    if settings.snapshot_dir:
        notifiers.append(SnapshotNotifier(settings.snapshot_dir))
    return NotifierHub(notifiers, Cooldown(settings.cooldown))


def _annotate(cv2, frame, detections, monitor: ZoneMonitor, fps: float):
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)
        cv2.putText(frame, f"person {det.confidence:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN, 2)
    for zone in monitor.zones:
        occupied = monitor.is_occupied(zone)
        color = _RED if occupied else _BLUE
        cv2.rectangle(frame, (zone.x1, zone.y1), (zone.x2, zone.y2), color, 2)
        status = "OCCUPIED" if occupied else "clear"
        cv2.putText(frame, f"{zone.name}: {status}", (zone.x1, max(zone.y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"{fps:.1f} FPS", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 2)
    return frame


def run(settings: Settings) -> None:
    import cv2

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    detector = create_detector(settings.backend, settings.model, settings.confidence)
    monitor = ZoneMonitor(settings.zones, settings.enter_frames, settings.exit_frames)
    hub = _build_hub(settings)

    stop = {"requested": False}

    def _handle_signal(signum, _frame):
        logger.info("Received signal %d, shutting down", signum)
        stop["requested"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    frame_index = 0
    detections: list = []
    fps = 0.0
    logger.info(
        "Watching %d zone(s) on %s (backend=%s, headless=%s)",
        len(settings.zones), settings.source, settings.backend, settings.headless,
    )

    with VideoStream(settings.source) as stream:
        while not stop["requested"]:
            frame = stream.read()
            if frame is None:
                break

            started = time.perf_counter()
            if frame_index % settings.detect_every == 0:
                detections = detector.detect(frame)
                events = monitor.update([d.box for d in detections])
            else:
                events = []
            frame_index += 1

            elapsed = time.perf_counter() - started
            instant = 1.0 / elapsed if elapsed > 0 else 0.0
            fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant

            annotated = _annotate(cv2, frame, detections, monitor, fps)
            for event in events:
                hub.dispatch(event, annotated)

            if not settings.headless:
                cv2.imshow("ZoneWatch", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit requested")
                    break

    if not settings.headless:
        cv2.destroyAllWindows()
    logger.info("Stopped after %d frames", frame_index)
