"""Event sinks: log, webhook, and annotated snapshots."""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime
from pathlib import Path

from .events import Cooldown, ZoneEvent

logger = logging.getLogger(__name__)


class LogNotifier:
    def notify(self, event: ZoneEvent, frame=None) -> None:
        verb = "entered" if event.kind == "enter" else "left"
        logger.info("Person %s zone '%s'", verb, event.zone.name)


class WebhookNotifier:
    """POSTs the event as JSON to a configured URL."""

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout

    def notify(self, event: ZoneEvent, frame=None) -> None:
        body = json.dumps(event.to_payload()).encode()
        request = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:  # noqa: S310
                logger.debug("Webhook delivered (%s): HTTP %s", event.kind, resp.status)
        except Exception as exc:
            logger.error("Webhook delivery failed: %s", exc)


class SnapshotNotifier:
    """Saves the annotated frame to disk on zone entry."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def notify(self, event: ZoneEvent, frame=None) -> None:
        if event.kind != "enter" or frame is None:
            return
        import cv2

        stamp = datetime.fromtimestamp(event.timestamp).strftime("%Y%m%d-%H%M%S")
        path = self.directory / f"{stamp}-{event.zone.name}.jpg"
        if cv2.imwrite(str(path), frame):
            logger.info("Snapshot saved: %s", path)
        else:
            logger.error("Failed to write snapshot: %s", path)


class NotifierHub:
    """Fans events out to all notifiers, applying a per-zone cooldown.

    Exit events bypass the cooldown so enter/exit pairs stay balanced.
    """

    def __init__(self, notifiers: list, cooldown: Cooldown | None = None):
        self.notifiers = notifiers
        self.cooldown = cooldown

    def dispatch(self, event: ZoneEvent, frame=None) -> bool:
        if (
            event.kind == "enter"
            and self.cooldown is not None
            and not self.cooldown.ready(event.zone.name)
        ):
            logger.debug("Suppressed '%s' enter event (cooldown)", event.zone.name)
            return False
        for notifier in self.notifiers:
            try:
                notifier.notify(event, frame)
            except Exception:
                logger.exception("Notifier %s failed", type(notifier).__name__)
        return True
