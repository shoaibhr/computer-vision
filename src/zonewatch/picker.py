"""Interactive zone picker: draw detection zones on the live video with the mouse.

No more guessing pixel coordinates — click and drag rectangles directly on
the camera feed. The drafting logic (ZoneDrafter) is pure Python so it can
be unit-tested; only ``pick_zones`` touches OpenCV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .stream import StreamUnavailable, VideoStream
from .zones import Zone

logger = logging.getLogger(__name__)

MIN_ZONE_SIZE = 8  # pixels; smaller drags are treated as accidental clicks

HELP_LINES = [
    "drag: draw zone   u: undo   r: reset   space: pause",
    "ENTER/s: save & start   q/ESC: cancel",
]


@dataclass
class ZoneDrafter:
    """Turns mouse press/drag/release into a list of named zones."""

    width: int
    height: int
    zones: list[Zone] = field(default_factory=list)
    _anchor: tuple | None = field(default=None, init=False)
    _live: tuple | None = field(default=None, init=False)

    def _clamp(self, x: int, y: int) -> tuple:
        return max(0, min(x, self.width - 1)), max(0, min(y, self.height - 1))

    def _normalize(self, a: tuple, b: tuple) -> tuple:
        (ax, ay), (bx, by) = a, b
        return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)

    def press(self, x: int, y: int) -> None:
        self._anchor = self._clamp(x, y)
        self._live = None

    def drag(self, x: int, y: int) -> None:
        if self._anchor is not None:
            self._live = self._normalize(self._anchor, self._clamp(x, y))

    def release(self, x: int, y: int) -> Zone | None:
        """Finish the drag; returns the new zone, or None for tiny/no drags."""
        if self._anchor is None:
            return None
        box = self._normalize(self._anchor, self._clamp(x, y))
        self._anchor = None
        self._live = None
        x1, y1, x2, y2 = box
        if x2 - x1 < MIN_ZONE_SIZE or y2 - y1 < MIN_ZONE_SIZE:
            return None
        zone = Zone(self._next_name(), x1, y1, x2, y2)
        self.zones.append(zone)
        return zone

    @property
    def live_box(self) -> tuple | None:
        """The in-progress rectangle while dragging, if any."""
        return self._live

    def undo(self) -> None:
        if self.zones:
            self.zones.pop()

    def reset(self) -> None:
        self.zones.clear()
        self._anchor = None
        self._live = None

    def _next_name(self) -> str:
        taken = {z.name for z in self.zones}
        i = len(self.zones) + 1
        while f"zone-{i}" in taken:
            i += 1
        return f"zone-{i}"


def _render(cv2, frame, drafter: ZoneDrafter, paused: bool):
    canvas = frame.copy()
    for zone in drafter.zones:
        cv2.rectangle(canvas, (zone.x1, zone.y1), (zone.x2, zone.y2), (255, 128, 0), 2)
        cv2.putText(canvas, zone.name, (zone.x1 + 4, max(zone.y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
    if drafter.live_box:
        x1, y1, x2, y2 = drafter.live_box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
    lines = list(HELP_LINES)
    if paused:
        lines.append("PAUSED")
    for i, text in enumerate(lines):
        y = canvas.shape[0] - 12 - 22 * (len(lines) - 1 - i)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return canvas


def pick_zones(source, initial: list | None = None,
               window: str = "ZoneWatch - draw zones") -> list | None:
    """Open a window on `source` and let the user draw zones with the mouse.

    Returns the drawn zones, or None if the user cancelled.
    """
    import cv2

    with VideoStream(source) as stream:
        frame = stream.read()
        if frame is None:
            raise StreamUnavailable(f"Source yielded no frames: {source}")
        height, width = frame.shape[:2]
        drafter = ZoneDrafter(width, height, list(initial or []))

        def on_mouse(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                drafter.press(x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                drafter.drag(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                zone = drafter.release(x, y)
                if zone:
                    logger.info("Zone added: %s (%d,%d,%d,%d)",
                                zone.name, zone.x1, zone.y1, zone.x2, zone.y2)

        try:
            cv2.namedWindow(window)
            cv2.setMouseCallback(window, on_mouse)
        except cv2.error as exc:
            raise RuntimeError(
                "The zone picker needs a display, but this OpenCV build/environment "
                "has no GUI support. On a desktop, install opencv-python (not "
                "-headless); on a server, configure zones with --zone or ZONES instead."
            ) from exc
        paused = False
        try:
            while True:
                if not paused:
                    nxt = stream.read()
                    if nxt is not None:
                        frame = nxt
                cv2.imshow(window, _render(cv2, frame, drafter, paused))
                key = cv2.waitKey(30) & 0xFF
                if key in (13, ord("s")):  # ENTER or s
                    if drafter.zones:
                        return list(drafter.zones)
                    logger.warning("Draw at least one zone before saving")
                elif key == ord("u"):
                    drafter.undo()
                elif key == ord("r"):
                    drafter.reset()
                elif key == ord(" "):
                    paused = not paused
                elif key in (27, ord("q")):  # ESC or q
                    return None
        finally:
            cv2.destroyWindow(window)
