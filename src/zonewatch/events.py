"""Debounced zone occupancy tracking.

Raw per-frame detections flicker; this module turns them into stable
ENTER/EXIT events: a zone becomes occupied only after ``enter_frames``
consecutive hits, and empty only after ``exit_frames`` consecutive misses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .zones import Zone


@dataclass(frozen=True)
class ZoneEvent:
    kind: str  # "enter" or "exit"
    zone: Zone
    timestamp: float

    def to_payload(self) -> dict:
        return {
            "event": self.kind,
            "zone": self.zone.name,
            "box": [self.zone.x1, self.zone.y1, self.zone.x2, self.zone.y2],
            "timestamp": self.timestamp,
        }


@dataclass
class _ZoneState:
    occupied: bool = False
    hits: int = 0
    misses: int = 0


@dataclass
class ZoneMonitor:
    zones: list[Zone]
    enter_frames: int = 3
    exit_frames: int = 15
    clock: Callable[[], float] = time.time
    _states: dict = field(init=False)

    def __post_init__(self) -> None:
        self._states = {zone.name: _ZoneState() for zone in self.zones}

    def is_occupied(self, zone: Zone) -> bool:
        return self._states[zone.name].occupied

    def update(self, boxes: list) -> list[ZoneEvent]:
        """Feed one frame's person boxes; return any ENTER/EXIT events fired."""
        events: list[ZoneEvent] = []
        now = self.clock()
        for zone in self.zones:
            state = self._states[zone.name]
            hit = any(zone.intersects(box) for box in boxes)
            if hit:
                state.hits += 1
                state.misses = 0
            else:
                state.misses += 1
                state.hits = 0
            if not state.occupied and state.hits >= self.enter_frames:
                state.occupied = True
                events.append(ZoneEvent("enter", zone, now))
            elif state.occupied and state.misses >= self.exit_frames:
                state.occupied = False
                events.append(ZoneEvent("exit", zone, now))
        return events


@dataclass
class Cooldown:
    """Rate-limits notifications per key (zone name)."""

    seconds: float
    clock: Callable[[], float] = time.time
    _last: dict = field(default_factory=dict, init=False)

    def ready(self, key: str) -> bool:
        """Return True (and start the cooldown) if `key` is not rate-limited."""
        now = self.clock()
        last: float | None = self._last.get(key)
        if last is not None and now - last < self.seconds:
            return False
        self._last[key] = now
        return True
