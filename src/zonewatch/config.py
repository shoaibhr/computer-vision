"""Configuration: environment variables (.env supported) with CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .zones import Zone, parse_zones

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass
class Settings:
    source: str | int
    zones: list[Zone] = field(default_factory=list)
    backend: str = "opencv"
    model: str = "yolov8n.pt"
    confidence: float = 0.5
    detect_every: int = 1
    enter_frames: int = 3
    exit_frames: int = 15
    webhook_url: str | None = None
    snapshot_dir: Path | None = None
    cooldown: float = 30.0
    headless: bool = False
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.zones:
            raise ValueError("At least one detection zone is required (set ZONES or --zone)")
        if self.backend not in ("opencv", "ultralytics"):
            raise ValueError(f"Unknown backend '{self.backend}' (expected opencv or ultralytics)")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        if self.detect_every < 1:
            raise ValueError("detect_every must be >= 1")
        if self.enter_frames < 1 or self.exit_frames < 1:
            raise ValueError("enter_frames and exit_frames must be >= 1")
        names = [z.name for z in self.zones]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate zone names: {', '.join(sorted(dupes))}")


def _coerce_source(raw: str) -> str | int:
    """Webcam indexes are given as bare integers; everything else is a URL/path."""
    return int(raw) if raw.isdigit() else raw


def _zones_from_env(env: dict) -> list[Zone]:
    spec = env.get("ZONES")
    if spec:
        return parse_zones(spec)
    # Legacy single-ROI variables from v0.x
    legacy = [env.get(k) for k in ("ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2")]
    if all(v is not None for v in legacy):
        x1, y1, x2, y2 = (int(v) for v in legacy)
        return [Zone("roi", x1, y1, x2, y2)]
    return []


def load_settings(env: dict | None = None, **overrides) -> Settings:
    """Build Settings from an environment mapping, then apply keyword overrides.

    ``overrides`` values of None are ignored, so CLI flags that weren't
    passed fall through to the environment (and then to defaults).
    """
    if env is None:
        env = dict(os.environ)

    values: dict = {}
    source = env.get("SOURCE") or env.get("RTSP_URL")
    if source:
        values["source"] = _coerce_source(source)
    zones = _zones_from_env(env)
    if zones:
        values["zones"] = zones
    if env.get("BACKEND"):
        values["backend"] = env["BACKEND"].lower()
    if env.get("MODEL"):
        values["model"] = env["MODEL"]
    if env.get("CONFIDENCE"):
        values["confidence"] = float(env["CONFIDENCE"])
    if env.get("DETECT_EVERY"):
        values["detect_every"] = int(env["DETECT_EVERY"])
    if env.get("ENTER_FRAMES"):
        values["enter_frames"] = int(env["ENTER_FRAMES"])
    if env.get("EXIT_FRAMES"):
        values["exit_frames"] = int(env["EXIT_FRAMES"])
    if env.get("WEBHOOK_URL"):
        values["webhook_url"] = env["WEBHOOK_URL"]
    if env.get("SNAPSHOT_DIR"):
        values["snapshot_dir"] = Path(env["SNAPSHOT_DIR"])
    if env.get("COOLDOWN"):
        values["cooldown"] = float(env["COOLDOWN"])
    if env.get("HEADLESS"):
        values["headless"] = env["HEADLESS"].strip().lower() in _TRUTHY
    if env.get("LOG_LEVEL"):
        values["log_level"] = env["LOG_LEVEL"].upper()

    values.update({k: v for k, v in overrides.items() if v is not None})

    if "source" not in values:
        raise ValueError("No video source configured (set SOURCE in .env or pass --source)")
    return Settings(**values)
