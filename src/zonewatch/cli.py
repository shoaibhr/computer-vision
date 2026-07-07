"""Command-line interface for ZoneWatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .config import _coerce_source, load_settings
from .zones import parse_zones


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zonewatch",
        description="Zone-based person detection for RTSP cameras, webcams, and video files. "
        "Options fall back to environment variables (a local .env file is loaded "
        "automatically).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--source", "-s",
                        help="RTSP URL, webcam index, or video file path (env: SOURCE)")
    parser.add_argument("--zone", "-z", action="append", dest="zones", metavar="NAME:X1,Y1,X2,Y2",
                        help="detection zone; repeat for multiple zones (env: ZONES)")
    parser.add_argument("--backend", choices=["opencv", "ultralytics"],
                        help="detection backend (env: BACKEND, default: opencv)")
    parser.add_argument("--model", help="Ultralytics model name/path (env: MODEL)")
    parser.add_argument("--confidence", "-c", type=float,
                        help="minimum detection confidence 0-1 (env: CONFIDENCE, default: 0.5)")
    parser.add_argument("--detect-every", type=int, metavar="N",
                        help="run detection every Nth frame (env: DETECT_EVERY, default: 1)")
    parser.add_argument("--enter-frames", type=int,
                        help="consecutive hits before ENTER fires (env: ENTER_FRAMES)")
    parser.add_argument("--exit-frames", type=int,
                        help="consecutive misses before EXIT fires (env: EXIT_FRAMES)")
    parser.add_argument("--webhook-url", help="POST events to this URL (env: WEBHOOK_URL)")
    parser.add_argument("--snapshot-dir", type=Path,
                        help="save annotated snapshots on entry (env: SNAPSHOT_DIR)")
    parser.add_argument("--cooldown", type=float,
                        help="min seconds between notifications per zone (env: COOLDOWN)")
    parser.add_argument("--headless", action="store_const", const=True,
                        help="disable the preview window (env: HEADLESS)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="logging verbosity (env: LOG_LEVEL, default: INFO)")
    return parser


def main(argv=None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(
            source=_coerce_source(args.source) if args.source else None,
            zones=parse_zones(";".join(args.zones)) if args.zones else None,
            backend=args.backend,
            model=args.model,
            confidence=args.confidence,
            detect_every=args.detect_every,
            enter_frames=args.enter_frames,
            exit_frames=args.exit_frames,
            webhook_url=args.webhook_url,
            snapshot_dir=args.snapshot_dir,
            cooldown=args.cooldown,
            headless=args.headless,
            log_level=args.log_level,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    from .app import run
    from .stream import StreamUnavailable

    try:
        run(settings)
    except StreamUnavailable as exc:
        print(f"Stream error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
