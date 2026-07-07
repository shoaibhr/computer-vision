"""Command-line interface for ZoneWatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from . import __version__
from .config import Settings, load_settings, parse_source, save_zones_to_env
from .zones import parse_zones


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zonewatch",
        description="Zone-based person detection for any video source: USB/laptop cameras, "
        "RTSP/HTTP network cameras, video files, or GStreamer pipelines. "
        "Options fall back to environment variables (a local .env file is loaded "
        "automatically).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--source", "-s",
                        help="camera index (0, 1, ...), /dev/videoN, 'auto' (first working "
                        "camera), RTSP/HTTP URL, video file, or GStreamer pipeline "
                        "(env: SOURCE)")
    parser.add_argument("--list-sources", action="store_true",
                        help="scan for connected cameras and exit")
    parser.add_argument("--zone", "-z", action="append", dest="zones", metavar="NAME:X1,Y1,X2,Y2",
                        help="detection zone; repeat for multiple zones (env: ZONES). "
                        "If no zones are configured, an interactive picker opens instead")
    parser.add_argument("--select-zones", action="store_true",
                        help="draw zones on the live video with the mouse, save them to .env, "
                        "then start watching")
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


def _list_sources() -> int:
    from .stream import probe_cameras

    print("Scanning camera indexes 0-9 ...")
    cameras = probe_cameras()
    if not cameras:
        print("No working cameras found. Network cameras (rtsp://...) and video "
              "files are not probed; pass them with --source.")
        return 1
    for cam in cameras:
        fps = f"{cam.fps:.0f} fps" if cam.fps else "unknown fps"
        print(f"  --source {cam.index}   {cam.width}x{cam.height} @ {fps}")
    return 0


def _select_zones(settings: Settings) -> int | None:
    """Run the interactive picker and persist the result. Returns an exit
    code on failure/cancel, or None to continue into the watch loop."""
    from .picker import pick_zones

    zones = pick_zones(settings.source, initial=settings.zones)
    if zones is None:
        print("Zone selection cancelled.", file=sys.stderr)
        return 1
    settings.zones = zones
    spec = save_zones_to_env(zones, Path(".env"))
    print(f"Saved {len(zones)} zone(s) to .env: ZONES={spec}")
    return None


def main(argv=None) -> int:
    # Search for .env from the user's working directory, not the package dir
    load_dotenv(find_dotenv(usecwd=True))
    args = build_parser().parse_args(argv)

    if args.list_sources:
        return _list_sources()

    try:
        settings = load_settings(
            source=parse_source(args.source) if args.source else None,
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
        if args.select_zones or not settings.zones:
            if settings.headless:
                if args.select_zones:
                    print("--select-zones needs a display; drop --headless.", file=sys.stderr)
                else:
                    print("Configuration error: no zones configured. Run once without "
                          "--headless to draw them, or set ZONES / --zone "
                          "(e.g. --zone \"entrance:118,6,218,206\").", file=sys.stderr)
                return 2
            exit_code = _select_zones(settings)
            if exit_code is not None:
                return exit_code
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
