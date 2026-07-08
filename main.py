"""Backwards-compatible entry point.

The original single-file script grew into the ``zonewatch`` package.
``python main.py`` still works and honours the legacy .env variables
(RTSP_URL, ROI_X1..ROI_Y2); new deployments should use the ``zonewatch``
CLI installed via ``pip install .``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from zonewatch.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
