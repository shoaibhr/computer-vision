# ZoneWatch

**Zone-based person detection for RTSP cameras, webcams, and video files.**

[![CI](https://github.com/shoaibhr/computer-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/shoaibhr/computer-vision/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ZoneWatch watches a video feed, detects people with YOLO, and fires **enter/exit
events** whenever someone crosses into a zone you define — with webhooks,
annotated snapshots, and rock-solid stream reconnection. Point it at a security
camera and know within seconds when someone steps into your driveway, loading
dock, or restricted area.

```
┌──────────────┐   ┌──────────┐   ┌───────────────┐   ┌──────────────────┐
│ RTSP/webcam/ │──▶│   YOLO   │──▶│  zone monitor │──▶│ log / webhook /  │
│  video file  │   │ detector │   │  (debounced)  │   │    snapshot      │
└──────────────┘   └──────────┘   └───────────────┘   └──────────────────┘
```

## Features

- **Multiple named zones** — watch the entrance and the loading dock at once
- **Debounced events** — enter/exit fire on sustained presence, not one-frame flickers
- **Webhook notifications** — JSON `POST` on every event, ready for Home Assistant, n8n, Slack bridges, or your own backend
- **Annotated snapshots** — a JPEG of the moment someone entered, saved to disk
- **Notification cooldown** — no alert storms; one alert per zone per cooldown window
- **Automatic reconnection** — RTSP drops are retried with exponential backoff
- **Two detection backends**
  - `opencv` (default): YOLOv4-tiny via OpenCV DNN — CPU-friendly, no PyTorch, model auto-downloads on first run
  - `ultralytics` (optional): any YOLOv8/YOLO11 model for higher accuracy
- **Headless mode** — runs on servers and in Docker with no display
- **Live preview** — bounding boxes, zone status, and FPS overlay when a display is available

## Quick start

```sh
git clone https://github.com/shoaibhr/computer-vision
cd computer-vision
pip install .

# Watch your webcam, alert when someone enters the left half of a 640x480 frame
zonewatch --source 0 --zone "desk:0,0,320,480"

# Watch an RTSP camera headless, with webhook alerts
zonewatch --source "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/102" \
          --zone "entrance:118,6,218,206" \
          --webhook-url "https://example.com/hooks/zonewatch" \
          --headless
```

Press `q` in the preview window to quit. On first run the YOLOv4-tiny model
(~24 MB) is downloaded to `~/.cache/zonewatch/`.

For higher accuracy, install the Ultralytics backend:

```sh
pip install ".[yolo]"
zonewatch --source 0 --zone "desk:0,0,320,480" --backend ultralytics --model yolov8n.pt
```

## Configuration

Every option can be set as a CLI flag or an environment variable (a local
`.env` file is loaded automatically — see [.env.example](.env.example)):

| CLI flag | Env var | Default | Description |
|---|---|---|---|
| `--source` | `SOURCE` (or legacy `RTSP_URL`) | — | RTSP URL, webcam index, or video file path |
| `--zone` (repeatable) | `ZONES` (`;`-separated) | — | Zone as `name:x1,y1,x2,y2` |
| `--backend` | `BACKEND` | `opencv` | `opencv` or `ultralytics` |
| `--model` | `MODEL` | `yolov8n.pt` | Ultralytics model name/path |
| `--confidence` | `CONFIDENCE` | `0.5` | Minimum detection confidence (0–1) |
| `--detect-every` | `DETECT_EVERY` | `1` | Run detection every Nth frame (raise to trade latency for CPU) |
| `--enter-frames` | `ENTER_FRAMES` | `3` | Consecutive hits before ENTER fires |
| `--exit-frames` | `EXIT_FRAMES` | `15` | Consecutive misses before EXIT fires |
| `--webhook-url` | `WEBHOOK_URL` | — | POST events as JSON to this URL |
| `--snapshot-dir` | `SNAPSHOT_DIR` | — | Save annotated JPEGs on zone entry |
| `--cooldown` | `COOLDOWN` | `30` | Min seconds between alerts per zone |
| `--headless` | `HEADLESS` | `false` | Disable the preview window |
| `--log-level` | `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

CLI flags override environment variables. The legacy `ROI_X1..ROI_Y2`
variables from v0.x are still honoured when `ZONES` is unset.

### Webhook payload

```json
{
  "event": "enter",
  "zone": "entrance",
  "box": [118, 6, 218, 206],
  "timestamp": 1751900000.0
}
```

`event` is `"enter"` or `"exit"`. Enter events respect the per-zone cooldown;
exit events always fire so your integrations stay consistent.

## Docker

```sh
cp .env.example .env   # set SOURCE and ZONES
docker compose up -d --build
```

The compose file runs headless, persists the model cache in a volume, and
writes snapshots to `./snapshots`. Or with plain Docker:

```sh
docker build -t zonewatch .
docker run --env-file .env zonewatch
```

## Picking zone coordinates

Zones are pixel rectangles in the frame: `x1,y1` is the top-left corner,
`x2,y2` the bottom-right. Run ZoneWatch once with a preview window (or grab a
frame with `ffmpeg -i rtsp://... -frames:v 1 frame.jpg`) and read coordinates
off the image in any viewer. In the preview, zones are drawn blue when clear
and red while occupied.

## Development

```sh
pip install -e ".[dev]"
pytest          # unit tests (no camera or model needed)
ruff check src tests
```

The codebase is a small, typed package:

```
src/zonewatch/
├── cli.py         # argparse CLI, env/flag merging
├── config.py      # Settings dataclass + validation
├── detectors.py   # OpenCV-DNN and Ultralytics backends
├── stream.py      # VideoCapture wrapper with auto-reconnect
├── zones.py       # zone parsing + box intersection
├── events.py      # debounced enter/exit state machine, cooldowns
├── notifiers.py   # log, webhook, snapshot sinks
└── app.py         # capture → detect → track → notify loop
```

CI runs lint and tests on Python 3.9–3.12 for every push and pull request.

## Troubleshooting

- **Stream won't open** — verify the RTSP URL with VLC or `ffplay` first;
  ZoneWatch retries 5 times with backoff before giving up.
- **Model download fails** — on an offline machine, download
  [`yolov4-tiny.weights`](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights),
  [`yolov4-tiny.cfg`](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg), and
  [`coco.names`](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names)
  manually into `~/.cache/zonewatch/` or the working directory.
- **High CPU usage** — raise `--detect-every` (e.g. `3` detects on every third
  frame), or use a sub-stream from your camera at lower resolution.
- **Missed or flickering alerts** — lower `--enter-frames` for faster alerts,
  raise `--exit-frames` if exits fire while someone is still there, and tune
  `--confidence` (lower = more sensitive).
- **No GUI / `cv2.imshow` errors on a server** — run with `--headless`.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [YOLOv4-tiny](https://github.com/AlexeyAB/darknet) and [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for detection models
- [OpenCV](https://opencv.org/) for video capture and image processing
