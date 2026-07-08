"""Person detection backends.

- ``opencv``: YOLOv4-tiny via OpenCV's DNN module. Lightweight (no PyTorch);
  model files are downloaded automatically on first run.
- ``ultralytics``: any Ultralytics YOLO model (yolov8n.pt, yolo11n.pt, ...).
  Higher accuracy; requires the ``zonewatch[yolo]`` extra.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = Path.home() / ".cache" / "zonewatch"

_YOLOV4_TINY_FILES = {
    "yolov4-tiny.weights": (
        "https://github.com/AlexeyAB/darknet/releases/download/"
        "darknet_yolo_v4_pre/yolov4-tiny.weights"
    ),
    "yolov4-tiny.cfg": (
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
    ),
    "coco.names": ("https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"),
}


@dataclass(frozen=True)
class Detection:
    box: tuple  # (x1, y1, x2, y2) in pixels
    confidence: float
    label: str


def _ensure_model_files(model_dir: Path) -> dict:
    """Locate YOLOv4-tiny files, preferring the CWD (legacy layout), else
    download them once into ``model_dir``."""
    paths = {}
    for filename, url in _YOLOV4_TINY_FILES.items():
        local = Path(filename)
        if local.is_file():  # backwards compatible with v0.x manual setup
            paths[filename] = local
            continue
        cached = model_dir / filename
        if not cached.is_file():
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s from %s ...", filename, url)
            tmp = cached.with_suffix(cached.suffix + ".part")
            try:
                urllib.request.urlretrieve(url, tmp)  # noqa: S310
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not download {filename} ({exc}). If this machine has no "
                    f"internet access, download it manually from {url} and place it in "
                    f"{model_dir} or the working directory."
                ) from exc
            tmp.rename(cached)
            logger.info("Saved %s (%.1f MB)", cached, cached.stat().st_size / 1e6)
        paths[filename] = cached
    return paths


class OpenCVDetector:
    """YOLOv4-tiny person detector using cv2.dnn (CPU, no heavy deps)."""

    def __init__(self, confidence: float = 0.5, nms_threshold: float = 0.4,
                 model_dir: Path = MODEL_DIR):
        import cv2

        self._cv2 = cv2
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        paths = _ensure_model_files(model_dir)
        self.net = cv2.dnn.readNet(str(paths["yolov4-tiny.weights"]),
                                   str(paths["yolov4-tiny.cfg"]))
        self.classes = paths["coco.names"].read_text().splitlines()
        layer_names = self.net.getLayerNames()
        self._output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        logger.info("OpenCV YOLOv4-tiny detector ready")

    def detect(self, frame) -> list:
        import numpy as np

        cv2 = self._cv2
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self._output_layers)

        boxes, confidences = [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence >= self.confidence and self.classes[class_id] == "person":
                    cx, cy = detection[0] * width, detection[1] * height
                    w, h = detection[2] * width, detection[3] * height
                    boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
                    confidences.append(confidence)

        detections = []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.nms_threshold)
        for i in np.array(indices).flatten():
            x, y, w, h = boxes[i]
            detections.append(Detection((x, y, x + w, y + h), confidences[i], "person"))
        return detections


class UltralyticsDetector:
    """Person detector backed by an Ultralytics YOLO model (yolov8n.pt, ...)."""

    def __init__(self, model: str = "yolov8n.pt", confidence: float = 0.5):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The 'ultralytics' backend requires extra dependencies: "
                "pip install zonewatch[yolo]"
            ) from exc
        self.confidence = confidence
        self.model = YOLO(model)
        logger.info("Ultralytics detector ready (%s)", model)

    def detect(self, frame) -> list:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        detections = []
        for result in results:
            names = result.names
            for box in result.boxes:
                label = names[int(box.cls[0])]
                if label != "person":
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(Detection((x1, y1, x2, y2), float(box.conf[0]), "person"))
        return detections


def create_detector(backend: str, model: str, confidence: float):
    if backend == "ultralytics":
        return UltralyticsDetector(model=model, confidence=confidence)
    return OpenCVDetector(confidence=confidence)
