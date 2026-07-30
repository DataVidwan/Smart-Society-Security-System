from ultralytics import YOLO
import os

_model = None

def load_model():
    global _model

    if _model is not None:
        return _model

    possible_paths = [
        "best.pt",
        "runs/detect/train2/weights/best.pt",
        "yolov8_plate.pt"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"[INFO] Loading YOLO model: {path}")
            _model = YOLO(path)
            return _model

    raise FileNotFoundError(
        "No YOLO model found. Place best.pt in the project root."
    )