"""
Day 06 — Object Detection API (YOLOv8)
Centralized configuration for the entire project.
"""

from pathlib import Path


# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── YOLOv8 Model ───────────────────────────────────────────────────
# Pre-trained model variants: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
YOLO_MODEL_NAME = "yolov8n.pt"  # Nano — fastest, smallest
YOLO_MODEL_PATH = MODELS_DIR / "best.pt"
YOLO_PRETRAINED_PATH = MODELS_DIR / YOLO_MODEL_NAME

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

NUM_CLASSES = len(COCO_CLASSES)

# ─── Detection Settings ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100
IMAGE_SIZE = 640  # Input image size for YOLOv8

# ─── Image Processing ───────────────────────────────────────────────
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
MAX_IMAGE_SIZE_MB = 20
MAX_IMAGE_DIMENSION = 4096

# ─── Training ────────────────────────────────────────────────────────
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 16
TRAIN_IMAGE_SIZE = 640
TRAIN_LR = 0.01
TRAIN_PATIENCE = 10
TRAIN_WORKERS = 4

# ─── API Settings ────────────────────────────────────────────────────
API_TITLE = "Object Detection API"
API_DESCRIPTION = "Detect objects in images using YOLOv8 — real-time, production-ready."
API_VERSION = "1.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Logging ─────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "app.log"
