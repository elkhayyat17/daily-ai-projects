"""
Centralized Configuration for Image Classification Pipeline
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Dataset Configuration
# ──────────────────────────────────────────────────────────────
DATASET_NAME = "CIFAR-10"
NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CLASS_EMOJIS = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦", "cat": "🐱",
    "deer": "🦌", "dog": "🐶", "frog": "🐸", "horse": "🐴",
    "ship": "🚢", "truck": "🚚",
}

# ──────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "resnet50"
PRETRAINED = True
INPUT_SIZE = 224  # ResNet expects 224x224
FREEZE_BACKBONE = True  # Freeze early layers for transfer learning
UNFREEZE_AFTER_EPOCH = 1  # Unfreeze backbone after N epochs for fine-tuning

# ──────────────────────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────────────────────
LEARNING_RATE = 1e-3
BACKBONE_LR = 1e-5  # Lower LR for pretrained backbone
BATCH_SIZE = 64
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 4
USE_AMP = True  # Automatic Mixed Precision

# ──────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────
AUGMENTATION_ENABLED = True
RANDOM_CROP_PADDING = 4
HORIZONTAL_FLIP_PROB = 0.5
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
RANDOM_ERASING_PROB = 0.1

# ──────────────────────────────────────────────────────────────
# Normalization (ImageNet stats)
# ──────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────
# API Configuration
# ──────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_VERSION = "1.0.0"
API_TITLE = "Image Classification API"
API_DESCRIPTION = "Classify images into 10 categories using ResNet50 transfer learning"
MAX_UPLOAD_SIZE_MB = 10

# ──────────────────────────────────────────────────────────────
# Streamlit Configuration
# ──────────────────────────────────────────────────────────────
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
