"""
Centralized Configuration for Sentiment Analysis Pipeline
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

# Create directories if they don't exist
for d in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
MAX_LENGTH = 256

# ──────────────────────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────────────────────
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
SEED = 42

# ──────────────────────────────────────────────────────────────
# API Configuration
# ──────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_VERSION = "1.0.0"
API_TITLE = "Sentiment Analysis API"
API_DESCRIPTION = "Real-time sentiment analysis powered by DistilBERT"

# ──────────────────────────────────────────────────────────────
# Streamlit Configuration
# ──────────────────────────────────────────────────────────────
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
