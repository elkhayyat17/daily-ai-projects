"""
Day 05 — Document Q&A with Vector Database
Centralized configuration for the entire project.
"""

from pathlib import Path


# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FAISS_INDEX_DIR = ARTIFACTS_DIR / "faiss_index"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for d in [RAW_DIR, PROCESSED_DIR, FAISS_INDEX_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Embedding Model ────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ─── Document Processing ────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md", ".docx", ".csv", ".json"}
MAX_FILE_SIZE_MB = 50
MAX_DOCUMENTS = 500

# ─── FAISS Index ─────────────────────────────────────────────────────
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
FAISS_METADATA_PATH = FAISS_INDEX_DIR / "metadata.pkl"
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.3

# ─── Answer Generation ──────────────────────────────────────────────
# Uses a local extractive QA model (no API key needed)
QA_MODEL_NAME = "deepset/minilm-uncased-squad2"
MAX_ANSWER_LENGTH = 512
NO_ANSWER_THRESHOLD = 0.01

# ─── API Settings ────────────────────────────────────────────────────
API_TITLE = "Document Q&A API"
API_DESCRIPTION = "Upload documents and ask questions — powered by vector search and extractive QA."
API_VERSION = "1.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Logging ─────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "app.log"
