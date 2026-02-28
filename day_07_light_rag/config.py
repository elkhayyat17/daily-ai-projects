"""Centralized configuration for Light RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root or parent directory
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
for _candidate in [_this_dir / ".env", _this_dir.parent / ".env"]:
    if _candidate.exists():
        load_dotenv(_candidate)
        break


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    # Paths ------------------------------------------------------------------
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("LIGHTRAG_BASE_DIR", str(Path(__file__).resolve().parent))
        )
    )
    data_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    index_dir: Path = field(init=False)

    # Project metadata -------------------------------------------------------
    project_name: str = field(
        default_factory=lambda: os.getenv(
            "LIGHTRAG_PROJECT_NAME", "Day 07 — Light RAG"
        )
    )
    version: str = field(
        default_factory=lambda: os.getenv("LIGHTRAG_VERSION", "0.1.0")
    )

    # Embedding --------------------------------------------------------------
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local").lower()
    )
    local_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
    )
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )
    openai_chat_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    )

    # Retrieval --------------------------------------------------------------
    top_k: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_TOP_K", "5"))
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_CHUNK_SIZE", "512"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("LIGHTRAG_CHUNK_OVERLAP", "64"))
    )
    bm25_weight: float = field(
        default_factory=lambda: float(os.getenv("LIGHTRAG_BM25_WEIGHT", "0.3"))
    )
    semantic_weight: float = field(
        default_factory=lambda: float(
            os.getenv("LIGHTRAG_SEMANTIC_WEIGHT", "0.7")
        )
    )
    max_context_chars: int = field(
        default_factory=lambda: int(
            os.getenv("LIGHTRAG_MAX_CONTEXT_CHARS", "4000")
        )
    )

    # Misc -------------------------------------------------------------------
    log_level: str = field(
        default_factory=lambda: os.getenv("LIGHTRAG_LOG_LEVEL", "INFO")
    )

    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        data_dir = Path(
            os.getenv("LIGHTRAG_DATA_DIR", str(self.base_dir / "data"))
        )
        processed_dir = Path(
            os.getenv("LIGHTRAG_PROCESSED_DIR", str(data_dir / "processed"))
        )
        artifacts_dir = Path(
            os.getenv("LIGHTRAG_ARTIFACTS_DIR", str(self.base_dir / "artifacts"))
        )
        index_dir = Path(
            os.getenv("LIGHTRAG_INDEX_DIR", str(artifacts_dir / "index"))
        )
        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "processed_dir", processed_dir)
        object.__setattr__(self, "artifacts_dir", artifacts_dir)
        object.__setattr__(self, "index_dir", index_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
