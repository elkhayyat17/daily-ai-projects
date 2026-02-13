from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    base_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_BASE_DIR", Path(__file__).resolve().parent)))
    data_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    vector_db_dir: Path = field(init=False)

    project_name: str = field(default_factory=lambda: os.getenv("RAG_PROJECT_NAME", "Day 03 - RAG Chatbot API"))
    version: str = field(default_factory=lambda: os.getenv("RAG_VERSION", "0.1.0"))

    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai").lower())
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    local_embedding_model: str = field(default_factory=lambda: os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "4")))
    max_context_chars: int = field(default_factory=lambda: int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000")))
    log_level: str = field(default_factory=lambda: os.getenv("RAG_LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        data_dir = Path(os.getenv("RAG_DATA_DIR", str(self.base_dir / "data")))
        processed_dir = Path(os.getenv("RAG_PROCESSED_DIR", str(data_dir / "processed")))
        artifacts_dir = Path(os.getenv("RAG_ARTIFACTS_DIR", str(self.base_dir / "artifacts")))
        vector_db_dir = Path(os.getenv("RAG_VECTOR_DB_DIR", str(artifacts_dir / "chroma")))

        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "processed_dir", processed_dir)
        object.__setattr__(self, "artifacts_dir", artifacts_dir)
        object.__setattr__(self, "vector_db_dir", vector_db_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""

    return Settings()
