from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from loguru import logger

from config import get_settings

SAMPLE_DOCS = [
    {
        "id": "doc-1",
        "title": "RAG Overview",
        "content": "Retrieval-Augmented Generation (RAG) combines vector search with language models to ground answers in relevant documents.",
    },
    {
        "id": "doc-2",
        "title": "ChromaDB",
        "content": "ChromaDB is an open-source vector database optimized for storing embeddings and performing similarity search.",
    },
    {
        "id": "doc-3",
        "title": "LangChain",
        "content": "LangChain provides abstractions for building LLM applications, including retrievers, document loaders, and chains.",
    },
    {
        "id": "doc-4",
        "title": "FastAPI",
        "content": "FastAPI is a modern Python web framework for building fast APIs with type hints and automatic OpenAPI docs.",
    },
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, items: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_data() -> Path:
    """Prepare sample docs for RAG training.

    Returns:
        Path to the processed docs jsonl file.
    """

    settings = get_settings()
    processed_dir = settings.processed_dir
    _ensure_dir(processed_dir)

    output_path = processed_dir / "docs.jsonl"
    _write_jsonl(output_path, SAMPLE_DOCS)
    logger.info("Prepared sample docs at {}", output_path)
    return output_path


if __name__ == "__main__":
    prepare_data()
