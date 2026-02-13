from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from config import get_settings
from data.prepare_data import prepare_data
from training.model import get_embeddings


def _load_docs(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            docs.append(
                Document(
                    page_content=payload["content"],
                    metadata={"id": payload["id"], "title": payload["title"]},
                )
            )
    return docs


def train() -> Path:
    settings = get_settings()
    docs_path = settings.processed_dir / "docs.jsonl"
    if not docs_path.exists():
        logger.warning("Processed docs missing. Preparing sample data...")
        docs_path = prepare_data()

    documents = _load_docs(docs_path)
    embeddings = get_embeddings()

    settings.vector_db_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(settings.vector_db_dir),
        collection_name="rag_docs",
    )
    logger.info("Vector store persisted to {}", settings.vector_db_dir)
    return settings.vector_db_dir


if __name__ == "__main__":
    train()
