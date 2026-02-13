from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from loguru import logger

from config import get_settings
from training.model import get_embeddings

SAMPLE_QUERIES = [
    ("What is RAG?", "doc-1"),
    ("What is ChromaDB?", "doc-2"),
    ("What is LangChain used for?", "doc-3"),
]


def evaluate() -> Path:
    settings = get_settings()
    if not settings.vector_db_dir.exists():
        raise FileNotFoundError("Vector store not found. Run training/train.py first.")

    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(settings.vector_db_dir),
        embedding_function=embeddings,
        collection_name="rag_docs",
    )

    hits = 0
    for query, expected_id in SAMPLE_QUERIES:
        results = vectorstore.similarity_search(query, k=1)
        if results and results[0].metadata.get("id") == expected_id:
            hits += 1

    hit_rate = hits / len(SAMPLE_QUERIES)
    report: Dict[str, float] = {"hit_rate": hit_rate, "queries": len(SAMPLE_QUERIES)}

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.artifacts_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Evaluation report saved to {}", report_path)
    return report_path


if __name__ == "__main__":
    evaluate()
