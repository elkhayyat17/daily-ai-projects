"""Evaluate the Light RAG index — hit-rate, MRR, and hybrid vs single mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from config import get_settings
from training.model import get_embeddings
from training.train import LightIndex, _tokenize

# ---------------------------------------------------------------------------
# Ground-truth evaluation queries  (query → expected doc_id)
# ---------------------------------------------------------------------------
EVAL_QUERIES: List[Tuple[str, str]] = [
    ("What is RAG and how does it work?", "doc-01"),
    ("Explain BM25 ranking function", "doc-02"),
    ("How do sentence transformers create embeddings?", "doc-03"),
    ("What is hybrid search?", "doc-04"),
    ("Best chunking strategies for retrieval", "doc-05"),
    ("What is FastAPI?", "doc-06"),
    ("NumPy vector similarity search", "doc-07"),
    ("How to write prompts for RAG?", "doc-08"),
    ("Metrics to evaluate a RAG system", "doc-09"),
    ("Difference between light and heavy RAG", "doc-10"),
]


def _semantic_search(
    query_vec: np.ndarray,
    index: LightIndex,
    k: int,
) -> List[Tuple[int, float]]:
    """Return (chunk_idx, cosine_similarity) top-k."""
    scores = index.embeddings @ query_vec.T
    scores = scores.flatten()
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_idx]


def _bm25_search(
    query: str,
    index: LightIndex,
    k: int,
) -> List[Tuple[int, float]]:
    """Return (chunk_idx, bm25_score) top-k."""
    tokens = _tokenize(query)
    scores = index.bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_idx]


def _hybrid_search(
    query: str,
    query_vec: np.ndarray,
    index: LightIndex,
    k: int,
    bm25_weight: float = 0.3,
    semantic_weight: float = 0.7,
) -> List[Tuple[int, float]]:
    """Weighted fusion of BM25 and semantic scores."""
    # Semantic
    sem_scores = (index.embeddings @ query_vec.T).flatten()
    sem_min, sem_max = sem_scores.min(), sem_scores.max()
    if sem_max - sem_min > 0:
        sem_norm = (sem_scores - sem_min) / (sem_max - sem_min)
    else:
        sem_norm = np.zeros_like(sem_scores)

    # BM25
    bm25_scores = index.bm25.get_scores(_tokenize(query))
    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    if bm25_max - bm25_min > 0:
        bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    fused = semantic_weight * sem_norm + bm25_weight * bm25_norm
    top_idx = np.argsort(fused)[::-1][:k]
    return [(int(i), float(fused[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(k: int = 5) -> Path:
    """Run evaluation and persist a JSON report."""
    settings = get_settings()
    if not settings.index_dir.exists():
        raise FileNotFoundError(
            f"Index not found at {settings.index_dir}. Run training/train.py first."
        )

    index = LightIndex.load(settings.index_dir)
    embedder = get_embeddings()

    results: Dict[str, dict] = {}
    for mode in ("semantic", "bm25", "hybrid"):
        hits = 0
        rr_sum = 0.0
        for query, expected_doc_id in EVAL_QUERIES:
            query_vec = embedder.encode([query])
            if mode == "semantic":
                ranked = _semantic_search(query_vec, index, k)
            elif mode == "bm25":
                ranked = _bm25_search(query, index, k)
            else:
                ranked = _hybrid_search(
                    query,
                    query_vec,
                    index,
                    k,
                    bm25_weight=settings.bm25_weight,
                    semantic_weight=settings.semantic_weight,
                )

            retrieved_doc_ids = [
                index.metadata[idx]["doc_id"] for idx, _ in ranked
            ]
            if expected_doc_id in retrieved_doc_ids:
                hits += 1
                rank = retrieved_doc_ids.index(expected_doc_id) + 1
                rr_sum += 1.0 / rank

        n = len(EVAL_QUERIES)
        results[mode] = {
            "hit_rate": round(hits / n, 4),
            "mrr": round(rr_sum / n, 4),
            "queries": n,
            "k": k,
        }

    report = {"modes": results}
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.artifacts_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Evaluation report → {}", report_path)
    for mode, metrics in results.items():
        logger.info(
            "  {}: hit_rate={:.2%}  MRR={:.4f}",
            mode,
            metrics["hit_rate"],
            metrics["mrr"],
        )
    return report_path


if __name__ == "__main__":
    evaluate()
