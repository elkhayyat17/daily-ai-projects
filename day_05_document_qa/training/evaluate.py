"""
Day 05 — Document Q&A: Evaluation
Evaluates retrieval quality and answer accuracy of the QA system.
"""

import json
import time
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from training.model import VectorStoreBuilder


# ─── Evaluation Queries ──────────────────────────────────────────────
EVAL_QUERIES = [
    {
        "question": "What is Python?",
        "expected_doc": "Python Programming Language",
        "expected_keywords": ["high-level", "programming", "Guido"],
    },
    {
        "question": "What are the types of machine learning?",
        "expected_doc": "Machine Learning Fundamentals",
        "expected_keywords": ["supervised", "unsupervised", "reinforcement"],
    },
    {
        "question": "What is the transformer architecture?",
        "expected_doc": "Transformers Architecture",
        "expected_keywords": ["attention", "encoder", "decoder"],
    },
    {
        "question": "What is FAISS?",
        "expected_doc": "Vector Databases and Similarity Search",
        "expected_keywords": ["Facebook", "similarity", "vector"],
    },
    {
        "question": "What is RAG?",
        "expected_doc": "Retrieval-Augmented Generation (RAG)",
        "expected_keywords": ["retrieval", "generation", "knowledge"],
    },
    {
        "question": "What is FastAPI?",
        "expected_doc": "FastAPI Web Framework",
        "expected_keywords": ["web", "framework", "Pydantic"],
    },
    {
        "question": "What is Docker?",
        "expected_doc": "Docker Containerization",
        "expected_keywords": ["container", "image", "Dockerfile"],
    },
    {
        "question": "What is NLP?",
        "expected_doc": "Natural Language Processing",
        "expected_keywords": ["language", "text", "BERT"],
    },
    {
        "question": "How does BERT work?",
        "expected_doc": "Natural Language Processing",
        "expected_keywords": ["bidirectional", "transformer", "understanding"],
    },
    {
        "question": "What are word embeddings?",
        "expected_doc": "Natural Language Processing",
        "expected_keywords": ["Word2Vec", "GloVe", "vectors"],
    },
]


def evaluate_retrieval(top_k: int = 5) -> dict:
    """
    Evaluate retrieval quality using the eval query set.

    Metrics:
        - Hit@1: Was the expected doc the top result?
        - Hit@K: Was the expected doc in the top K results?
        - MRR: Mean Reciprocal Rank
        - Keyword Coverage: Fraction of expected keywords found in retrieved text
    """
    logger.info("🔍 Evaluating retrieval quality...")

    builder = VectorStoreBuilder()
    if not builder.load():
        logger.error("No index found. Run train.py first!")
        return {}

    results = {
        "queries": [],
        "hit_at_1": 0,
        "hit_at_k": 0,
        "mrr_sum": 0.0,
        "keyword_coverage_sum": 0.0,
        "total": len(EVAL_QUERIES),
    }

    for query_data in EVAL_QUERIES:
        question = query_data["question"]
        expected_doc = query_data["expected_doc"]
        expected_keywords = query_data["expected_keywords"]

        search_results = builder.search(question, top_k=top_k)

        # Check hits
        titles_found = [r.get("title", "") for r in search_results]
        hit_at_1 = expected_doc in titles_found[:1]
        hit_at_k = expected_doc in titles_found

        # MRR
        reciprocal_rank = 0.0
        for i, title in enumerate(titles_found):
            if title == expected_doc:
                reciprocal_rank = 1.0 / (i + 1)
                break

        # Keyword coverage
        all_text = " ".join(r.get("text", "") for r in search_results).lower()
        keywords_found = sum(
            1 for kw in expected_keywords if kw.lower() in all_text
        )
        keyword_coverage = keywords_found / len(expected_keywords) if expected_keywords else 0

        query_result = {
            "question": question,
            "expected_doc": expected_doc,
            "hit_at_1": hit_at_1,
            "hit_at_k": hit_at_k,
            "reciprocal_rank": reciprocal_rank,
            "keyword_coverage": keyword_coverage,
            "top_result_title": titles_found[0] if titles_found else "N/A",
            "top_score": search_results[0]["score"] if search_results else 0.0,
        }
        results["queries"].append(query_result)

        results["hit_at_1"] += int(hit_at_1)
        results["hit_at_k"] += int(hit_at_k)
        results["mrr_sum"] += reciprocal_rank
        results["keyword_coverage_sum"] += keyword_coverage

    # Aggregate metrics
    n = results["total"]
    results["metrics"] = {
        "hit_rate_at_1": results["hit_at_1"] / n,
        "hit_rate_at_k": results["hit_at_k"] / n,
        "mrr": results["mrr_sum"] / n,
        "avg_keyword_coverage": results["keyword_coverage_sum"] / n,
        "top_k": top_k,
    }

    # Log results
    logger.info("=" * 60)
    logger.info("📊 Retrieval Evaluation Results")
    logger.info("=" * 60)
    for q in results["queries"]:
        status = "✅" if q["hit_at_1"] else ("⚠️" if q["hit_at_k"] else "❌")
        logger.info(
            f"  {status} {q['question']:<45s} "
            f"top='{q['top_result_title']}' score={q['top_score']:.3f}"
        )
    logger.info("-" * 60)
    m = results["metrics"]
    logger.info(f"  Hit@1:              {m['hit_rate_at_1']:.1%}")
    logger.info(f"  Hit@{top_k}:              {m['hit_rate_at_k']:.1%}")
    logger.info(f"  MRR:                {m['mrr']:.3f}")
    logger.info(f"  Keyword Coverage:   {m['avg_keyword_coverage']:.1%}")

    # Save results
    eval_path = config.ARTIFACTS_DIR / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"Saved evaluation results → {eval_path}")

    return results


if __name__ == "__main__":
    logger.add(config.LOG_FILE, rotation="10 MB", level=config.LOG_LEVEL)
    evaluate_retrieval()
