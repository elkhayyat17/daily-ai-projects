"""Data preparation — download / generate sample documents and chunk them."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from loguru import logger

from config import get_settings

# ---------------------------------------------------------------------------
# Sample corpus — diverse topics so hybrid search is meaningful
# ---------------------------------------------------------------------------
SAMPLE_DOCS: List[Dict[str, str]] = [
    {
        "id": "doc-01",
        "title": "What is Retrieval-Augmented Generation?",
        "content": (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances "
            "large language model outputs by first retrieving relevant documents "
            "from an external knowledge base and injecting that context into the "
            "prompt. This grounds the model's answers in factual data and reduces "
            "hallucinations. RAG pipelines typically consist of an indexing stage, "
            "a retrieval stage, and a generation stage. During indexing, documents "
            "are chunked and embedded into a vector space. At query time the user's "
            "question is embedded with the same model and the most similar chunks "
            "are retrieved via approximate nearest-neighbor search."
        ),
    },
    {
        "id": "doc-02",
        "title": "BM25 — The Classic Keyword Retriever",
        "content": (
            "BM25 (Best Matching 25) is a probabilistic ranking function used in "
            "information retrieval. It scores documents based on term frequency, "
            "inverse document frequency, and document length normalization. BM25 "
            "is fast, interpretable, and surprisingly effective for keyword-heavy "
            "queries. In hybrid RAG systems, BM25 complements dense vector search "
            "by capturing exact lexical matches that embedding models may miss."
        ),
    },
    {
        "id": "doc-03",
        "title": "Sentence-Transformers for Dense Retrieval",
        "content": (
            "Sentence-Transformers is a Python library that provides pre-trained "
            "models for computing dense vector representations of sentences and "
            "paragraphs. Models like all-MiniLM-L6-v2 map text to a 384-dimensional "
            "space where semantically similar texts are close together. These "
            "embeddings power dense retrieval: given a query embedding, cosine "
            "similarity identifies the most relevant passages regardless of exact "
            "keyword overlap."
        ),
    },
    {
        "id": "doc-04",
        "title": "Hybrid Search — Combining BM25 and Vector Retrieval",
        "content": (
            "Hybrid search fuses sparse (BM25) and dense (embedding) retrieval to "
            "leverage the strengths of both. A common approach is Reciprocal Rank "
            "Fusion (RRF), which merges ranked lists without needing score "
            "calibration. Alternatively, a simple weighted sum of normalized scores "
            "can blend keyword relevance with semantic similarity. Hybrid search "
            "consistently outperforms either method alone on benchmarks like BEIR."
        ),
    },
    {
        "id": "doc-05",
        "title": "Chunking Strategies for RAG",
        "content": (
            "Good chunking is critical for RAG quality. Common strategies include "
            "fixed-size chunking (e.g., 512 tokens with overlap), recursive "
            "character splitting that respects paragraph and sentence boundaries, "
            "and semantic chunking that groups sentences with similar embeddings. "
            "Chunk size directly impacts retrieval granularity: smaller chunks "
            "improve precision but may lose context; larger chunks retain context "
            "but reduce recall."
        ),
    },
    {
        "id": "doc-06",
        "title": "FastAPI — Modern Python Web Framework",
        "content": (
            "FastAPI is a high-performance Python web framework for building APIs. "
            "It leverages Python type hints and Pydantic for automatic request "
            "validation, serialization, and OpenAPI documentation. FastAPI supports "
            "asynchronous request handling via ASGI and is one of the fastest "
            "Python frameworks available, making it ideal for serving ML models."
        ),
    },
    {
        "id": "doc-07",
        "title": "NumPy-Based Vector Search",
        "content": (
            "For small-to-medium corpora (up to ~100k vectors), a simple NumPy "
            "dot-product search is fast and dependency-free. Embeddings are stored "
            "as a 2-D float32 array; at query time the query vector is normalized "
            "and multiplied against the matrix to produce cosine similarities. This "
            "eliminates the need for dedicated vector databases like FAISS or "
            "ChromaDB, keeping the stack lightweight."
        ),
    },
    {
        "id": "doc-08",
        "title": "Prompt Engineering for RAG",
        "content": (
            "Effective RAG prompts instruct the language model to answer strictly "
            "from the provided context. A typical template includes a system "
            "message describing the assistant's role, the retrieved passages under "
            "a 'Context' header, and the user question. Adding instructions like "
            "'If the context does not contain the answer, say I don't know' "
            "reduces confabulation."
        ),
    },
    {
        "id": "doc-09",
        "title": "Evaluation Metrics for RAG Systems",
        "content": (
            "RAG systems are evaluated on both retrieval quality and generation "
            "quality. Retrieval metrics include Hit Rate (whether the gold document "
            "appears in top-k), Mean Reciprocal Rank (MRR), and Normalized "
            "Discounted Cumulative Gain (nDCG). Generation metrics include ROUGE, "
            "BERTScore, and faithfulness (how well the answer is supported by the "
            "retrieved context)."
        ),
    },
    {
        "id": "doc-10",
        "title": "Light RAG vs Heavy RAG",
        "content": (
            "A 'Light RAG' system minimizes external dependencies by using NumPy "
            "for vector search, rank-bm25 for keyword matching, and sentence-"
            "transformers for embeddings — no LangChain, no vector database server. "
            "This makes it easy to deploy in constrained environments. A 'Heavy "
            "RAG' stack adds orchestration frameworks, managed vector databases, "
            "re-rankers, and query routers for enterprise-scale workloads."
        ),
    },
]


# ---------------------------------------------------------------------------
# Lightweight recursive text splitter
# ---------------------------------------------------------------------------
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
) -> List[str]:
    """Split *text* into overlapping chunks respecting sentence boundaries."""
    seps = separators or _SEPARATORS
    chunks: List[str] = []
    _recursive_split(text, seps, chunk_size, chunk_overlap, chunks)
    return [c.strip() for c in chunks if c.strip()]


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int,
    out: List[str],
) -> None:
    if len(text) <= chunk_size:
        out.append(text)
        return

    sep = separators[0] if separators else ""
    parts = text.split(sep) if sep else list(text)

    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) > chunk_size and current:
            out.append(current)
            # overlap: keep the tail of the previous chunk
            tail = current[-overlap:] if overlap else ""
            current = tail + sep + part if tail else part
        else:
            current = candidate

    if current:
        if len(current) > chunk_size and len(separators) > 1:
            _recursive_split(current, separators[1:], chunk_size, overlap, out)
        else:
            out.append(current)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, items: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_data() -> Path:
    """Prepare sample documents → chunk → write to JSONL."""
    settings = get_settings()
    _ensure_dir(settings.processed_dir)

    chunks: List[dict] = []
    for doc in SAMPLE_DOCS:
        doc_chunks = chunk_text(
            doc["content"],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        for idx, chunk in enumerate(doc_chunks):
            chunks.append(
                {
                    "id": f"{doc['id']}-chunk-{idx}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "content": chunk,
                }
            )

    output_path = settings.processed_dir / "chunks.jsonl"
    _write_jsonl(output_path, chunks)
    logger.info(
        "Prepared {} chunks from {} docs → {}",
        len(chunks),
        len(SAMPLE_DOCS),
        output_path,
    )
    return output_path


if __name__ == "__main__":
    prepare_data()
