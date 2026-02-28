"""Build the Light RAG index — embeddings matrix + BM25 + metadata."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from config import get_settings
from data.prepare_data import prepare_data
from training.model import get_embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenizer for BM25."""
    return text.lower().split()


def _load_chunks(path: Path) -> List[dict]:
    chunks: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------------
# Index structure persisted to disk
# ---------------------------------------------------------------------------

class LightIndex:
    """Lightweight hybrid index: dense vectors + BM25."""

    def __init__(
        self,
        embeddings: np.ndarray,
        bm25: BM25Okapi,
        metadata: List[dict],
        texts: List[str],
    ) -> None:
        self.embeddings = embeddings      # (N, D) float32, L2-normalized
        self.bm25 = bm25                  # BM25Okapi instance
        self.metadata = metadata           # [{id, doc_id, title}, …]
        self.texts = texts                 # raw chunk texts

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(str(directory / "embeddings.npy"), self.embeddings)
        with (directory / "bm25.pkl").open("wb") as fh:
            pickle.dump(self.bm25, fh)
        with (directory / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, ensure_ascii=False)
        with (directory / "texts.json").open("w", encoding="utf-8") as fh:
            json.dump(self.texts, fh, ensure_ascii=False)
        logger.info("Index saved to {}", directory)

    @classmethod
    def load(cls, directory: Path) -> "LightIndex":
        embeddings = np.load(str(directory / "embeddings.npy"))
        with (directory / "bm25.pkl").open("rb") as fh:
            bm25 = pickle.load(fh)
        with (directory / "metadata.json").open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        with (directory / "texts.json").open("r", encoding="utf-8") as fh:
            texts = json.load(fh)
        return cls(embeddings=embeddings, bm25=bm25, metadata=metadata, texts=texts)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train() -> Path:
    """Build the Light RAG index and persist to disk."""
    settings = get_settings()

    # 1. Ensure processed chunks exist
    chunks_path = settings.processed_dir / "chunks.jsonl"
    if not chunks_path.exists():
        logger.warning("Chunks not found — running data preparation…")
        chunks_path = prepare_data()

    chunks = _load_chunks(chunks_path)
    texts = [c["content"] for c in chunks]
    metadata = [
        {"id": c["id"], "doc_id": c["doc_id"], "title": c["title"]}
        for c in chunks
    ]

    logger.info("Building index for {} chunks…", len(chunks))

    # 2. Compute dense embeddings
    embedder = get_embeddings()
    embeddings = embedder.encode(texts)
    logger.info("Embeddings shape: {}", embeddings.shape)

    # 3. Build BM25
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 index built with {} documents", len(tokenized))

    # 4. Persist
    index = LightIndex(
        embeddings=embeddings,
        bm25=bm25,
        metadata=metadata,
        texts=texts,
    )
    index.save(settings.index_dir)
    return settings.index_dir


if __name__ == "__main__":
    train()
