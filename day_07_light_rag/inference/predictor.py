"""Light RAG predictor — hybrid retrieval (BM25 + cosine) with optional LLM generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from config import get_settings
from inference.preprocessing import clean_query
from training.model import get_embeddings
from training.train import LightIndex, _tokenize


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SourceDoc:
    id: str
    doc_id: str
    title: str
    score: float
    snippet: str


@dataclass
class Prediction:
    answer: str
    sources: List[SourceDoc]
    mode: str  # "hybrid" | "semantic" | "bm25"


# ---------------------------------------------------------------------------
# Predictor (singleton)
# ---------------------------------------------------------------------------

class LightRAGPredictor:
    """Production-ready Light RAG inference engine."""

    _instance: "LightRAGPredictor | None" = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = get_embeddings()
        self.index: LightIndex | None = None
        self._load_index()

    @classmethod
    def get_instance(cls) -> "LightRAGPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- Index management ---------------------------------------------------

    def _load_index(self) -> None:
        if not self.settings.index_dir.exists():
            logger.warning("Index not found at {}. Running in fallback mode.", self.settings.index_dir)
            return
        try:
            self.index = LightIndex.load(self.settings.index_dir)
            logger.info(
                "Loaded index: {} chunks, embedding dim={}",
                len(self.index.texts),
                self.index.embeddings.shape[1],
            )
        except Exception as exc:
            logger.error("Failed to load index: {}", exc)
            self.index = None

    @property
    def is_ready(self) -> bool:
        return self.index is not None

    # -- Retrieval ----------------------------------------------------------

    def _semantic_scores(self, query_vec: np.ndarray) -> np.ndarray:
        return (self.index.embeddings @ query_vec.T).flatten()

    def _bm25_scores(self, query: str) -> np.ndarray:
        return self.index.bm25.get_scores(_tokenize(query))

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo > 0:
            return (arr - lo) / (hi - lo)
        return np.zeros_like(arr)

    def _retrieve(
        self,
        query: str,
        top_k: int,
        mode: str = "hybrid",
    ) -> List[tuple[int, float]]:
        query_vec = self.embedder.encode([query])

        if mode == "semantic":
            scores = self._semantic_scores(query_vec)
        elif mode == "bm25":
            scores = self._bm25_scores(query).astype(np.float64)
        else:  # hybrid
            sem = self._minmax(self._semantic_scores(query_vec))
            bm = self._minmax(self._bm25_scores(query).astype(np.float64))
            scores = (
                self.settings.semantic_weight * sem
                + self.settings.bm25_weight * bm
            )

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

    # -- Context assembly ---------------------------------------------------

    def _build_context(self, ranked: List[tuple[int, float]]) -> str:
        parts: List[str] = []
        total = 0
        for idx, _score in ranked:
            snippet = self.index.texts[idx].strip()
            if total + len(snippet) > self.settings.max_context_chars:
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n---\n\n".join(parts)

    # -- Generation ---------------------------------------------------------

    def _generate_answer(self, query: str, context: str) -> str:
        """Use OpenAI if available, else return extractive context."""
        if (
            self.settings.embedding_provider == "openai"
            or self.settings.openai_api_key
        ) and self.settings.openai_api_key:
            try:
                import openai

                client = openai.OpenAI(api_key=self.settings.openai_api_key)
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful RAG assistant. Answer the user's "
                            "question based ONLY on the provided context. If the "
                            "context does not contain the answer, say 'I don't "
                            "know based on the available documents.'"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    },
                ]
                resp = client.chat.completions.create(
                    model=self.settings.openai_chat_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("OpenAI call failed, falling back: {}", exc)

        # Extractive fallback
        if not context:
            return (
                "The knowledge base is empty. "
                "Please ingest documents and run training/train.py."
            )
        return (
            "Based on the retrieved documents:\n\n"
            f"{context}\n\n"
            "(💡 Set OPENAI_API_KEY for AI-generated answers.)"
        )

    # -- Public API ---------------------------------------------------------

    def predict(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> Prediction:
        """Run the full Light RAG pipeline: retrieve → generate."""
        cleaned = clean_query(query)

        if self.index is None:
            return Prediction(
                answer="Index not available. Run training/train.py to build it.",
                sources=[],
                mode=mode,
            )

        k = top_k or self.settings.top_k
        ranked = self._retrieve(cleaned.text, k, mode=mode)

        sources = [
            SourceDoc(
                id=self.index.metadata[idx]["id"],
                doc_id=self.index.metadata[idx]["doc_id"],
                title=self.index.metadata[idx]["title"],
                score=round(score, 4),
                snippet=self.index.texts[idx][:200],
            )
            for idx, score in ranked
        ]

        context = self._build_context(ranked)
        answer = self._generate_answer(cleaned.text, context)
        return Prediction(answer=answer, sources=sources, mode=mode)

    def ingest(self, docs: List[dict]) -> int:
        """Add new documents to the live index (no disk persistence)."""
        if self.index is None:
            raise RuntimeError(
                "Index not loaded. Run training/train.py first."
            )

        new_texts = [d["content"] for d in docs]
        new_meta = [
            {
                "id": d["id"],
                "doc_id": d.get("doc_id", d["id"]),
                "title": d.get("title", ""),
            }
            for d in docs
        ]

        # Embed
        new_vecs = self.embedder.encode(new_texts)

        # Extend dense index
        self.index.embeddings = np.vstack([self.index.embeddings, new_vecs])
        self.index.texts.extend(new_texts)
        self.index.metadata.extend(new_meta)

        # Rebuild BM25 (fast for small corpora)
        tokenized = [_tokenize(t) for t in self.index.texts]
        self.index.bm25 = BM25Okapi(tokenized)

        logger.info("Ingested {} documents (total: {})", len(docs), len(self.index.texts))
        return len(docs)
