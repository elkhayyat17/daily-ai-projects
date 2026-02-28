"""Embedding model factory — lightweight, no LangChain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger

from config import get_settings


@dataclass
class EmbeddingConfig:
    provider: str
    local_model: str
    openai_api_key: str | None
    openai_model: str


class FakeEmbeddings:
    """Deterministic random embeddings for testing."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        rng = np.random.RandomState(42)
        vecs = rng.randn(len(texts), self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class OpenAIEmbeddings:
    """Thin wrapper around OpenAI embedding API."""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        client = self._get_client()
        resp = client.embeddings.create(input=texts, model=self.model)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class LocalEmbeddings:
    """sentence-transformers wrapper."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)


class EmbeddingFactory:
    """Creates the appropriate embedding backend."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    def create(self) -> FakeEmbeddings | OpenAIEmbeddings | LocalEmbeddings:
        if self.config.provider == "fake":
            logger.info("Using FAKE embeddings (test mode)")
            return FakeEmbeddings()

        if self.config.provider == "openai" and self.config.openai_api_key:
            logger.info("Using OpenAI embeddings: {}", self.config.openai_model)
            return OpenAIEmbeddings(
                model=self.config.openai_model,
                api_key=self.config.openai_api_key,
            )

        logger.info("Using local embeddings: {}", self.config.local_model)
        return LocalEmbeddings(self.config.local_model)


def get_embeddings() -> FakeEmbeddings | OpenAIEmbeddings | LocalEmbeddings:
    """Return an embedding model based on current settings."""
    settings = get_settings()
    config = EmbeddingConfig(
        provider=settings.embedding_provider,
        local_model=settings.local_embedding_model,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_embedding_model,
    )
    return EmbeddingFactory(config).create()
