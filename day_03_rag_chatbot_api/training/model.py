from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from config import get_settings


@dataclass
class EmbeddingConfig:
    provider: str
    openai_api_key: str | None
    openai_embedding_model: str
    local_embedding_model: str


class EmbeddingFactory:
    """Factory for LangChain embeddings."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    def create(self):
        if self.config.provider == "fake":
            from langchain_community.embeddings import FakeEmbeddings

            logger.info("Using fake embeddings for tests")
            return FakeEmbeddings(size=384)

        if self.config.provider == "openai" and self.config.openai_api_key:
            from langchain_openai import OpenAIEmbeddings

            logger.info("Using OpenAI embeddings: {}", self.config.openai_embedding_model)
            return OpenAIEmbeddings(
                model=self.config.openai_embedding_model,
                api_key=self.config.openai_api_key,
            )

        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Using local embeddings: {}", self.config.local_embedding_model)
        return HuggingFaceEmbeddings(model_name=self.config.local_embedding_model)


def get_embeddings():
    settings = get_settings()
    config = EmbeddingConfig(
        provider=settings.embedding_provider,
        openai_api_key=settings.openai_api_key,
        openai_embedding_model=settings.openai_embedding_model,
        local_embedding_model=settings.local_embedding_model,
    )
    return EmbeddingFactory(config).create()
