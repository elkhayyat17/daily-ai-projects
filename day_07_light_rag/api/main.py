"""FastAPI application — Light RAG."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from api.routes import router
from config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("🚀 Starting {} v{}", settings.project_name, settings.version)
    yield
    logger.info("👋 Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description=(
            "Lightweight Retrieval-Augmented Generation — hybrid BM25 + "
            "cosine search with NumPy, no external vector database."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
