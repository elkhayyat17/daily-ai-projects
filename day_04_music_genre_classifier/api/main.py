"""
api/main.py
===========
FastAPI application factory for the Music Genre Classifier.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from api.routes import router
from config import get_settings
from inference.predictor import MusicPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting {} v{}", settings.project_name, settings.version)
    # Eagerly load predictor so the first request isn't slow
    predictor = MusicPredictor.get_instance()
    if predictor.is_ready:
        logger.success("Model loaded and ready ✓")
    else:
        logger.warning(
            "Model not found — serving in fallback mode. "
            "Run `python -m training.train` to train the model."
        )
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description=(
            "🎵 **Music Genre Classifier API**\n\n"
            "Upload an audio file (.wav, .mp3, .ogg, .flac, .m4a) and receive "
            "an ML-powered genre prediction across 10 genres: "
            "blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
