"""
Day 05 — Document Q&A: FastAPI Application
Main application with lifespan management.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from api.routes import router
from inference.predictor import DocumentQAPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — load models on startup."""
    logger.info("🚀 Starting Document Q&A API...")
    logger.add(config.LOG_FILE, rotation="10 MB", level=config.LOG_LEVEL)

    # Initialize predictor and load models
    predictor = DocumentQAPredictor()
    try:
        predictor.load()
        if predictor.is_ready:
            logger.success(
                f"✅ System ready — {predictor.vector_store.num_vectors} vectors indexed"
            )
        else:
            logger.warning(
                "⚠️ No index found. Upload documents via /ingest or /upload."
            )
    except Exception as e:
        logger.warning(f"⚠️ Startup warning: {e}")

    yield

    logger.info("👋 Shutting down Document Q&A API...")


app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "📄 Document Q&A API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
