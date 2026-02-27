"""
Day 06 — Object Detection API: FastAPI Application
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
from inference.predictor import ObjectDetectionPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — load model on startup."""
    logger.info("🚀 Starting Object Detection API...")
    logger.add(config.LOG_FILE, rotation="10 MB", level=config.LOG_LEVEL)

    # Initialize predictor and load model
    predictor = ObjectDetectionPredictor()
    try:
        success = predictor.load()
        if success:
            logger.success(
                f"✅ Model ready — {config.NUM_CLASSES} classes, "
                f"conf={config.CONFIDENCE_THRESHOLD}"
            )
        else:
            logger.warning(
                "⚠️ Model not loaded. Detection will not be available."
            )
    except Exception as e:
        logger.warning(f"⚠️ Startup warning: {e}")

    yield

    logger.info("👋 Shutting down Object Detection API...")


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
        "message": "🔍 Object Detection API — YOLOv8",
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
