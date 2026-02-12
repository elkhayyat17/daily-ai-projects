"""
FastAPI Application Entry Point
Image Classification REST API
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from api.routes import router
from inference.predictor import get_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("🚀 Starting Image Classification API...")
    predictor = get_predictor()
    info = predictor.model_info
    logger.info(f"✅ Model: {info['model_name']} | Classes: {info['num_classes']} | Device: {info['device']}")
    yield
    logger.info("👋 Shutting down API...")


app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    """API root."""
    return {
        "message": "🖼️ Image Classification API",
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "classes": "/classes",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
