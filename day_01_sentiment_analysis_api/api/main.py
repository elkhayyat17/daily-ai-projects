"""
FastAPI Application Entry Point
Sentiment Analysis REST API
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
    logger.info("🚀 Starting Sentiment Analysis API...")
    
    # Pre-load the model
    predictor = get_predictor()
    logger.info(f"✅ Model loaded: {predictor.model_info['model_name']}")
    logger.info(f"🖥️  Device: {predictor.model_info['device']}")
    
    yield
    
    logger.info("👋 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "message": "🎯 Sentiment Analysis API",
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
