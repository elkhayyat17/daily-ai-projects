"""
API Route Definitions
Image classification endpoints.
"""

import sys
import requests
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from inference.predictor import get_predictor
from api.schemas import (
    PredictionFromURL,
    PredictionResponse,
    Top5Prediction,
    ClassListResponse,
    HealthResponse,
    ModelInfoResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    return HealthResponse(status="healthy", version=config.API_VERSION)


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model metadata and configuration."""
    predictor = get_predictor()
    return ModelInfoResponse(**predictor.model_info)


@router.get("/classes", response_model=ClassListResponse, tags=["Model"])
async def list_classes():
    """List all supported classification classes."""
    classes = [
        {
            "index": i,
            "name": name,
            "emoji": config.CLASS_EMOJIS.get(name, ""),
        }
        for i, name in enumerate(config.CLASS_NAMES)
    ]
    return ClassListResponse(num_classes=config.NUM_CLASSES, classes=classes)


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_image(file: UploadFile = File(..., description="Image file to classify")):
    """
    Classify an uploaded image.

    Accepts JPEG, PNG, BMP, WEBP formats (max 10MB).
    Returns predicted class with confidence and top-5 predictions.
    """
    try:
        # Read file
        image_bytes = await file.read()
        filename = file.filename or "upload"

        # Predict
        predictor = get_predictor()
        result = predictor.predict_from_bytes(image_bytes, filename)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        logger.info(
            f"🖼️  Classified '{filename}' → {result['emoji']} {result['predicted_class']} "
            f"({result['confidence']:.3f})"
        )

        return PredictionResponse(
            filename=result["filename"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            emoji=result["emoji"],
            top_5=[
                Top5Prediction(**{
                    "class": p["class"],
                    "confidence": p["confidence"],
                    "emoji": p.get("emoji", ""),
                })
                for p in result["top_5"]
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/url", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_url(request: PredictionFromURL):
    """
    Classify an image from a URL.

    Downloads the image and returns classification results.
    """
    try:
        # Download image
        response = requests.get(request.url, timeout=10, stream=True)
        response.raise_for_status()

        content_length = int(response.headers.get("content-length", 0))
        if content_length > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large")

        image_bytes = response.content
        filename = request.url.split("/")[-1].split("?")[0] or "url_image"

        # Predict
        predictor = get_predictor()
        result = predictor.predict_from_bytes(image_bytes, filename)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return PredictionResponse(
            filename=result["filename"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            emoji=result["emoji"],
            top_5=[
                Top5Prediction(**{
                    "class": p["class"],
                    "confidence": p["confidence"],
                    "emoji": p.get("emoji", ""),
                })
                for p in result["top_5"]
            ],
        )

    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        logger.error(f"URL prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
