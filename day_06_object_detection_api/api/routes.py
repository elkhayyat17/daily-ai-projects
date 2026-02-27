"""
Day 06 — Object Detection API: API Routes
All FastAPI endpoint definitions for object detection.
"""

import io
import requests as http_requests
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, status
from fastapi.responses import Response
from loguru import logger

from api.schemas import (
    DetectionResponse,
    DetectionURLRequest,
    HealthResponse,
    ErrorResponse,
    ClassListResponse,
)
from inference.predictor import ObjectDetectionPredictor
from inference.preprocessing import ImagePreprocessor
import config

router = APIRouter()


def get_predictor() -> ObjectDetectionPredictor:
    """Get the singleton predictor instance."""
    return ObjectDetectionPredictor()


# ─── Health Check ────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check():
    """Check API health and model status."""
    predictor = get_predictor()
    st = predictor.status
    return HealthResponse(
        status="healthy" if predictor.is_ready else "degraded",
        model_loaded=st["model_loaded"],
        model_name=st["model_name"],
        num_classes=st["num_classes"],
        confidence_threshold=st["confidence_threshold"],
        iou_threshold=st["iou_threshold"],
        image_size=st["image_size"],
    )


# ─── Supported Classes ──────────────────────────────────────────────

@router.get(
    "/classes",
    response_model=ClassListResponse,
    tags=["System"],
    summary="List supported object classes",
)
async def list_classes():
    """Get the list of object classes the model can detect."""
    return ClassListResponse(
        num_classes=config.NUM_CLASSES,
        classes=config.COCO_CLASSES,
    )


# ─── Detect from Upload ─────────────────────────────────────────────

@router.post(
    "/detect",
    response_model=DetectionResponse,
    tags=["Detection"],
    summary="Detect objects in an uploaded image",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def detect_upload(
    file: UploadFile = File(..., description="Image file (JPG, PNG, BMP, WebP)"),
    confidence: float = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold"),
    iou_threshold: float = Query(0.45, ge=0.01, le=1.0, description="IoU threshold for NMS"),
    max_detections: int = Query(100, ge=1, le=300, description="Max detections"),
):
    """
    Upload an image and detect objects using YOLOv8.

    Returns bounding boxes, class names, and confidence scores for each detection.
    """
    try:
        # Read and validate image
        data = await file.read()
        filename = file.filename or "upload.jpg"
        image = ImagePreprocessor.load_image_from_bytes(data, filename)

        # Run detection
        predictor = get_predictor()
        result = predictor.detect(
            image=image,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )

        return DetectionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}",
        )


# ─── Detect and Annotate ────────────────────────────────────────────

@router.post(
    "/detect/annotate",
    tags=["Detection"],
    summary="Detect objects and return annotated image",
    responses={
        200: {"content": {"image/png": {}}},
        400: {"model": ErrorResponse},
    },
)
async def detect_and_annotate(
    file: UploadFile = File(..., description="Image file"),
    confidence: float = Query(0.25, ge=0.01, le=1.0),
    iou_threshold: float = Query(0.45, ge=0.01, le=1.0),
    max_detections: int = Query(100, ge=1, le=300),
):
    """
    Upload an image, detect objects, and return the annotated image with bounding boxes drawn.

    Returns a PNG image with bounding boxes and labels overlaid.
    """
    try:
        data = await file.read()
        filename = file.filename or "upload.jpg"
        image = ImagePreprocessor.load_image_from_bytes(data, filename)

        predictor = get_predictor()
        result, png_bytes = predictor.detect_and_annotate(
            image=image,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "X-Num-Detections": str(result["num_detections"]),
                "X-Elapsed-Ms": str(result["elapsed_ms"]),
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Annotate error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ─── Detect from URL ────────────────────────────────────────────────

@router.post(
    "/detect/url",
    response_model=DetectionResponse,
    tags=["Detection"],
    summary="Detect objects from an image URL",
    responses={400: {"model": ErrorResponse}},
)
async def detect_from_url(request: DetectionURLRequest):
    """
    Provide an image URL and detect objects in it.

    The server will download the image and run detection.
    """
    try:
        # Download image
        resp = http_requests.get(request.url, timeout=30, stream=True)
        resp.raise_for_status()

        data = resp.content
        if len(data) > config.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Image too large (max {config.MAX_IMAGE_SIZE_MB}MB)")

        image = ImagePreprocessor.load_image_from_bytes(data, request.url)

        # Run detection
        predictor = get_predictor()
        result = predictor.detect(
            image=image,
            confidence=request.confidence,
            iou_threshold=request.iou_threshold,
            max_detections=request.max_detections,
        )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )

        return DetectionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except http_requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download image: {str(e)}",
        )
    except Exception as e:
        logger.error(f"URL detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ─── Model Info ──────────────────────────────────────────────────────

@router.get(
    "/model/info",
    tags=["System"],
    summary="Get model information",
)
async def model_info():
    """Get detailed information about the loaded model."""
    predictor = get_predictor()
    return {
        "is_ready": predictor.is_ready,
        "status": predictor.status,
        "supported_formats": list(config.SUPPORTED_IMAGE_FORMATS),
        "max_image_size_mb": config.MAX_IMAGE_SIZE_MB,
        "default_confidence": config.CONFIDENCE_THRESHOLD,
        "default_iou": config.IOU_THRESHOLD,
    }
