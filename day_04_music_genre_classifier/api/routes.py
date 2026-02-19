"""
api/routes.py
=============
FastAPI route handlers for the Music Genre Classifier.

Endpoints
---------
GET  /health          — service health & model status
POST /predict         — upload an audio file, receive genre prediction
POST /reload          — hot-reload the model from disk (after re-training)
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from api.schemas import GenrePredictionResponse, HealthResponse, ReloadResponse
from config import get_settings
from inference.predictor import MusicPredictor
from inference.preprocessing import AudioValidationError

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check() -> HealthResponse:
    """Return service health and model status."""
    settings = get_settings()
    predictor = MusicPredictor.get_instance()
    return HealthResponse(
        status="ok",
        model_ready=predictor.is_ready,
        version=settings.version,
        genres=list(settings.genres),
    )


@router.post(
    "/predict",
    response_model=GenrePredictionResponse,
    tags=["Prediction"],
    summary="Classify the genre of an uploaded audio file",
)
async def predict(
    file: UploadFile = File(
        ...,
        description="Audio file to classify (.wav, .mp3, .ogg, .flac, .m4a)",
    ),
) -> GenrePredictionResponse:
    """
    Upload an audio file and get a music genre prediction.

    - Accepted formats: WAV, MP3, OGG, FLAC, M4A
    - Maximum file size: configurable (default 50 MB)
    - Returns genre label, confidence score, and full probability distribution
    """
    if file.filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    logger.info("Received upload: {} ({} bytes)", file.filename, file.size or "?")

    try:
        data = await file.read()
        predictor = MusicPredictor.get_instance()
        result = predictor.predict_from_bytes(data, filename=file.filename)
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {exc}",
        ) from exc

    return GenrePredictionResponse(
        genre=result.genre,
        confidence=result.confidence,
        probabilities=result.probabilities,
        duration_seconds=result.duration_seconds,
        model_ready=result.model_ready,
    )


@router.post("/reload", response_model=ReloadResponse, tags=["Utility"])
def reload_model() -> ReloadResponse:
    """
    Hot-reload the model from disk.

    Call this endpoint after running `training/train.py` to pick up
    a freshly trained model without restarting the server.
    """
    predictor = MusicPredictor.get_instance()
    success = predictor.reload()
    return ReloadResponse(
        success=success,
        model_ready=success,
        message="Model reloaded successfully." if success else "Model file not found.",
    )
