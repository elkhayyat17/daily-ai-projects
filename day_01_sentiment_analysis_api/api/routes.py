"""
API Route Definitions
Defines all API endpoints for the sentiment analysis service.
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from inference.predictor import get_predictor
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
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
    info = predictor.model_info
    return ModelInfoResponse(**info)


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for a single text.
    
    Returns the predicted sentiment label (positive/negative/neutral),
    confidence score, and probability distribution.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(request.text)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Predicted '{result['sentiment']}' ({result['confidence']:.3f}) for text: {request.text[:50]}...")
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts (max 100).
    
    Efficiently processes texts in batches for better throughput.
    """
    try:
        predictor = get_predictor()
        results = predictor.predict_batch(request.texts)
        
        predictions = []
        for result in results:
            if "error" not in result:
                predictions.append(PredictionResponse(**result))
        
        logger.info(f"Batch prediction: {len(predictions)}/{len(request.texts)} successful")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
