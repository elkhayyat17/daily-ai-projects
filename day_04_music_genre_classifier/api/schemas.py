"""
api/schemas.py
==============
Pydantic request / response models for the Music Genre Classifier API.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class GenrePredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    genre: str = Field(..., description="Predicted music genre")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top-1 confidence score")
    probabilities: Dict[str, float] = Field(
        ..., description="Probability distribution over all genres"
    )
    duration_seconds: float = Field(..., description="Analysed audio clip duration")
    model_ready: bool = Field(..., description="False if model has not been trained yet")


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    model_ready: bool
    version: str
    genres: list[str]


class ReloadResponse(BaseModel):
    """Response from the /reload endpoint."""

    success: bool
    model_ready: bool
    message: str
