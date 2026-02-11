"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single text prediction request."""
    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Text to analyze for sentiment",
        examples=["This product is amazing! Best purchase ever."],
    )


class BatchPredictionRequest(BaseModel):
    """Batch text prediction request."""
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to analyze (max 100)",
        examples=[["I love this!", "This is terrible.", "It's okay I guess."]],
    )


class SentimentProbabilities(BaseModel):
    """Probability distribution across sentiment classes."""
    positive: float = Field(..., ge=0, le=1)
    negative: float = Field(..., ge=0, le=1)
    neutral: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    """Single prediction response."""
    text: str
    sentiment: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: SentimentProbabilities

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "This product is amazing!",
                    "sentiment": "positive",
                    "confidence": 0.9847,
                    "probabilities": {
                        "positive": 0.9847,
                        "negative": 0.0089,
                        "neutral": 0.0064,
                    },
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    num_labels: int
    labels: list[str]
    max_length: int
    device: str
    loaded: bool


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str = ""
