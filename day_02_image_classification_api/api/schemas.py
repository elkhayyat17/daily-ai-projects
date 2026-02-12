"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel, Field, HttpUrl


class PredictionFromURL(BaseModel):
    """Classify an image from a URL."""
    url: str = Field(
        ...,
        description="URL of the image to classify",
        examples=["https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"],
    )


class Top5Prediction(BaseModel):
    """A single top-5 prediction entry."""
    class_name: str = Field(..., alias="class")
    confidence: float = Field(..., ge=0, le=1)
    emoji: str = ""


class PredictionResponse(BaseModel):
    """Image classification response."""
    filename: str
    predicted_class: str
    confidence: float = Field(..., ge=0, le=1)
    emoji: str = ""
    top_5: list[Top5Prediction]

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "filename": "cat.jpg",
                    "predicted_class": "cat",
                    "confidence": 0.9723,
                    "emoji": "🐱",
                    "top_5": [
                        {"class": "cat", "confidence": 0.9723, "emoji": "🐱"},
                        {"class": "dog", "confidence": 0.0156, "emoji": "🐶"},
                    ],
                }
            ]
        },
    }


class ClassListResponse(BaseModel):
    """List of supported classes."""
    num_classes: int
    classes: list[dict]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str


class ModelInfoResponse(BaseModel):
    """Model information."""
    model_name: str
    dataset: str
    num_classes: int
    class_names: list[str]
    input_size: int
    device: str
    loaded: bool


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str = ""
