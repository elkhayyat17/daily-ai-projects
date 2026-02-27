"""
Day 06 — Object Detection API: Pydantic Request/Response Schemas
"""

from pydantic import BaseModel, Field


# ─── Request Models ──────────────────────────────────────────────────

class DetectionRequest(BaseModel):
    """Request parameters for object detection (used with query params)."""
    confidence: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Minimum confidence threshold for detections.",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.01,
        le=1.0,
        description="IoU threshold for Non-Maximum Suppression.",
    )
    max_detections: int = Field(
        default=100,
        ge=1,
        le=300,
        description="Maximum number of detections to return.",
    )
    annotate: bool = Field(
        default=False,
        description="Whether to return an annotated image.",
    )


class DetectionURLRequest(BaseModel):
    """Request body for detecting objects from an image URL."""
    url: str = Field(
        ...,
        min_length=10,
        max_length=2048,
        description="URL of the image to analyze.",
        examples=["https://example.com/photo.jpg"],
    )
    confidence: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Minimum confidence threshold.",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.01,
        le=1.0,
        description="IoU threshold for NMS.",
    )
    max_detections: int = Field(
        default=100,
        ge=1,
        le=300,
        description="Maximum number of detections.",
    )


# ─── Response Models ────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float = Field(description="Top-left X coordinate (pixels).")
    y1: float = Field(description="Top-left Y coordinate (pixels).")
    x2: float = Field(description="Bottom-right X coordinate (pixels).")
    y2: float = Field(description="Bottom-right Y coordinate (pixels).")


class BoundingBoxNormalized(BaseModel):
    """Normalized bounding box coordinates (0-1)."""
    x1: float = Field(description="Top-left X (normalized 0-1).")
    y1: float = Field(description="Top-left Y (normalized 0-1).")
    x2: float = Field(description="Bottom-right X (normalized 0-1).")
    y2: float = Field(description="Bottom-right Y (normalized 0-1).")


class Detection(BaseModel):
    """A single detected object."""
    class_id: int = Field(description="Class ID from the model.")
    class_name: str = Field(description="Human-readable class name.")
    confidence: float = Field(description="Detection confidence (0-1).")
    bbox: BoundingBox = Field(description="Bounding box in pixels.")
    bbox_normalized: BoundingBoxNormalized = Field(
        description="Bounding box normalized to image dimensions."
    )
    area: float = Field(description="Bounding box area in pixels².")


class ImageSize(BaseModel):
    """Image dimensions."""
    width: int
    height: int


class DetectionResponse(BaseModel):
    """Response from object detection."""
    detections: list[Detection] = Field(description="List of detected objects.")
    num_detections: int = Field(description="Total number of detections.")
    class_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each detected class.",
    )
    image_size: ImageSize = Field(description="Original image dimensions.")
    elapsed_ms: float = Field(description="Processing time in milliseconds.")
    confidence_threshold: float = Field(description="Applied confidence threshold.")
    iou_threshold: float = Field(description="Applied IoU threshold.")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status: healthy or degraded.")
    model_loaded: bool
    model_name: str
    num_classes: int
    confidence_threshold: float
    iou_threshold: float
    image_size: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str = ""


class ClassListResponse(BaseModel):
    """Response with supported class names."""
    num_classes: int
    classes: list[str]
