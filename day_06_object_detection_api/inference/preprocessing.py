"""
Day 06 — Object Detection API: Input Preprocessing
Image validation, loading, and transformation utilities.
"""

import io
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class ImagePreprocessor:
    """
    Validates, loads, and preprocesses images for YOLOv8 inference.
    """

    SUPPORTED_FORMATS = config.SUPPORTED_IMAGE_FORMATS
    MAX_SIZE_MB = config.MAX_IMAGE_SIZE_MB
    MAX_DIMENSION = config.MAX_IMAGE_DIMENSION

    @classmethod
    def validate_image_bytes(cls, data: bytes, filename: str = "unknown") -> None:
        """
        Validate raw image bytes before processing.

        Args:
            data: Raw image bytes.
            filename: Original filename for extension checking.

        Raises:
            ValueError: If validation fails.
        """
        if not data:
            raise ValueError("Empty image data.")

        # Check file size
        size_mb = len(data) / (1024 * 1024)
        if size_mb > cls.MAX_SIZE_MB:
            raise ValueError(
                f"Image too large: {size_mb:.1f}MB (max {cls.MAX_SIZE_MB}MB)"
            )

        # Check extension
        ext = Path(filename).suffix.lower()
        if ext and ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {ext}. Supported: {cls.SUPPORTED_FORMATS}"
            )

    @classmethod
    def validate_image(cls, image: Image.Image) -> None:
        """
        Validate a PIL Image.

        Args:
            image: PIL Image to validate.

        Raises:
            ValueError: If validation fails.
        """
        if image is None:
            raise ValueError("Image is None.")

        w, h = image.size
        if w == 0 or h == 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")

        if w > cls.MAX_DIMENSION or h > cls.MAX_DIMENSION:
            raise ValueError(
                f"Image too large: {w}x{h} (max {cls.MAX_DIMENSION}x{cls.MAX_DIMENSION})"
            )

    @classmethod
    def load_image_from_bytes(cls, data: bytes, filename: str = "upload.jpg") -> Image.Image:
        """
        Load and validate an image from raw bytes.

        Args:
            data: Raw image bytes.
            filename: Original filename.

        Returns:
            PIL Image in RGB mode.
        """
        cls.validate_image_bytes(data, filename)

        try:
            image = Image.open(io.BytesIO(data))
            image.load()  # Force load to catch corrupt files
        except Exception as e:
            raise ValueError(f"Cannot read image: {e}")

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        cls.validate_image(image)
        return image

    @classmethod
    def load_image_from_path(cls, path: Union[str, Path]) -> Image.Image:
        """
        Load and validate an image from a file path.

        Args:
            path: Path to the image file.

        Returns:
            PIL Image in RGB mode.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {ext}. Supported: {cls.SUPPORTED_FORMATS}"
            )

        try:
            image = Image.open(path)
            image.load()
        except Exception as e:
            raise ValueError(f"Cannot read image {path}: {e}")

        if image.mode != "RGB":
            image = image.convert("RGB")

        cls.validate_image(image)
        return image

    @classmethod
    def resize_if_needed(
        cls,
        image: Image.Image,
        max_size: int = config.IMAGE_SIZE,
    ) -> Image.Image:
        """
        Resize image if it exceeds the maximum size while maintaining aspect ratio.

        Args:
            image: PIL Image.
            max_size: Maximum dimension (width or height).

        Returns:
            Resized image (or original if already small enough).
        """
        w, h = image.size
        if max(w, h) <= max_size:
            return image

        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), Image.LANCZOS)

    @classmethod
    def image_to_numpy(cls, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array (H, W, C) in RGB."""
        return np.array(image)

    @classmethod
    def numpy_to_image(cls, array: np.ndarray) -> Image.Image:
        """Convert numpy array (H, W, C) to PIL Image."""
        return Image.fromarray(array.astype(np.uint8))


def validate_confidence_threshold(conf: float) -> float:
    """Validate and clamp confidence threshold."""
    if not isinstance(conf, (int, float)):
        raise ValueError("Confidence threshold must be a number.")
    return max(0.01, min(1.0, float(conf)))


def validate_iou_threshold(iou: float) -> float:
    """Validate and clamp IoU threshold."""
    if not isinstance(iou, (int, float)):
        raise ValueError("IoU threshold must be a number.")
    return max(0.01, min(1.0, float(iou)))
