"""
Image Preprocessing Utilities
Handles image loading, validation, and preparation for inference.
"""

import io
from PIL import Image
from loguru import logger


SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP", "GIF", "TIFF"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load an image from raw bytes.

    Args:
        image_bytes: Raw image file bytes

    Returns:
        PIL Image in RGB mode

    Raises:
        ValueError: If image cannot be loaded or is invalid
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")


def load_image_from_path(image_path: str) -> Image.Image:
    """
    Load an image from a file path.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image in RGB mode
    """
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")


def validate_image(image_bytes: bytes, filename: str = "unknown") -> tuple[bool, str]:
    """
    Validate uploaded image file.

    Args:
        image_bytes: Raw image bytes
        filename: Original filename

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file size
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(image_bytes) / (1024 * 1024)
        return False, f"File too large: {size_mb:.1f}MB (max: {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f}MB)"

    if len(image_bytes) < 100:
        return False, "File too small to be a valid image"

    # Try loading the image
    try:
        image = Image.open(io.BytesIO(image_bytes))

        # Check format
        if image.format and image.format.upper() not in SUPPORTED_FORMATS:
            return False, f"Unsupported format: {image.format}. Supported: {', '.join(SUPPORTED_FORMATS)}"

        # Check dimensions
        width, height = image.size
        if width < 10 or height < 10:
            return False, f"Image too small: {width}x{height} (min: 10x10)"

        if width > 10000 or height > 10000:
            return False, f"Image too large: {width}x{height} (max: 10000x10000)"

        return True, ""

    except Exception as e:
        return False, f"Invalid image file: {e}"


def get_image_info(image: Image.Image) -> dict:
    """
    Get metadata about an image.

    Args:
        image: PIL Image

    Returns:
        Dictionary with image metadata
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": getattr(image, "format", "unknown"),
    }
