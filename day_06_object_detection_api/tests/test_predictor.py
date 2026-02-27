"""
Day 06 — Object Detection API: Unit Tests for Predictor & Core Components
"""

import io
import sys
import struct
import zlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.preprocessing import (
    ImagePreprocessor,
    validate_confidence_threshold,
    validate_iou_threshold,
)
from data.prepare_data import (
    prepare_sample_images,
    load_image_manifest,
    SAMPLE_IMAGES,
    _create_minimal_png,
)
import config


# ═══════════════════════════════════════════════════════════════════════
#  Helper: Create test images
# ═══════════════════════════════════════════════════════════════════════

def create_test_image(width=100, height=100, color=(255, 0, 0)) -> Image.Image:
    """Create a simple test image."""
    return Image.new("RGB", (width, height), color)


def create_test_image_bytes(width=100, height=100, fmt="PNG") -> bytes:
    """Create test image bytes."""
    img = create_test_image(width, height)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
#  ImagePreprocessor Tests
# ═══════════════════════════════════════════════════════════════════════

class TestImagePreprocessor:
    """Tests for the ImagePreprocessor class."""

    def test_load_image_from_bytes_png(self):
        """Should load a valid PNG image from bytes."""
        data = create_test_image_bytes(200, 150, "PNG")
        img = ImagePreprocessor.load_image_from_bytes(data, "test.png")
        assert img.size == (200, 150)
        assert img.mode == "RGB"

    def test_load_image_from_bytes_jpeg(self):
        """Should load a valid JPEG image from bytes."""
        data = create_test_image_bytes(200, 150, "JPEG")
        img = ImagePreprocessor.load_image_from_bytes(data, "test.jpg")
        assert img.size == (200, 150)

    def test_load_image_empty_bytes(self):
        """Should raise on empty bytes."""
        with pytest.raises(ValueError, match="Empty"):
            ImagePreprocessor.load_image_from_bytes(b"", "test.png")

    def test_load_image_invalid_bytes(self):
        """Should raise on invalid image data."""
        with pytest.raises(ValueError, match="Cannot read"):
            ImagePreprocessor.load_image_from_bytes(b"not an image", "test.png")

    def test_load_image_unsupported_extension(self):
        """Should raise on unsupported file extension."""
        data = create_test_image_bytes()
        with pytest.raises(ValueError, match="Unsupported"):
            ImagePreprocessor.load_image_from_bytes(data, "test.xyz")

    def test_load_image_too_large(self):
        """Should raise if image bytes exceed max size."""
        original_max = ImagePreprocessor.MAX_SIZE_MB
        ImagePreprocessor.MAX_SIZE_MB = 0.0001  # ~100 bytes
        try:
            data = create_test_image_bytes(200, 200)
            with pytest.raises(ValueError, match="too large"):
                ImagePreprocessor.load_image_from_bytes(data, "test.png")
        finally:
            ImagePreprocessor.MAX_SIZE_MB = original_max

    def test_validate_image_none(self):
        """Should raise on None image."""
        with pytest.raises(ValueError, match="None"):
            ImagePreprocessor.validate_image(None)

    def test_validate_image_valid(self):
        """Should not raise on valid image."""
        img = create_test_image()
        ImagePreprocessor.validate_image(img)  # Should not raise

    def test_validate_image_too_large_dimensions(self):
        """Should raise on oversized image dimensions."""
        img = create_test_image(5000, 5000)
        with pytest.raises(ValueError, match="too large"):
            ImagePreprocessor.validate_image(img)

    def test_load_image_from_path(self, tmp_path):
        """Should load image from file path."""
        img = create_test_image(320, 240)
        path = tmp_path / "test.png"
        img.save(path)

        loaded = ImagePreprocessor.load_image_from_path(path)
        assert loaded.size == (320, 240)
        assert loaded.mode == "RGB"

    def test_load_image_from_path_nonexistent(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            ImagePreprocessor.load_image_from_path("/nonexistent/image.png")

    def test_load_image_from_path_unsupported(self, tmp_path):
        """Should raise on unsupported extension."""
        path = tmp_path / "test.gif"
        path.write_bytes(b"GIF89a")
        with pytest.raises(ValueError, match="Unsupported"):
            ImagePreprocessor.load_image_from_path(path)

    def test_resize_if_needed_small_image(self):
        """Small images should not be resized."""
        img = create_test_image(100, 100)
        result = ImagePreprocessor.resize_if_needed(img, max_size=640)
        assert result.size == (100, 100)

    def test_resize_if_needed_large_image(self):
        """Large images should be resized."""
        img = create_test_image(1280, 960)
        result = ImagePreprocessor.resize_if_needed(img, max_size=640)
        assert max(result.size) <= 640

    def test_resize_maintains_aspect_ratio(self):
        """Resize should maintain aspect ratio."""
        img = create_test_image(800, 400)
        result = ImagePreprocessor.resize_if_needed(img, max_size=400)
        w, h = result.size
        assert abs(w / h - 2.0) < 0.1  # ~2:1 ratio

    def test_image_to_numpy(self):
        """Should convert PIL to numpy."""
        img = create_test_image(100, 80)
        arr = ImagePreprocessor.image_to_numpy(img)
        assert arr.shape == (80, 100, 3)
        assert arr.dtype == np.uint8

    def test_numpy_to_image(self):
        """Should convert numpy to PIL."""
        arr = np.zeros((80, 100, 3), dtype=np.uint8)
        img = ImagePreprocessor.numpy_to_image(arr)
        assert img.size == (100, 80)

    def test_rgba_to_rgb_conversion(self):
        """RGBA images should be converted to RGB."""
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()

        loaded = ImagePreprocessor.load_image_from_bytes(data, "test.png")
        assert loaded.mode == "RGB"


# ═══════════════════════════════════════════════════════════════════════
#  Threshold Validation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestThresholdValidation:
    """Tests for threshold validation functions."""

    def test_valid_confidence(self):
        assert validate_confidence_threshold(0.5) == 0.5

    def test_confidence_clamped_high(self):
        assert validate_confidence_threshold(1.5) == 1.0

    def test_confidence_clamped_low(self):
        assert validate_confidence_threshold(-0.5) == 0.01

    def test_confidence_invalid_type(self):
        with pytest.raises(ValueError, match="number"):
            validate_confidence_threshold("bad")

    def test_valid_iou(self):
        assert validate_iou_threshold(0.45) == 0.45

    def test_iou_clamped_high(self):
        assert validate_iou_threshold(2.0) == 1.0

    def test_iou_clamped_low(self):
        assert validate_iou_threshold(-1.0) == 0.01

    def test_iou_invalid_type(self):
        with pytest.raises(ValueError, match="number"):
            validate_iou_threshold("bad")


# ═══════════════════════════════════════════════════════════════════════
#  Data Preparation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDataPreparation:
    """Tests for data preparation functions."""

    def test_sample_images_not_empty(self):
        """Sample images list should not be empty."""
        assert len(SAMPLE_IMAGES) > 0

    def test_sample_images_have_required_fields(self):
        """Each sample image should have name, description, color, width, height."""
        for img in SAMPLE_IMAGES:
            assert "name" in img
            assert "description" in img
            assert "color" in img
            assert "width" in img
            assert "height" in img

    def test_create_minimal_png(self):
        """_create_minimal_png should create valid PNG bytes."""
        data = _create_minimal_png(10, 10, (255, 0, 0))
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        img = Image.open(io.BytesIO(data))
        assert img.size == (10, 10)

    def test_prepare_sample_images(self):
        """prepare_sample_images should create image files."""
        result = prepare_sample_images()
        assert result.exists()
        # Check that files were created
        for img_info in SAMPLE_IMAGES:
            img_path = config.RAW_DIR / img_info["name"]
            assert img_path.exists(), f"Missing: {img_path}"

    def test_load_image_manifest(self):
        """load_image_manifest should return list of dicts."""
        prepare_sample_images()
        manifest = load_image_manifest()
        assert len(manifest) == len(SAMPLE_IMAGES)
        for item in manifest:
            assert "filename" in item
            assert "width" in item
            assert "height" in item


# ═══════════════════════════════════════════════════════════════════════
#  Config Tests
# ═══════════════════════════════════════════════════════════════════════

class TestConfig:
    """Tests for configuration."""

    def test_paths_exist(self):
        """Config directories should be created."""
        assert config.DATA_DIR.exists()
        assert config.ARTIFACTS_DIR.exists()
        assert config.MODELS_DIR.exists()

    def test_coco_classes_count(self):
        """Should have 80 COCO classes."""
        assert config.NUM_CLASSES == 80
        assert len(config.COCO_CLASSES) == 80

    def test_coco_classes_common(self):
        """Should contain common COCO classes."""
        assert "person" in config.COCO_CLASSES
        assert "car" in config.COCO_CLASSES
        assert "dog" in config.COCO_CLASSES
        assert "cat" in config.COCO_CLASSES

    def test_confidence_threshold_range(self):
        """Confidence threshold should be valid."""
        assert 0 < config.CONFIDENCE_THRESHOLD < 1

    def test_iou_threshold_range(self):
        """IoU threshold should be valid."""
        assert 0 < config.IOU_THRESHOLD < 1

    def test_image_size_positive(self):
        """Image size should be positive."""
        assert config.IMAGE_SIZE > 0

    def test_supported_formats(self):
        """Should support common image formats."""
        assert ".jpg" in config.SUPPORTED_IMAGE_FORMATS
        assert ".png" in config.SUPPORTED_IMAGE_FORMATS
        assert ".jpeg" in config.SUPPORTED_IMAGE_FORMATS


# ═══════════════════════════════════════════════════════════════════════
#  Predictor Tests (with mock model)
# ═══════════════════════════════════════════════════════════════════════

class TestObjectDetectionPredictor:
    """Tests for the ObjectDetectionPredictor (using mocked YOLO model)."""

    def setup_method(self):
        """Reset singleton before each test."""
        from inference.predictor import ObjectDetectionPredictor
        ObjectDetectionPredictor.reset()

    def test_singleton_pattern(self):
        """Should return same instance."""
        from inference.predictor import ObjectDetectionPredictor
        p1 = ObjectDetectionPredictor()
        p2 = ObjectDetectionPredictor()
        assert p1 is p2

    def test_initial_state(self):
        """Should start in not-ready state."""
        from inference.predictor import ObjectDetectionPredictor
        pred = ObjectDetectionPredictor()
        assert not pred.is_ready

    def test_status_keys(self):
        """Status should have expected keys."""
        from inference.predictor import ObjectDetectionPredictor
        pred = ObjectDetectionPredictor()
        status = pred.status
        assert "model_loaded" in status
        assert "model_name" in status
        assert "num_classes" in status

    def test_detect_without_model(self):
        """Detection should return graceful error when model not loaded."""
        from inference.predictor import ObjectDetectionPredictor
        pred = ObjectDetectionPredictor()
        img = create_test_image()
        result = pred.detect(img)
        assert result["num_detections"] == 0
        assert result["model_loaded"] is False

    def test_reset(self):
        """Reset should clear singleton."""
        from inference.predictor import ObjectDetectionPredictor
        p1 = ObjectDetectionPredictor()
        ObjectDetectionPredictor.reset()
        p2 = ObjectDetectionPredictor()
        assert p1 is not p2
