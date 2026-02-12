"""
Unit Tests for Image Predictor
"""

import sys
import io
import numpy as np
from pathlib import Path
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _create_test_image(width=64, height=64, color=(255, 0, 0)) -> bytes:
    """Create a simple test image and return as bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_test_pil_image(width=64, height=64) -> Image.Image:
    """Create a simple test PIL image."""
    return Image.new("RGB", (width, height), (0, 128, 255))


class TestImagePreprocessing:
    """Tests for image preprocessing utilities."""

    def test_load_image_from_bytes(self):
        from inference.preprocessing import load_image_from_bytes

        image_bytes = _create_test_image()
        image = load_image_from_bytes(image_bytes)
        assert image.mode == "RGB"
        assert image.size == (64, 64)

    def test_load_invalid_bytes(self):
        from inference.preprocessing import load_image_from_bytes

        with pytest.raises(ValueError):
            load_image_from_bytes(b"not an image")

    def test_validate_valid_image(self):
        from inference.preprocessing import validate_image

        image_bytes = _create_test_image()
        is_valid, msg = validate_image(image_bytes, "test.jpg")
        assert is_valid is True
        assert msg == ""

    def test_validate_too_small(self):
        from inference.preprocessing import validate_image

        is_valid, msg = validate_image(b"tiny", "test.jpg")
        assert is_valid is False

    def test_validate_too_large(self):
        from inference.preprocessing import validate_image

        huge_bytes = b"x" * (11 * 1024 * 1024)  # 11 MB
        is_valid, msg = validate_image(huge_bytes, "test.jpg")
        assert is_valid is False
        assert "large" in msg.lower()

    def test_get_image_info(self):
        from inference.preprocessing import get_image_info

        img = _create_test_pil_image(100, 200)
        info = get_image_info(img)
        assert info["width"] == 100
        assert info["height"] == 200
        assert info["mode"] == "RGB"


class TestImagePredictor:
    """Tests for the ImagePredictor class."""

    def test_predictor_init(self):
        from inference.predictor import ImagePredictor

        predictor = ImagePredictor()
        assert predictor._loaded is False

    def test_model_info(self):
        from inference.predictor import ImagePredictor

        predictor = ImagePredictor()
        info = predictor.model_info
        assert info["num_classes"] == 10
        assert "class_names" in info
        assert len(info["class_names"]) == 10

    def test_predict_from_bytes(self):
        from inference.predictor import ImagePredictor

        predictor = ImagePredictor()
        predictor.load()

        image_bytes = _create_test_image()
        result = predictor.predict_from_bytes(image_bytes, "test.jpg")

        assert "predicted_class" in result
        assert result["predicted_class"] in [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
        assert "top_5" in result
        assert len(result["top_5"]) == 5

    def test_predict_from_pil(self):
        from inference.predictor import ImagePredictor

        predictor = ImagePredictor()
        predictor.load()

        image = _create_test_pil_image()
        result = predictor.predict_from_pil(image, "test.jpg")

        assert "predicted_class" in result
        assert "confidence" in result

    def test_predict_invalid_bytes(self):
        from inference.predictor import ImagePredictor

        predictor = ImagePredictor()
        predictor.load()

        result = predictor.predict_from_bytes(b"not an image", "bad.jpg")
        assert "error" in result

    def test_top5_probabilities_sum(self):
        from inference.predictor import ImagePredictor

        predictor = ImagePredictor()
        predictor.load()

        image_bytes = _create_test_image()
        result = predictor.predict_from_bytes(image_bytes, "test.jpg")

        total = sum(result["all_probabilities"].values())
        assert abs(total - 1.0) < 0.01


class TestTransforms:
    """Tests for data transforms."""

    def test_train_transforms(self):
        from training.transforms import get_train_transforms

        transform = get_train_transforms()
        img = _create_test_pil_image(32, 32)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transforms(self):
        from training.transforms import get_val_transforms

        transform = get_val_transforms()
        img = _create_test_pil_image(32, 32)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_inference_transforms(self):
        from training.transforms import get_inference_transforms

        transform = get_inference_transforms()
        img = _create_test_pil_image(500, 300)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
