"""
API Integration Tests
Tests for FastAPI image classification endpoints.
"""

import sys
import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.main import app

client = TestClient(app)


def _create_test_image_bytes(fmt="JPEG") -> bytes:
    """Create a test image as bytes."""
    img = Image.new("RGB", (64, 64), (255, 128, 0))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


class TestHealthEndpoint:
    """Tests for health check."""

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


class TestClassesEndpoint:
    """Tests for class listing."""

    def test_list_classes(self):
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert data["num_classes"] == 10
        assert len(data["classes"]) == 10

    def test_class_structure(self):
        response = client.get("/classes")
        data = response.json()
        first_class = data["classes"][0]
        assert "index" in first_class
        assert "name" in first_class
        assert "emoji" in first_class


class TestPredictEndpoint:
    """Tests for image prediction."""

    def test_predict_jpeg(self):
        image_bytes = _create_test_image_bytes("JPEG")
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "top_5" in data

    def test_predict_png(self):
        image_bytes = _create_test_image_bytes("PNG")
        response = client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_predict_response_structure(self):
        image_bytes = _create_test_image_bytes("JPEG")
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        data = response.json()

        # Required fields
        assert "filename" in data
        assert "predicted_class" in data
        assert "confidence" in data
        assert "top_5" in data

        # Confidence range
        assert 0 <= data["confidence"] <= 1

        # Top-5 structure
        assert len(data["top_5"]) == 5
        for pred in data["top_5"]:
            assert "class" in pred
            assert "confidence" in pred

    def test_predict_invalid_file(self):
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code in [400, 500]


class TestModelInfoEndpoint:
    """Tests for model info."""

    def test_model_info(self):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["num_classes"] == 10
        assert data["model_name"] == "resnet50"
        assert "class_names" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
