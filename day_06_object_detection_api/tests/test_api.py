"""
Day 06 — Object Detection API: API Integration Tests
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app
from inference.predictor import ObjectDetectionPredictor


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_predictor():
    """Reset the singleton predictor before each test."""
    ObjectDetectionPredictor.reset()
    yield
    ObjectDetectionPredictor.reset()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def _create_test_image_bytes(width=200, height=150, fmt="PNG") -> bytes:
    """Create test image bytes."""
    img = Image.new("RGB", (width, height), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
#  Root & Health Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root(self, client):
        """GET / should return welcome message."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "Object Detection" in data["message"]

    def test_root_has_docs_link(self, client):
        """Root should include docs link."""
        resp = client.get("/")
        data = resp.json()
        assert "docs" in data

    def test_root_has_health_link(self, client):
        """Root should include health link."""
        resp = client.get("/")
        data = resp.json()
        assert "health" in data


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health(self, client):
        """GET /api/v1/health should return status."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data
        assert "num_classes" in data

    def test_health_has_thresholds(self, client):
        """Health should include threshold info."""
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "confidence_threshold" in data
        assert "iou_threshold" in data

    def test_health_num_classes(self, client):
        """Health should report correct number of classes."""
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert data["num_classes"] == 80


# ═══════════════════════════════════════════════════════════════════════
#  Classes Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════

class TestClassesEndpoint:
    """Tests for the /classes endpoint."""

    def test_classes(self, client):
        """GET /api/v1/classes should return class list."""
        resp = client.get("/api/v1/classes")
        assert resp.status_code == 200
        data = resp.json()
        assert "num_classes" in data
        assert "classes" in data
        assert data["num_classes"] == 80
        assert len(data["classes"]) == 80

    def test_classes_contain_common_objects(self, client):
        """Class list should contain common objects."""
        resp = client.get("/api/v1/classes")
        classes = resp.json()["classes"]
        assert "person" in classes
        assert "car" in classes
        assert "dog" in classes


# ═══════════════════════════════════════════════════════════════════════
#  Detect Upload Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDetectUploadEndpoint:
    """Tests for the /detect endpoint."""

    def test_detect_png(self, client):
        """Should accept a PNG image."""
        data = _create_test_image_bytes(200, 150, "PNG")
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.png", data, "image/png")},
        )
        assert resp.status_code == 200
        result = resp.json()
        assert "detections" in result
        assert "num_detections" in result
        assert "elapsed_ms" in result

    def test_detect_jpeg(self, client):
        """Should accept a JPEG image."""
        data = _create_test_image_bytes(200, 150, "JPEG")
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_detect_with_custom_confidence(self, client):
        """Should accept custom confidence threshold."""
        data = _create_test_image_bytes()
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.png", data, "image/png")},
            params={"confidence": 0.5},
        )
        assert resp.status_code == 200

    def test_detect_with_custom_iou(self, client):
        """Should accept custom IoU threshold."""
        data = _create_test_image_bytes()
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.png", data, "image/png")},
            params={"iou_threshold": 0.7},
        )
        assert resp.status_code == 200

    def test_detect_invalid_confidence(self, client):
        """Should reject confidence > 1.0."""
        data = _create_test_image_bytes()
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.png", data, "image/png")},
            params={"confidence": 1.5},
        )
        assert resp.status_code == 422

    def test_detect_invalid_max_detections(self, client):
        """Should reject max_detections > 300."""
        data = _create_test_image_bytes()
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.png", data, "image/png")},
            params={"max_detections": 500},
        )
        assert resp.status_code == 422

    def test_detect_unsupported_format(self, client):
        """Should reject unsupported file formats."""
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.exe", b"not an image", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_detect_response_schema(self, client):
        """Response should have all required fields."""
        data = _create_test_image_bytes()
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.png", data, "image/png")},
        )
        assert resp.status_code == 200
        result = resp.json()
        for field in [
            "detections", "num_detections", "class_counts",
            "image_size", "elapsed_ms", "confidence_threshold", "iou_threshold",
        ]:
            assert field in result, f"Missing field: {field}"


# ═══════════════════════════════════════════════════════════════════════
#  Detect and Annotate Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDetectAnnotateEndpoint:
    """Tests for the /detect/annotate endpoint."""

    def test_annotate_returns_png(self, client):
        """Should return a PNG image."""
        data = _create_test_image_bytes(200, 150)
        resp = client.post(
            "/api/v1/detect/annotate",
            files={"file": ("test.png", data, "image/png")},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        # Should be valid image
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "PNG"

    def test_annotate_has_metadata_headers(self, client):
        """Response should include detection metadata in headers."""
        data = _create_test_image_bytes()
        resp = client.post(
            "/api/v1/detect/annotate",
            files={"file": ("test.png", data, "image/png")},
        )
        assert resp.status_code == 200
        assert "x-num-detections" in resp.headers
        assert "x-elapsed-ms" in resp.headers


# ═══════════════════════════════════════════════════════════════════════
#  Detect from URL Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDetectURLEndpoint:
    """Tests for the /detect/url endpoint."""

    def test_detect_url_missing_url(self, client):
        """Should fail when URL is missing."""
        resp = client.post("/api/v1/detect/url", json={})
        assert resp.status_code == 422

    def test_detect_url_short_url(self, client):
        """Should fail with too-short URL."""
        resp = client.post(
            "/api/v1/detect/url",
            json={"url": "http://x"},
        )
        assert resp.status_code == 422

    def test_detect_url_invalid_url(self, client):
        """Should fail on unreachable URL."""
        resp = client.post(
            "/api/v1/detect/url",
            json={"url": "http://localhost:99999/nonexistent.jpg"},
        )
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════
#  Model Info Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════

class TestModelInfoEndpoint:
    """Tests for the /model/info endpoint."""

    def test_model_info(self, client):
        """GET /api/v1/model/info should return model metadata."""
        resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_ready" in data
        assert "status" in data
        assert "supported_formats" in data
        assert "max_image_size_mb" in data

    def test_model_info_formats(self, client):
        """Should list supported formats."""
        resp = client.get("/api/v1/model/info")
        formats = resp.json()["supported_formats"]
        assert ".jpg" in formats
        assert ".png" in formats


# ═══════════════════════════════════════════════════════════════════════
#  OpenAPI Docs Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDocs:
    """Tests for API documentation."""

    def test_openapi_json(self, client):
        """OpenAPI spec should be accessible."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        assert spec["info"]["title"] == "Object Detection API"
        assert "paths" in spec

    def test_docs_page(self, client):
        """Swagger UI should be accessible."""
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_page(self, client):
        """ReDoc should be accessible."""
        resp = client.get("/redoc")
        assert resp.status_code == 200
