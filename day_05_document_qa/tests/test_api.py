"""
Day 05 — Document Q&A: API Integration Tests
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app
from inference.predictor import DocumentQAPredictor


@pytest.fixture(autouse=True)
def reset_predictor():
    """Reset the singleton predictor before each test."""
    DocumentQAPredictor.reset()
    yield
    DocumentQAPredictor.reset()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════════
#  Root & Health Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """GET / should return welcome message."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "Document Q&A" in data["message"]

    def test_root_has_docs_link(self, client):
        resp = client.get("/")
        assert "docs" in resp.json()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health(self, client):
        """GET /api/v1/health should return status."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "num_vectors" in data
        assert "embedding_model" in data

    def test_health_has_model_info(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "qa_model_loaded" in data
        assert "index_loaded" in data


# ═══════════════════════════════════════════════════════════════════════
#  Ask Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAskEndpoint:
    """Tests for the /ask endpoint."""

    def test_ask_without_index(self, client):
        """Ask should return graceful response when no docs indexed."""
        resp = client.post(
            "/api/v1/ask",
            json={"question": "What is Python?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "question" in data

    def test_ask_empty_question(self, client):
        """Empty question should return 422."""
        resp = client.post("/api/v1/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_short_question(self, client):
        """Very short question should return 422."""
        resp = client.post("/api/v1/ask", json={"question": "Hi"})
        assert resp.status_code == 422

    def test_ask_with_top_k(self, client):
        """Should accept top_k parameter."""
        resp = client.post(
            "/api/v1/ask",
            json={"question": "What is Python?", "top_k": 3},
        )
        assert resp.status_code == 200

    def test_ask_response_schema(self, client):
        """Response should have required fields."""
        resp = client.post(
            "/api/v1/ask",
            json={"question": "What is machine learning?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        for field in ["question", "answer", "confidence", "sources", "elapsed_ms", "mode"]:
            assert field in data

    def test_ask_invalid_top_k(self, client):
        """top_k > 20 should fail validation."""
        resp = client.post(
            "/api/v1/ask",
            json={"question": "What is Python?", "top_k": 100},
        )
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════
#  Ingest Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════

class TestIngestEndpoint:
    """Tests for the /ingest endpoint."""

    def test_ingest_document(self, client):
        """Should ingest a document successfully."""
        resp = client.post(
            "/api/v1/ingest",
            json={
                "documents": [
                    {
                        "title": "Test Doc",
                        "content": "This is a test document with enough content for chunking.",
                    }
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["documents_ingested"] == 1
        assert data["chunks_created"] >= 1

    def test_ingest_empty_documents(self, client):
        """Empty documents list should fail validation."""
        resp = client.post("/api/v1/ingest", json={"documents": []})
        assert resp.status_code == 422

    def test_ingest_missing_title(self, client):
        """Missing title should fail validation."""
        resp = client.post(
            "/api/v1/ingest",
            json={"documents": [{"content": "Some content here for testing."}]},
        )
        assert resp.status_code == 422

    def test_ingest_short_content(self, client):
        """Content too short should fail validation."""
        resp = client.post(
            "/api/v1/ingest",
            json={"documents": [{"title": "Test", "content": "Short"}]},
        )
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════
#  Upload Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUploadEndpoint:
    """Tests for the /upload endpoint."""

    def test_upload_txt_file(self, client):
        """Should upload a .txt file."""
        content = b"This is a test text file with enough content for indexing."
        resp = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    def test_upload_unsupported_format(self, client):
        """Should reject unsupported file types."""
        resp = client.post(
            "/api/v1/upload",
            files={"file": ("test.exe", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════
#  Index Info & Clear Tests
# ═══════════════════════════════════════════════════════════════════════

class TestIndexEndpoints:
    """Tests for index management endpoints."""

    def test_index_info(self, client):
        """GET /index/info should return index metadata."""
        resp = client.get("/api/v1/index/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_ready" in data
        assert "num_vectors" in data
        assert "embedding_model" in data

    def test_clear_index(self, client):
        """DELETE /index should clear the index."""
        resp = client.delete("/api/v1/index")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"


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
        assert spec["info"]["title"] == "Document Q&A API"
        assert "paths" in spec

    def test_docs_page(self, client):
        """Swagger UI should be accessible."""
        resp = client.get("/docs")
        assert resp.status_code == 200
