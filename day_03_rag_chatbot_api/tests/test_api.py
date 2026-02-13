from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from config import get_settings
from inference.predictor import RAGPredictor
from training.train import train


@pytest.fixture(autouse=True)
def _set_test_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    get_settings.cache_clear()
    RAGPredictor._instance = None
    yield
    get_settings.cache_clear()
    RAGPredictor._instance = None


def test_health_without_vectorstore():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["vector_store_ready"] is False


def test_chat_fallback_without_vectorstore():
    app = create_app()
    client = TestClient(app)
    response = client.post("/chat", json={"query": "What is RAG?"})
    assert response.status_code == 200
    assert "Vector store not available" in response.json()["answer"]


def test_ingest_conflict_without_vectorstore():
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/ingest",
        json={"items": [{"id": "doc-1", "title": "Title", "content": "Content"}]},
    )
    assert response.status_code == 409


def test_chat_with_vectorstore():
    train()
    RAGPredictor._instance = None
    app = create_app()
    client = TestClient(app)
    response = client.post("/chat", json={"query": "What is RAG?"})
    assert response.status_code == 200
    assert response.json()["sources"]


def test_ingest_with_vectorstore():
    train()
    RAGPredictor._instance = None
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/ingest",
        json={"items": [{"id": "doc-8", "title": "Extra", "content": "Extra doc"}]},
    )
    assert response.status_code == 200
    assert response.json()["inserted"] == 1


def test_chat_validation_error():
    app = create_app()
    client = TestClient(app)
    response = client.post("/chat", json={"query": "hi"})
    assert response.status_code == 422
