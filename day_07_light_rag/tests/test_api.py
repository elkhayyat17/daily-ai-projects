"""API integration tests for Light RAG."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from config import get_settings
from inference.predictor import LightRAGPredictor
from training.train import train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTRAG_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    get_settings.cache_clear()
    LightRAGPredictor._instance = None
    yield
    get_settings.cache_clear()
    LightRAGPredictor._instance = None


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def trained_client(client):
    train()
    LightRAGPredictor._instance = None
    return client


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_without_index(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["index_ready"] is False
        assert body["total_chunks"] == 0

    def test_health_with_index(self, trained_client):
        resp = trained_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["index_ready"] is True
        assert body["total_chunks"] > 0


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_fallback(self, client):
        resp = client.post("/query", json={"query": "What is RAG?"})
        assert resp.status_code == 200
        assert "not available" in resp.json()["answer"].lower() or "index" in resp.json()["answer"].lower()

    def test_query_returns_sources(self, trained_client):
        resp = trained_client.post("/query", json={"query": "What is RAG?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["sources"]
        assert body["mode"] == "hybrid"

    def test_query_semantic_mode(self, trained_client):
        resp = trained_client.post(
            "/query", json={"query": "sentence transformers", "mode": "semantic"}
        )
        assert resp.status_code == 200
        assert resp.json()["mode"] == "semantic"

    def test_query_bm25_mode(self, trained_client):
        resp = trained_client.post(
            "/query", json={"query": "BM25 keyword", "mode": "bm25"}
        )
        assert resp.status_code == 200
        assert resp.json()["mode"] == "bm25"

    def test_query_custom_top_k(self, trained_client):
        resp = trained_client.post(
            "/query", json={"query": "chunking strategies", "top_k": 3}
        )
        assert resp.status_code == 200
        assert len(resp.json()["sources"]) <= 3

    def test_query_validation_short(self, client):
        resp = client.post("/query", json={"query": "hi"})
        assert resp.status_code == 422

    def test_query_validation_invalid_mode(self, client):
        resp = client.post("/query", json={"query": "valid query", "mode": "invalid"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_fails_without_index(self, client):
        resp = client.post(
            "/ingest",
            json={"items": [{"id": "x", "title": "T", "content": "C"}]},
        )
        assert resp.status_code == 409

    def test_ingest_success(self, trained_client):
        resp = trained_client.post(
            "/ingest",
            json={
                "items": [
                    {"id": "ing-1", "title": "Ingested", "content": "Test doc"},
                ]
            },
        )
        assert resp.status_code == 200
        assert resp.json()["inserted"] == 1

    def test_ingest_multiple(self, trained_client):
        resp = trained_client.post(
            "/ingest",
            json={
                "items": [
                    {"id": "m1", "title": "A", "content": "First"},
                    {"id": "m2", "title": "B", "content": "Second"},
                ]
            },
        )
        assert resp.status_code == 200
        assert resp.json()["inserted"] == 2


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_without_index(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 503

    def test_stats_with_index(self, trained_client):
        resp = trained_client.get("/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_chunks"] > 0
        assert body["embedding_dim"] > 0
        assert body["embedding_provider"] == "fake"
