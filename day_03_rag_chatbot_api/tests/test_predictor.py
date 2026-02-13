from __future__ import annotations

import os

import pytest

from config import get_settings
from data.prepare_data import prepare_data
from inference.preprocessing import clean_query
from inference.predictor import RAGPredictor
from training.evaluate import evaluate
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


def test_clean_query_valid():
    assert clean_query("hello world").text == "hello world"


def test_clean_query_too_short():
    with pytest.raises(ValueError):
        clean_query("hi")


def test_clean_query_too_long():
    with pytest.raises(ValueError):
        clean_query("x" * 3000)


def test_prepare_data_creates_file():
    output = prepare_data()
    assert output.exists()


def test_train_builds_vectorstore():
    path = train()
    assert path.exists()


def test_predictor_fallback_without_vectorstore():
    predictor = RAGPredictor.get_instance()
    result = predictor.predict("What is RAG?")
    assert "Vector store not available" in result.answer


def test_predictor_returns_sources_after_training():
    train()
    RAGPredictor._instance = None
    predictor = RAGPredictor.get_instance()
    result = predictor.predict("What is RAG?")
    assert result.sources


def test_ingest_adds_documents():
    train()
    RAGPredictor._instance = None
    predictor = RAGPredictor.get_instance()
    inserted = predictor.ingest([
        {"id": "doc-9", "title": "New", "content": "Extra document for ingestion"}
    ])
    assert inserted == 1


def test_evaluate_generates_report():
    train()
    report_path = evaluate()
    assert report_path.exists()


def test_embedding_provider_fake_env():
    settings = get_settings()
    assert settings.embedding_provider == "fake"
