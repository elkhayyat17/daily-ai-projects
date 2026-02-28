"""Unit tests for the Light RAG predictor, data pipeline, and training."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from config import get_settings
from data.prepare_data import chunk_text, prepare_data
from inference.preprocessing import CleanedQuery, clean_query
from inference.predictor import LightRAGPredictor
from training.evaluate import evaluate
from training.model import FakeEmbeddings, get_embeddings
from training.train import LightIndex, train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Point all paths to a temp directory and use fake embeddings."""
    monkeypatch.setenv("LIGHTRAG_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    get_settings.cache_clear()
    LightRAGPredictor._instance = None
    yield
    get_settings.cache_clear()
    LightRAGPredictor._instance = None


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_clean_query_valid(self):
        result = clean_query("What is hybrid search?")
        assert isinstance(result, CleanedQuery)
        assert result.text == "What is hybrid search?"

    def test_clean_query_collapses_whitespace(self):
        assert clean_query("  hello   world  ").text == "hello world"

    def test_clean_query_too_short(self):
        with pytest.raises(ValueError, match="at least 3"):
            clean_query("hi")

    def test_clean_query_too_long(self):
        with pytest.raises(ValueError, match="at most"):
            clean_query("x" * 2500)


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

class TestChunking:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world", chunk_size=512)
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        text = "word " * 500  # ~2500 chars
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_overlap_present(self):
        text = "A. B. C. D. E. F. G. H. I. J. " * 30
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=30)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Data preparation tests
# ---------------------------------------------------------------------------

class TestPrepareData:
    def test_prepare_data_creates_file(self):
        output = prepare_data()
        assert output.exists()
        assert output.name == "chunks.jsonl"

    def test_prepare_data_jsonl_format(self):
        output = prepare_data()
        with output.open() as fh:
            lines = fh.readlines()
        assert len(lines) > 0
        first = json.loads(lines[0])
        assert "id" in first
        assert "content" in first


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_fake_embeddings_shape(self):
        emb = FakeEmbeddings(dim=384)
        vecs = emb.encode(["hello", "world"])
        assert vecs.shape == (2, 384)

    def test_fake_embeddings_normalized(self):
        emb = FakeEmbeddings(dim=128)
        vecs = emb.encode(["test"])
        norm = np.linalg.norm(vecs[0])
        assert abs(norm - 1.0) < 1e-5

    def test_get_embeddings_returns_fake(self):
        emb = get_embeddings()
        assert isinstance(emb, FakeEmbeddings)


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTraining:
    def test_train_builds_index(self):
        index_dir = train()
        assert index_dir.exists()
        assert (index_dir / "embeddings.npy").exists()
        assert (index_dir / "bm25.pkl").exists()
        assert (index_dir / "metadata.json").exists()
        assert (index_dir / "texts.json").exists()

    def test_light_index_save_load_roundtrip(self, tmp_path):
        emb = np.random.randn(5, 64).astype(np.float32)
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi([["a", "b"], ["c", "d"], ["e"], ["f", "g"], ["h"]])
        meta = [{"id": f"c-{i}", "doc_id": f"d-{i}", "title": f"T{i}"} for i in range(5)]
        texts = [f"text {i}" for i in range(5)]

        idx = LightIndex(emb, bm25, meta, texts)
        save_dir = tmp_path / "idx"
        idx.save(save_dir)

        loaded = LightIndex.load(save_dir)
        assert np.allclose(loaded.embeddings, emb)
        assert loaded.metadata == meta
        assert loaded.texts == texts


# ---------------------------------------------------------------------------
# Predictor tests
# ---------------------------------------------------------------------------

class TestPredictor:
    def test_fallback_without_index(self):
        predictor = LightRAGPredictor.get_instance()
        result = predictor.predict("What is RAG?")
        assert "not available" in result.answer.lower() or "index" in result.answer.lower()
        assert result.sources == []

    def test_predict_after_training(self):
        train()
        LightRAGPredictor._instance = None
        predictor = LightRAGPredictor.get_instance()
        result = predictor.predict("What is RAG?")
        assert result.sources
        assert result.mode == "hybrid"

    def test_predict_semantic_mode(self):
        train()
        LightRAGPredictor._instance = None
        predictor = LightRAGPredictor.get_instance()
        result = predictor.predict("What is RAG?", mode="semantic")
        assert result.mode == "semantic"
        assert result.sources

    def test_predict_bm25_mode(self):
        train()
        LightRAGPredictor._instance = None
        predictor = LightRAGPredictor.get_instance()
        result = predictor.predict("BM25 ranking", mode="bm25")
        assert result.mode == "bm25"

    def test_ingest_adds_docs(self):
        train()
        LightRAGPredictor._instance = None
        predictor = LightRAGPredictor.get_instance()
        original_count = len(predictor.index.texts)
        inserted = predictor.ingest([
            {"id": "new-1", "title": "New", "content": "Newly ingested doc"},
        ])
        assert inserted == 1
        assert len(predictor.index.texts) == original_count + 1

    def test_ingest_fails_without_index(self):
        predictor = LightRAGPredictor.get_instance()
        with pytest.raises(RuntimeError, match="not loaded"):
            predictor.ingest([{"id": "x", "content": "y"}])

    def test_is_ready_false_before_training(self):
        predictor = LightRAGPredictor.get_instance()
        assert predictor.is_ready is False

    def test_is_ready_true_after_training(self):
        train()
        LightRAGPredictor._instance = None
        predictor = LightRAGPredictor.get_instance()
        assert predictor.is_ready is True


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_evaluate_generates_report(self):
        train()
        report_path = evaluate()
        assert report_path.exists()

    def test_evaluate_report_structure(self):
        train()
        report_path = evaluate()
        report = json.loads(report_path.read_text())
        assert "modes" in report
        for mode in ("semantic", "bm25", "hybrid"):
            assert mode in report["modes"]
            assert "hit_rate" in report["modes"][mode]
            assert "mrr" in report["modes"][mode]

    def test_evaluate_fails_without_index(self):
        with pytest.raises(FileNotFoundError):
            evaluate()
