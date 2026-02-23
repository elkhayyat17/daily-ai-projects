"""
Day 05 — Document Q&A: Unit Tests for Predictor & Core Components
"""

import sys
import json
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.preprocessing import DocumentChunker, DocumentParser, validate_question
from data.prepare_data import prepare_sample_data, load_documents, SAMPLE_DOCS
import config


# ═══════════════════════════════════════════════════════════════════════
#  DocumentChunker Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDocumentChunker:
    """Tests for the DocumentChunker class."""

    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    def test_chunk_short_text(self):
        """Short text should produce a single chunk."""
        chunks = self.chunker.chunk_text("Hello world, this is a test.")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello world, this is a test."
        assert chunks[0]["chunk_index"] == 0

    def test_chunk_long_text(self):
        """Long text should produce multiple chunks."""
        text = "This is a sentence. " * 50  # ~1000 chars
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_overlap(self):
        """Chunks should overlap by the specified amount."""
        text = "A" * 300
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        # Each subsequent chunk starts before the previous one ended
        for i in range(1, len(chunks)):
            assert chunks[i]["start_char"] < chunks[i - 1]["end_char"]

    def test_chunk_empty_text(self):
        """Empty text should return empty list."""
        assert self.chunker.chunk_text("") == []
        assert self.chunker.chunk_text("   ") == []

    def test_chunk_with_metadata(self):
        """Metadata should be attached to each chunk."""
        meta = {"title": "Test Doc", "source": "unit_test"}
        chunks = self.chunker.chunk_text("Hello world test document.", metadata=meta)
        assert len(chunks) == 1
        assert chunks[0]["title"] == "Test Doc"
        assert chunks[0]["source"] == "unit_test"

    def test_chunk_has_required_fields(self):
        """Each chunk should have text, chunk_index, chunk_id, etc."""
        chunks = self.chunker.chunk_text("Test document content here.")
        chunk = chunks[0]
        assert "text" in chunk
        assert "chunk_index" in chunk
        assert "chunk_id" in chunk
        assert "start_char" in chunk
        assert "end_char" in chunk

    def test_clean_text(self):
        """clean_text should normalize whitespace."""
        result = DocumentChunker.clean_text("  hello   world  \n\n  test  ")
        assert result == "hello world test"

    def test_clean_text_empty(self):
        """clean_text should handle empty strings."""
        assert DocumentChunker.clean_text("") == ""
        assert DocumentChunker.clean_text(None) == ""


# ═══════════════════════════════════════════════════════════════════════
#  DocumentParser Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDocumentParser:
    """Tests for the DocumentParser class."""

    def test_parse_txt_file(self, tmp_path):
        """Should parse .txt files."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello from a text file.", encoding="utf-8")

        result = DocumentParser.parse_file(txt_file)
        assert result["content"] == "Hello from a text file."
        assert result["format"] == ".txt"
        assert result["title"] == "Test"

    def test_parse_md_file(self, tmp_path):
        """Should parse .md files."""
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Hello\nThis is markdown.", encoding="utf-8")

        result = DocumentParser.parse_file(md_file)
        assert "Hello" in result["content"]
        assert result["format"] == ".md"

    def test_parse_json_file(self, tmp_path):
        """Should parse .json files."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")

        result = DocumentParser.parse_file(json_file)
        assert "key" in result["content"]
        assert result["format"] == ".json"

    def test_parse_nonexistent_file(self):
        """Should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DocumentParser.parse_file(Path("/nonexistent/file.txt"))

    def test_parse_unsupported_extension(self, tmp_path):
        """Should raise ValueError for unsupported formats."""
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("test", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported format"):
            DocumentParser.parse_file(unsupported)

    def test_supported_extensions(self):
        """SUPPORTED should contain the expected extensions."""
        assert ".txt" in DocumentParser.SUPPORTED
        assert ".pdf" in DocumentParser.SUPPORTED
        assert ".md" in DocumentParser.SUPPORTED
        assert ".docx" in DocumentParser.SUPPORTED


# ═══════════════════════════════════════════════════════════════════════
#  Question Validation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestValidateQuestion:
    """Tests for the validate_question function."""

    def test_valid_question(self):
        assert validate_question("What is Python?") == "What is Python?"

    def test_strips_whitespace(self):
        assert validate_question("  Hello?  ") == "Hello?"

    def test_empty_question(self):
        with pytest.raises(ValueError, match="empty"):
            validate_question("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError, match="empty"):
            validate_question("   ")

    def test_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            validate_question("Hi")

    def test_too_long(self):
        with pytest.raises(ValueError, match="too long"):
            validate_question("x" * 1001)


# ═══════════════════════════════════════════════════════════════════════
#  Data Preparation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDataPreparation:
    """Tests for data preparation functions."""

    def test_sample_docs_not_empty(self):
        """Sample docs should exist."""
        assert len(SAMPLE_DOCS) > 0

    def test_sample_docs_have_required_fields(self):
        """Each sample doc should have title and content."""
        for doc in SAMPLE_DOCS:
            assert "title" in doc
            assert "content" in doc
            assert len(doc["content"]) > 50

    def test_prepare_sample_data(self):
        """prepare_sample_data should create a JSONL file."""
        output = prepare_sample_data()
        assert output.exists()
        assert output.suffix == ".jsonl"

    def test_load_documents(self):
        """load_documents should return list of dicts."""
        prepare_sample_data()
        docs = load_documents()
        assert len(docs) == len(SAMPLE_DOCS)
        for doc in docs:
            assert "id" in doc
            assert "title" in doc
            assert "content" in doc


# ═══════════════════════════════════════════════════════════════════════
#  Config Tests
# ═══════════════════════════════════════════════════════════════════════

class TestConfig:
    """Tests for configuration."""

    def test_paths_exist(self):
        """Config directories should be created."""
        assert config.DATA_DIR.exists()
        assert config.ARTIFACTS_DIR.exists()

    def test_embedding_dimension(self):
        assert config.EMBEDDING_DIMENSION == 384

    def test_chunk_config(self):
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
