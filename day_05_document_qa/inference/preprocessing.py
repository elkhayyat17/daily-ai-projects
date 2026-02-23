"""
Day 05 — Document Q&A: Input Preprocessing
Document parsing, chunking, and text cleaning utilities.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Optional
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class DocumentChunker:
    """Splits documents into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self, text: str, metadata: Optional[dict] = None
    ) -> list[dict]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of dicts with 'text' and metadata fields.
        """
        text = self.clean_text(text)
        if not text:
            return []

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a sentence boundary
            if end < len(text):
                # Look for the last period, question mark, or newline before end
                for sep in [". ", "? ", "! ", "\n"]:
                    last_sep = text.rfind(sep, start + self.chunk_size // 2, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = {
                    "text": chunk_text,
                    "chunk_index": chunk_idx,
                    "start_char": start,
                    "end_char": end,
                    "chunk_id": self._generate_chunk_id(chunk_text, chunk_idx),
                }
                if metadata:
                    chunk.update(metadata)
                chunks.append(chunk)
                chunk_idx += 1

            # Move start forward
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove control characters (except newlines)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()

    @staticmethod
    def _generate_chunk_id(text: str, index: int) -> str:
        """Generate a unique ID for a chunk."""
        content = f"{text[:100]}_{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class DocumentParser:
    """Parses different document formats into plain text."""

    SUPPORTED = config.SUPPORTED_EXTENSIONS

    @classmethod
    def parse_file(cls, file_path: Path) -> dict:
        """
        Parse a file into a document dict.

        Returns:
            Dict with 'title', 'content', 'source', 'format'.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED:
            raise ValueError(
                f"Unsupported format: {ext}. Supported: {cls.SUPPORTED}"
            )

        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > config.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large: {size_mb:.1f}MB (max {config.MAX_FILE_SIZE_MB}MB)"
            )

        content = ""
        if ext == ".txt" or ext == ".md":
            content = cls._parse_text(path)
        elif ext == ".pdf":
            content = cls._parse_pdf(path)
        elif ext == ".docx":
            content = cls._parse_docx(path)
        elif ext == ".csv":
            content = cls._parse_csv(path)
        elif ext == ".json":
            content = cls._parse_json(path)

        return {
            "title": path.stem.replace("_", " ").title(),
            "content": content,
            "source": str(path),
            "format": ext,
        }

    @staticmethod
    def _parse_text(path: Path) -> str:
        """Parse plain text file."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            import chardet
            raw = path.read_bytes()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding", "utf-8")
            return raw.decode(encoding, errors="replace")

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        """Parse PDF file using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            logger.warning("PyPDF2 not installed. Cannot parse PDF files.")
            return ""
        except Exception as e:
            logger.error(f"Error parsing PDF {path}: {e}")
            return ""

    @staticmethod
    def _parse_docx(path: Path) -> str:
        """Parse DOCX file using python-docx."""
        try:
            from docx import Document
            doc = Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            logger.warning("python-docx not installed. Cannot parse DOCX files.")
            return ""
        except Exception as e:
            logger.error(f"Error parsing DOCX {path}: {e}")
            return ""

    @staticmethod
    def _parse_csv(path: Path) -> str:
        """Parse CSV file — concatenate rows as text."""
        try:
            import csv
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_text = " | ".join(f"{k}: {v}" for k, v in row.items())
                    rows.append(row_text)
            return "\n".join(rows)
        except Exception as e:
            logger.error(f"Error parsing CSV {path}: {e}")
            return ""

    @staticmethod
    def _parse_json(path: Path) -> str:
        """Parse JSON file — flatten to text."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Error parsing JSON {path}: {e}")
            return ""


def validate_question(question: str) -> str:
    """Validate and clean a question string."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    question = question.strip()
    if len(question) < 3:
        raise ValueError("Question is too short (min 3 characters).")
    if len(question) > 1000:
        raise ValueError("Question is too long (max 1000 characters).")
    return question
