"""
Day 05 — Document Q&A: Production Inference Engine
Singleton QA predictor that combines vector retrieval with extractive QA.
"""

import threading
import time
from pathlib import Path
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from training.model import VectorStoreBuilder
from inference.preprocessing import DocumentChunker, DocumentParser, validate_question


class DocumentQAPredictor:
    """
    Production-grade Document QA engine (Singleton).

    Combines:
    1. FAISS vector search for relevant chunk retrieval
    2. Extractive QA model for answer extraction
    """

    _instance: Optional["DocumentQAPredictor"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.vector_store = VectorStoreBuilder()
        self.qa_pipeline = None
        self._model_loaded = False
        self._index_loaded = False
        logger.info("DocumentQAPredictor initialized.")

    def load(self) -> bool:
        """Load the vector index and QA model."""
        success = True

        # Load FAISS index
        if not self._index_loaded:
            if self.vector_store.load():
                self._index_loaded = True
                logger.success(
                    f"Vector index loaded ({self.vector_store.num_vectors} vectors)"
                )
            else:
                logger.warning("No vector index found. Upload documents first.")
                success = False

        # Load QA model
        if not self._model_loaded:
            try:
                from transformers import pipeline

                logger.info(f"Loading QA model: {config.QA_MODEL_NAME}")
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=config.QA_MODEL_NAME,
                    tokenizer=config.QA_MODEL_NAME,
                )
                self._model_loaded = True
                logger.success("QA model loaded.")
            except Exception as e:
                logger.warning(f"Could not load QA model: {e}. Will use retrieval-only mode.")
                success = False

        return success

    def ask(
        self,
        question: str,
        top_k: int = config.TOP_K_RESULTS,
    ) -> dict:
        """
        Answer a question using the document knowledge base.

        Args:
            question: The question to answer.
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with answer, confidence, sources, and retrieved chunks.
        """
        start_time = time.time()

        # Validate
        question = validate_question(question)

        # Check readiness
        if not self.vector_store.is_ready:
            return {
                "question": question,
                "answer": "No documents have been indexed yet. Please upload documents first.",
                "confidence": 0.0,
                "sources": [],
                "retrieved_chunks": [],
                "elapsed_ms": 0,
                "mode": "error",
            }

        # Retrieve relevant chunks
        retrieved = self.vector_store.search(question, top_k=top_k)
        logger.info(
            f"Retrieved {len(retrieved)} chunks for: '{question[:80]}...'"
        )

        # Try extractive QA
        answer = ""
        confidence = 0.0
        mode = "retrieval_only"

        if self.qa_pipeline and retrieved:
            # Combine top chunks as context
            context = "\n\n".join(r["text"] for r in retrieved)

            try:
                qa_result = self.qa_pipeline(
                    question=question,
                    context=context[:2048],  # Limit context length
                )
                answer = qa_result.get("answer", "")
                confidence = qa_result.get("score", 0.0)
                mode = "extractive_qa"

                if confidence < config.NO_ANSWER_THRESHOLD:
                    answer = self._fallback_answer(retrieved)
                    mode = "retrieval_only"
            except Exception as e:
                logger.warning(f"QA model failed: {e}. Using retrieval-only.")
                answer = self._fallback_answer(retrieved)
        elif retrieved:
            answer = self._fallback_answer(retrieved)
        else:
            answer = "No relevant information found in the knowledge base."

        elapsed = (time.time() - start_time) * 1000

        # Collect unique sources
        sources = list(
            {r.get("title", "Unknown") for r in retrieved if r.get("title")}
        )

        return {
            "question": question,
            "answer": answer,
            "confidence": round(confidence, 4),
            "sources": sources,
            "retrieved_chunks": [
                {
                    "text": r["text"][:300],
                    "title": r.get("title", "Unknown"),
                    "score": round(r["score"], 4),
                }
                for r in retrieved
            ],
            "elapsed_ms": round(elapsed, 1),
            "mode": mode,
        }

    def ingest_documents(self, documents: list[dict]) -> dict:
        """
        Ingest new documents into the vector store.

        Args:
            documents: List of dicts with 'title' and 'content' keys.

        Returns:
            Ingestion statistics.
        """
        chunker = DocumentChunker()
        all_chunks = []

        for doc in documents:
            chunks = chunker.chunk_text(
                text=doc.get("content", ""),
                metadata={
                    "title": doc.get("title", "Untitled"),
                    "source": doc.get("source", "upload"),
                },
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"status": "error", "message": "No valid content to index."}

        # Build index
        self.vector_store.build_index(all_chunks, text_key="text")
        self.vector_store.save()
        self._index_loaded = True

        stats = {
            "status": "success",
            "documents_ingested": len(documents),
            "chunks_created": len(all_chunks),
            "total_vectors": self.vector_store.num_vectors,
        }
        logger.success(f"Ingested {len(documents)} docs → {len(all_chunks)} chunks")
        return stats

    def ingest_file(self, file_path: str | Path) -> dict:
        """Parse and ingest a single file."""
        parser = DocumentParser()
        doc = parser.parse_file(Path(file_path))
        return self.ingest_documents([doc])

    @staticmethod
    def _fallback_answer(retrieved: list[dict]) -> str:
        """Generate a fallback answer from retrieved chunks."""
        if not retrieved:
            return "No relevant information found."
        top = retrieved[0]
        text = top["text"]
        # Return the most relevant chunk text (truncated)
        if len(text) > 500:
            text = text[:500] + "..."
        return f"Based on the document '{top.get('title', 'Unknown')}': {text}"

    @property
    def is_ready(self) -> bool:
        """Check if the system is ready for queries."""
        return self._index_loaded

    @property
    def status(self) -> dict:
        """Get system status."""
        return {
            "index_loaded": self._index_loaded,
            "qa_model_loaded": self._model_loaded,
            "num_vectors": self.vector_store.num_vectors,
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "qa_model": config.QA_MODEL_NAME if self._model_loaded else "not loaded",
        }

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None
