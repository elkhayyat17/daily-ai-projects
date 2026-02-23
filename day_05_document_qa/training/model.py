"""
Day 05 — Document Q&A: Vector Store Builder (Model Architecture)
Builds and manages the FAISS vector index for document retrieval.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class VectorStoreBuilder:
    """
    Builds a FAISS vector index from document chunks.
    Handles embedding generation, index construction, and persistence.
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        dimension: int = config.EMBEDDING_DIMENSION,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: list[dict] = []

    def load_embedding_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.success("Embedding model loaded.")
        return self.model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        model = self.load_embedding_model()
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # For cosine similarity via inner product
        )
        return np.array(embeddings, dtype=np.float32)

    def build_index(
        self, chunks: list[dict], text_key: str = "text"
    ) -> faiss.IndexFlatIP:
        """
        Build a FAISS index from document chunks.

        Args:
            chunks: List of dicts with at least a `text_key` field.
            text_key: Key to extract text from each chunk.

        Returns:
            Built FAISS index.
        """
        texts = [chunk[text_key] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        # Build FAISS inner-product index (cosine sim with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        # Store metadata
        self.metadata = []
        for i, chunk in enumerate(chunks):
            meta = {k: v for k, v in chunk.items() if k != text_key}
            meta["text"] = chunk[text_key]
            meta["index_id"] = i
            self.metadata.append(meta)

        logger.success(
            f"Built FAISS index with {self.index.ntotal} vectors "
            f"(dim={self.dimension})"
        )
        return self.index

    def save(
        self,
        index_path: Path = config.FAISS_INDEX_PATH,
        metadata_path: Path = config.FAISS_METADATA_PATH,
    ) -> None:
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save. Call build_index() first.")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.success(f"Saved index → {index_path}")
        logger.success(f"Saved metadata ({len(self.metadata)} chunks) → {metadata_path}")

    def load(
        self,
        index_path: Path = config.FAISS_INDEX_PATH,
        metadata_path: Path = config.FAISS_METADATA_PATH,
    ) -> bool:
        """Load a previously saved index and metadata."""
        if not index_path.exists() or not metadata_path.exists():
            logger.warning("Index files not found.")
            return False

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        logger.success(
            f"Loaded FAISS index ({self.index.ntotal} vectors) from {index_path}"
        )
        return True

    def search(
        self, query: str, top_k: int = config.TOP_K_RESULTS
    ) -> list[dict]:
        """
        Search the index for chunks similar to the query.

        Returns:
            List of dicts with 'text', 'score', and metadata fields.
        """
        if self.index is None:
            raise ValueError("No index loaded. Call build_index() or load() first.")

        query_embedding = self.embed_texts([query])
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)

        return results

    @property
    def is_ready(self) -> bool:
        """Check if the index is loaded and ready for search."""
        return self.index is not None and len(self.metadata) > 0

    @property
    def num_vectors(self) -> int:
        """Get number of vectors in the index."""
        return self.index.ntotal if self.index else 0
