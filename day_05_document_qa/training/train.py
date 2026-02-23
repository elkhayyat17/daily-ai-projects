"""
Day 05 — Document Q&A: Training / Indexing Pipeline
Chunks documents, embeds them, and builds the FAISS index.
"""

import json
import time
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.prepare_data import load_documents, prepare_sample_data
from training.model import VectorStoreBuilder
from inference.preprocessing import DocumentChunker


def run_indexing_pipeline(
    documents: list[dict] | None = None,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> dict:
    """
    Full indexing pipeline: prepare data → chunk → embed → build index → save.

    Returns:
        Dictionary with pipeline statistics.
    """
    start_time = time.time()

    # Step 1: Load or prepare documents
    if documents is None:
        doc_file = config.PROCESSED_DIR / "documents.jsonl"
        if not doc_file.exists():
            logger.info("No documents found. Preparing sample data...")
            prepare_sample_data()
        documents = load_documents()

    logger.info(f"📄 Loaded {len(documents)} documents")

    # Step 2: Chunk documents
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []

    for doc in documents:
        chunks = chunker.chunk_text(
            text=doc["content"],
            metadata={
                "doc_id": doc.get("id", "unknown"),
                "title": doc.get("title", "Untitled"),
                "source": doc.get("source", "unknown"),
            },
        )
        all_chunks.extend(chunks)

    logger.info(f"✂️  Created {len(all_chunks)} chunks from {len(documents)} documents")

    # Step 3: Build FAISS index
    builder = VectorStoreBuilder()
    builder.build_index(all_chunks, text_key="text")

    # Step 4: Save index
    builder.save()

    elapsed = time.time() - start_time
    stats = {
        "num_documents": len(documents),
        "num_chunks": len(all_chunks),
        "num_vectors": builder.num_vectors,
        "embedding_dim": config.EMBEDDING_DIMENSION,
        "index_path": str(config.FAISS_INDEX_PATH),
        "elapsed_seconds": round(elapsed, 2),
    }

    logger.success(f"✅ Indexing complete in {elapsed:.1f}s")
    logger.info(f"   Documents: {stats['num_documents']}")
    logger.info(f"   Chunks:    {stats['num_chunks']}")
    logger.info(f"   Vectors:   {stats['num_vectors']}")

    # Save stats
    stats_path = config.ARTIFACTS_DIR / "indexing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    logger.add(config.LOG_FILE, rotation="10 MB", level=config.LOG_LEVEL)
    stats = run_indexing_pipeline()
    print(json.dumps(stats, indent=2))
