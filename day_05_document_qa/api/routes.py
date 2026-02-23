"""
Day 05 — Document Q&A: API Routes
All FastAPI endpoint definitions.
"""

import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from loguru import logger

from api.schemas import (
    QuestionRequest,
    IngestRequest,
    AnswerResponse,
    IngestResponse,
    HealthResponse,
    ErrorResponse,
)
from inference.predictor import DocumentQAPredictor
import config

router = APIRouter()


def get_predictor() -> DocumentQAPredictor:
    """Get the singleton predictor instance."""
    return DocumentQAPredictor()


# ─── Health Check ────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check():
    """Check API health and model status."""
    predictor = get_predictor()
    st = predictor.status
    return HealthResponse(
        status="healthy" if predictor.is_ready else "degraded",
        index_loaded=st["index_loaded"],
        qa_model_loaded=st["qa_model_loaded"],
        num_vectors=st["num_vectors"],
        embedding_model=st["embedding_model"],
        qa_model=st["qa_model"],
    )


# ─── Question Answering ─────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=AnswerResponse,
    tags=["Q&A"],
    summary="Ask a question about the documents",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer based on indexed documents.

    The system retrieves relevant document chunks using vector similarity
    search and then extracts an answer using an extractive QA model.
    """
    try:
        predictor = get_predictor()
        result = predictor.ask(
            question=request.question,
            top_k=request.top_k,
        )
        return AnswerResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


# ─── Document Ingestion ─────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Documents"],
    summary="Ingest documents via JSON",
    responses={400: {"model": ErrorResponse}},
)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the vector store for Q&A.

    Provide a list of documents with title and content.
    Documents will be chunked, embedded, and indexed.
    """
    try:
        predictor = get_predictor()
        docs = [doc.model_dump() for doc in request.documents]
        result = predictor.ingest_documents(docs)
        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {str(e)}",
        )


@router.post(
    "/upload",
    response_model=IngestResponse,
    tags=["Documents"],
    summary="Upload a document file",
    responses={400: {"model": ErrorResponse}},
)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document file (TXT, PDF, MD, DOCX, CSV, JSON) for ingestion.

    The file will be parsed, chunked, embedded, and indexed.
    """
    # Validate extension
    filename = file.filename or "unknown.txt"
    ext = Path(filename).suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}. Supported: {config.SUPPORTED_EXTENSIONS}",
        )

    try:
        # Save to temp file
        content = await file.read()
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, dir=str(config.RAW_DIR)
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Ingest
        predictor = get_predictor()
        result = predictor.ingest_file(tmp_path)

        # Cleanup temp file
        try:
            tmp_path.unlink()
        except Exception:
            pass

        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload error: {str(e)}",
        )


# ─── Index Info ──────────────────────────────────────────────────────

@router.get(
    "/index/info",
    tags=["Documents"],
    summary="Get index information",
)
async def index_info():
    """Get information about the current vector index."""
    predictor = get_predictor()
    return {
        "is_ready": predictor.is_ready,
        "num_vectors": predictor.vector_store.num_vectors,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_dimension": config.EMBEDDING_DIMENSION,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }


@router.delete(
    "/index",
    tags=["Documents"],
    summary="Clear the index",
)
async def clear_index():
    """Clear the vector index and all indexed documents."""
    try:
        # Remove index files
        if config.FAISS_INDEX_PATH.exists():
            config.FAISS_INDEX_PATH.unlink()
        if config.FAISS_METADATA_PATH.exists():
            config.FAISS_METADATA_PATH.unlink()

        # Reset predictor
        DocumentQAPredictor.reset()

        return {"status": "success", "message": "Index cleared."}
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
