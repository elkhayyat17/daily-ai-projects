"""
Day 05 — Document Q&A: Pydantic Request/Response Schemas
"""

from pydantic import BaseModel, Field


# ─── Request Models ──────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Request body for asking a question."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to ask about the documents.",
        examples=["What is machine learning?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve.",
    )


class DocumentInput(BaseModel):
    """A single document for ingestion."""
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Document title.",
        examples=["Python Guide"],
    )
    content: str = Field(
        ...,
        min_length=10,
        description="Document text content.",
        examples=["Python is a high-level programming language..."],
    )
    source: str = Field(
        default="api_upload",
        description="Source identifier.",
    )


class IngestRequest(BaseModel):
    """Request body for ingesting documents."""
    documents: list[DocumentInput] = Field(
        ...,
        min_length=1,
        description="List of documents to ingest.",
    )


# ─── Response Models ────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single retrieved chunk from the vector store."""
    text: str = Field(description="Chunk text (truncated).")
    title: str = Field(description="Source document title.")
    score: float = Field(description="Similarity score.")


class AnswerResponse(BaseModel):
    """Response to a question."""
    question: str
    answer: str
    confidence: float = Field(description="Answer confidence score (0-1).")
    sources: list[str] = Field(description="Source document titles.")
    retrieved_chunks: list[RetrievedChunk] = Field(
        description="Retrieved context chunks."
    )
    elapsed_ms: float = Field(description="Processing time in milliseconds.")
    mode: str = Field(description="Answer mode: extractive_qa or retrieval_only.")


class IngestResponse(BaseModel):
    """Response after ingesting documents."""
    status: str
    documents_ingested: int = 0
    chunks_created: int = 0
    total_vectors: int = 0
    message: str = ""


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    index_loaded: bool
    qa_model_loaded: bool
    num_vectors: int
    embedding_model: str
    qa_model: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str = ""
