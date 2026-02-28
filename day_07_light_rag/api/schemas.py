"""Pydantic request / response models for the Light RAG API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Chat / query request."""
    query: str = Field(..., min_length=3, max_length=2000, description="User question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of results")
    mode: Optional[str] = Field(
        default="hybrid",
        pattern="^(hybrid|semantic|bm25)$",
        description="Retrieval mode",
    )


class IngestItem(BaseModel):
    id: str
    title: str = ""
    doc_id: str = ""
    content: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    items: List[IngestItem] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class SourceItemResponse(BaseModel):
    id: str
    doc_id: str
    title: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItemResponse]
    mode: str


class IngestResponse(BaseModel):
    inserted: int


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    total_chunks: int


class StatsResponse(BaseModel):
    total_chunks: int
    embedding_dim: int
    embedding_provider: str
    bm25_weight: float
    semantic_weight: float
