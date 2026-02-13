from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)


class SourceItem(BaseModel):
    id: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class IngestItem(BaseModel):
    id: str
    title: str
    content: str


class IngestRequest(BaseModel):
    items: List[IngestItem]


class IngestResponse(BaseModel):
    inserted: int


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
