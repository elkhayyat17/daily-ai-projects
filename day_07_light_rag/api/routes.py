"""API routes for Light RAG."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceItemResponse,
    StatsResponse,
)
from inference.predictor import LightRAGPredictor

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    predictor = LightRAGPredictor.get_instance()
    total = len(predictor.index.texts) if predictor.index else 0
    return HealthResponse(
        status="ok",
        index_ready=predictor.is_ready,
        total_chunks=total,
    )


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    predictor = LightRAGPredictor.get_instance()
    result = predictor.predict(
        payload.query,
        top_k=payload.top_k,
        mode=payload.mode or "hybrid",
    )
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceItemResponse(
                id=s.id,
                doc_id=s.doc_id,
                title=s.title,
                score=s.score,
                snippet=s.snippet,
            )
            for s in result.sources
        ],
        mode=result.mode,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    predictor = LightRAGPredictor.get_instance()
    try:
        inserted = predictor.ingest([item.model_dump() for item in payload.items])
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return IngestResponse(inserted=inserted)


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    predictor = LightRAGPredictor.get_instance()
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Index not loaded.")
    return StatsResponse(
        total_chunks=len(predictor.index.texts),
        embedding_dim=predictor.index.embeddings.shape[1],
        embedding_provider=predictor.settings.embedding_provider,
        bm25_weight=predictor.settings.bm25_weight,
        semantic_weight=predictor.settings.semantic_weight,
    )
