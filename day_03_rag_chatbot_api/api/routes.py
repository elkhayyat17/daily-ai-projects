from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse, HealthResponse, IngestRequest, IngestResponse
from inference.predictor import RAGPredictor

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    predictor = RAGPredictor.get_instance()
    return HealthResponse(
        status="ok",
        vector_store_ready=predictor.vectorstore is not None,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    predictor = RAGPredictor.get_instance()
    prediction = predictor.predict(payload.query, top_k=payload.top_k)
    return ChatResponse(
        answer=prediction.answer,
        sources=[{"id": s.id, "score": s.score} for s in prediction.sources],
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    predictor = RAGPredictor.get_instance()
    try:
        inserted = predictor.ingest([item.model_dump() for item in payload.items])
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return IngestResponse(inserted=inserted)
