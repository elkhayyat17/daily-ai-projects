from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_chroma import Chroma
from loguru import logger

from config import get_settings
from inference.preprocessing import clean_query
from training.model import get_embeddings


@dataclass
class SourceDoc:
    id: str
    score: float


@dataclass
class Prediction:
    answer: str
    sources: List[SourceDoc]


class RAGPredictor:
    _instance: "RAGPredictor | None" = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self._load_vectorstore()

    @classmethod
    def get_instance(cls) -> "RAGPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_vectorstore(self) -> None:
        if not self.settings.vector_db_dir.exists():
            logger.warning("Vector store not found. Predictor will run in fallback mode.")
            return
        self.vectorstore = Chroma(
            persist_directory=str(self.settings.vector_db_dir),
            embedding_function=self.embeddings,
            collection_name="rag_docs",
        )

    def _build_context(self, docs) -> str:
        context_parts = []
        total_chars = 0
        for doc, _score in docs:
            snippet = doc.page_content.strip()
            if total_chars + len(snippet) > self.settings.max_context_chars:
                break
            context_parts.append(snippet)
            total_chars += len(snippet)
        return "\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        if self.settings.embedding_provider == "openai" and self.settings.openai_api_key:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=self.settings.openai_model, api_key=self.settings.openai_api_key)
            prompt = (
                "You are a helpful RAG assistant. Answer the user based on the context. "
                "If the context is insufficient, say you are not sure.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
            return llm.invoke(prompt).content

        if not context:
            return "The knowledge base is empty. Please ingest documents and retrain the vector store."
        return (
            "Based on the retrieved documents, here is a concise answer:\n"
            f"{context}\n\n"
            "(Tip: Configure OPENAI_API_KEY for enhanced responses.)"
        )

    def predict(self, query: str, top_k: int | None = None) -> Prediction:
        cleaned = clean_query(query)
        if self.vectorstore is None:
            return Prediction(
                answer="Vector store not available. Run training/train.py to build it.",
                sources=[],
            )

        k = top_k or self.settings.top_k
        results = self.vectorstore.similarity_search_with_relevance_scores(cleaned.text, k=k)
        sources = [SourceDoc(id=item[0].metadata.get("id", "unknown"), score=float(item[1])) for item in results]
        context = self._build_context(results)
        answer = self._generate_answer(cleaned.text, context)
        return Prediction(answer=answer, sources=sources)

    def ingest(self, docs: List[dict]) -> int:
        if self.vectorstore is None:
            self._load_vectorstore()
        if self.vectorstore is None:
            raise RuntimeError("Vector store not ready.")

        documents = []
        for item in docs:
            documents.append(
                {
                    "page_content": item["content"],
                    "metadata": {"id": item["id"], "title": item.get("title", "")},
                }
            )

        self.vectorstore.add_texts(
            texts=[d["page_content"] for d in documents],
            metadatas=[d["metadata"] for d in documents],
        )
        logger.info("Ingested {} documents", len(documents))
        return len(documents)
