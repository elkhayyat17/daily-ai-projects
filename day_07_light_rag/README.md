# 🪶 Day 07 — Light RAG: Lightweight Retrieval-Augmented Generation

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal?logo=fastapi)
![NumPy](https://img.shields.io/badge/NumPy-Vector_Search-orange?logo=numpy)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

> **Zero-dependency vector database** — hybrid BM25 + cosine search powered by NumPy and `rank-bm25`. No LangChain. No ChromaDB. No FAISS. Just pure, lightweight RAG.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Light RAG Pipeline                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  📥 Documents ──► Chunking ──► Embedding ──► NumPy Matrix    │
│                                    │              │           │
│                              BM25 Index     Dense Index       │
│                                    │              │           │
│  🔍 Query ─────► Preprocessing ───►├──────────────┤           │
│                                    │              │           │
│                               BM25 Scores   Cosine Scores    │
│                                    │              │           │
│                                    ▼              ▼           │
│                              ┌─────────────────────┐         │
│                              │   Weighted Fusion    │         │
│                              │  (0.3 BM25 + 0.7    │         │
│                              │   Semantic)          │         │
│                              └────────┬────────────┘         │
│                                       │                       │
│                                  Top-K Chunks                 │
│                                       │                       │
│                              ┌────────▼────────────┐         │
│                              │  Answer Generation   │         │
│                              │  (OpenAI or Extract) │         │
│                              └─────────────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Differentiators

| Feature | Day 03 (Heavy RAG) | **Day 07 (Light RAG)** |
|---------|--------------------|-----------------------|
| Embeddings | LangChain wrapper | Direct `sentence-transformers` |
| Vector Store | ChromaDB server | **NumPy `.npy` file** |
| Keyword Search | ❌ None | **BM25Okapi** |
| Retrieval | Semantic only | **Hybrid (BM25 + Cosine)** |
| Dependencies | LangChain, ChromaDB, etc. | **NumPy, rank-bm25** |
| Index Size | ~100 MB+ | **< 5 MB** |
| Startup Time | Seconds | **Milliseconds** |

---

## 🚀 Quick Start

### 1. Install

```bash
cd day_07_light_rag
pip install -r requirements.txt
```

### 2. Prepare Data & Build Index

```bash
python -m data.prepare_data
python -m training.train
```

### 3. Evaluate

```bash
python -m training.evaluate
```

### 4. Start API

```bash
uvicorn api.main:app --reload
```

### 5. Launch Demo UI

```bash
streamlit run app/streamlit_app.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + index status |
| `POST` | `/query` | Ask a question (hybrid/semantic/bm25) |
| `POST` | `/ingest` | Add documents to the live index |
| `GET` | `/stats` | Index statistics |

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is hybrid search?", "mode": "hybrid", "top_k": 5}'
```

### Example: Ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"items": [{"id": "new-1", "title": "My Doc", "content": "New content..."}]}'
```

---

## 🐳 Docker

```bash
cd docker
docker compose up --build
```

---

## 🧪 Tests

```bash
cd day_07_light_rag
pytest tests/ -v
```

**30+ tests** covering:
- Query preprocessing & validation
- Text chunking strategies
- Data pipeline
- Embedding models
- Index save/load roundtrip
- Predictor (all 3 modes)
- Ingestion
- API endpoints (health, query, ingest, stats)
- Evaluation metrics

---

## 📁 Project Structure

```
day_07_light_rag/
├── config.py                    # Centralized settings
├── requirements.txt             # Minimal dependencies
├── README.md                    # This file
├── api/
│   ├── main.py                  # FastAPI application
│   ├── routes.py                # API endpoints
│   └── schemas.py               # Pydantic models
├── app/
│   └── streamlit_app.py         # Interactive demo
├── data/
│   └── prepare_data.py          # Document chunking pipeline
├── training/
│   ├── model.py                 # Embedding factory (local/OpenAI/fake)
│   ├── train.py                 # Build hybrid index (NumPy + BM25)
│   └── evaluate.py              # Hit-rate & MRR evaluation
├── inference/
│   ├── predictor.py             # Light RAG predictor (singleton)
│   └── preprocessing.py         # Query validation
├── docker/
│   ├── Dockerfile               # Multi-stage build
│   └── docker-compose.yml       # Service orchestration
├── tests/
│   ├── test_predictor.py        # Unit tests (20+)
│   └── test_api.py              # Integration tests (15+)
└── notebooks/
    └── exploration.ipynb        # Data exploration
```

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `EMBEDDING_PROVIDER` | `local` | `local`, `openai`, or `fake` |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `OPENAI_API_KEY` | — | Enables OpenAI embeddings + generation |
| `LIGHTRAG_TOP_K` | `5` | Default retrieval depth |
| `LIGHTRAG_CHUNK_SIZE` | `512` | Characters per chunk |
| `LIGHTRAG_BM25_WEIGHT` | `0.3` | BM25 score weight in hybrid mode |
| `LIGHTRAG_SEMANTIC_WEIGHT` | `0.7` | Semantic score weight in hybrid mode |

---

## 📊 Retrieval Modes

- **`hybrid`** (default) — Weighted fusion of BM25 + cosine similarity
- **`semantic`** — Pure dense vector search (cosine similarity)
- **`bm25`** — Pure keyword search (BM25Okapi)

---

Built with ❤️ as part of the [Daily AI Projects](https://github.com/elkhayyat17/daily-ai-projects) challenge.
