# 📄 Day 05 — Document Q&A with Vector Database

> Upload documents, ask questions, get answers — powered by FAISS vector search and extractive QA.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)

---

## 🏗️ Architecture

```
📤 Document Upload          ❓ Question
       │                         │
       ▼                         ▼
┌─────────────┐         ┌──────────────┐
│  Doc Parser │         │   Embedding  │
│ (PDF/TXT/MD)│         │    Model     │
└──────┬──────┘         └──────┬───────┘
       │                       │
       ▼                       ▼
┌─────────────┐         ┌──────────────┐
│  Chunking   │         │ FAISS Search │◄── Vector Index
│  Pipeline   │         │  (Top-K)     │
└──────┬──────┘         └──────┬───────┘
       │                       │
       ▼                       ▼
┌─────────────┐         ┌──────────────┐
│  Embedding  │         │ Extractive   │
│  + Indexing │         │  QA Model    │
└──────┬──────┘         └──────┬───────┘
       │                       │
       ▼                       ▼
   FAISS Index            💡 Answer
```

## ✨ Features

- 📄 **Multi-format support** — TXT, PDF, Markdown, DOCX, CSV, JSON
- 🔍 **FAISS vector search** — Fast approximate nearest neighbor retrieval
- 🤖 **Extractive QA** — MiniLM model extracts precise answers from context
- 📊 **Confidence scoring** — Know how reliable each answer is
- 🌐 **REST API** — Full FastAPI backend with Swagger docs
- 🎨 **Streamlit UI** — Interactive demo interface
- 🐳 **Docker ready** — One-command deployment
- ✅ **25+ tests** — Comprehensive unit and integration tests

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare sample data & build index
python data/prepare_data.py
python training/train.py

# 3. Evaluate retrieval quality
python training/evaluate.py

# 4. Start the API
uvicorn api.main:app --reload

# 5. Launch the UI
streamlit run app/streamlit_app.py
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/api/v1/health` | Health check & system status |
| `POST` | `/api/v1/ask` | Ask a question |
| `POST` | `/api/v1/ingest` | Ingest documents (JSON) |
| `POST` | `/api/v1/upload` | Upload a document file |
| `GET` | `/api/v1/index/info` | Get index information |
| `DELETE` | `/api/v1/index` | Clear the index |

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "top_k": 5}'
```

### Ingest Documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"title": "My Doc", "content": "Document content here..."}
    ]
  }'
```

### Upload a File

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@document.pdf"
```

## 🐳 Docker

```bash
cd docker
docker-compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:8501
- Docs: http://localhost:8000/docs

## 🧪 Testing

```bash
pytest tests/ -v
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (Inner Product / Cosine) |
| QA Model | `deepset/minilm-uncased-squad2` |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Doc Parsing | PyPDF2, python-docx |
| Container | Docker + Docker Compose |

## 📁 Project Structure

```
day_05_document_qa/
├── config.py                    # Centralized configuration
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── data/
│   └── prepare_data.py          # Sample data preparation
├── training/
│   ├── model.py                 # VectorStoreBuilder (FAISS + embeddings)
│   ├── train.py                 # Indexing pipeline
│   └── evaluate.py              # Retrieval evaluation metrics
├── inference/
│   ├── predictor.py             # DocumentQAPredictor (singleton)
│   └── preprocessing.py         # Document parsing & chunking
├── api/
│   ├── main.py                  # FastAPI application
│   ├── routes.py                # API endpoints
│   └── schemas.py               # Pydantic models
├── app/
│   └── streamlit_app.py         # Interactive demo UI
├── docker/
│   ├── Dockerfile               # Multi-stage build
│   └── docker-compose.yml       # Service orchestration
├── tests/
│   ├── test_predictor.py        # Unit tests (18 tests)
│   └── test_api.py              # Integration tests (16 tests)
└── notebooks/
    └── exploration.ipynb         # Data exploration notebook
```

---

Built with ❤️ as part of the [Daily AI Projects](https://github.com/elkhayyat17/daily-ai-projects) challenge.
