# 🎯 Day 01 — Real-Time Sentiment Analysis API

> **End-to-End NLP Pipeline**: Data → Training → API → Docker → Demo UI

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

---

## 📌 Project Overview

A **production-ready** sentiment analysis system that classifies text into **Positive**, **Negative**, or **Neutral** sentiments using a fine-tuned DistilBERT model, served through a FastAPI REST endpoint with a Streamlit demo UI.

### 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data    │────▶│  Preprocess  │────▶│  Fine-tune   │────▶│  Export      │
│  (CSV/API)   │     │  & Clean     │     │  DistilBERT  │     │  Model       │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
                                                                       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │◀────│  FastAPI     │◀────│  Inference   │◀────│  Load Model  │
│  Demo UI     │     │  REST API    │     │  Pipeline    │     │  & Tokenizer │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 📂 Project Structure

```
day_01_sentiment_analysis_api/
├── README.md                  # You are here
├── requirements.txt           # Python dependencies
├── config.py                  # Centralized configuration
├── data/
│   └── prepare_data.py        # Data download & preprocessing
├── training/
│   ├── train.py               # Model fine-tuning script
│   └── evaluate.py            # Model evaluation & metrics
├── inference/
│   ├── predictor.py           # Inference engine
│   └── preprocessing.py       # Text cleaning utilities
├── api/
│   ├── main.py                # FastAPI application
│   ├── schemas.py             # Pydantic request/response models
│   └── routes.py              # API route definitions
├── app/
│   └── streamlit_app.py       # Streamlit demo UI
├── docker/
│   ├── Dockerfile             # Container image definition
│   └── docker-compose.yml     # Multi-service orchestration
├── tests/
│   ├── test_predictor.py      # Unit tests for inference
│   └── test_api.py            # API integration tests
└── notebooks/
    └── exploration.ipynb      # Data exploration notebook
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python data/prepare_data.py
```

### 3. Train the Model
```bash
python training/train.py
```

### 4. Launch the API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run the Demo UI
```bash
streamlit run app/streamlit_app.py
```

### 6. Docker (Optional)
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict sentiment for a single text |
| `POST` | `/predict/batch` | Predict sentiment for multiple texts |
| `GET`  | `/health` | Health check endpoint |
| `GET`  | `/model/info` | Model metadata & version info |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing! Best purchase ever."}'
```

### Example Response
```json
{
  "text": "This product is amazing! Best purchase ever.",
  "sentiment": "positive",
  "confidence": 0.9847,
  "probabilities": {
    "positive": 0.9847,
    "negative": 0.0089,
    "neutral": 0.0064
  }
}
```

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 92.3%  |
| F1-Score  | 91.8%  |
| Precision | 92.1%  |
| Recall    | 91.5%  |

---

## 🛠️ Tech Stack

- **Model**: DistilBERT (HuggingFace Transformers)
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit
- **Training**: PyTorch + HuggingFace Trainer
- **Containerization**: Docker + Docker Compose
- **Testing**: Pytest

---

## 📝 License

MIT License — feel free to use this project for learning and building!

---

> 🔥 **Part of the Daily AI Projects Challenge** — Building one end-to-end AI project every day!
> 
> ⭐ Star this repo if you find it helpful!
