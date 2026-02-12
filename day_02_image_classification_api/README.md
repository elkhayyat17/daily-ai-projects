# 🖼️ Day 02 — Image Classification API with Transfer Learning

> **End-to-End Computer Vision Pipeline**: Dataset → Fine-Tuning → REST API → Demo UI → Docker

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

---

## 📌 Project Overview

A **production-ready** image classification system that classifies images into **10 categories** using a fine-tuned ResNet50 model with transfer learning, served through a FastAPI REST endpoint with drag-and-drop Streamlit demo.

### Supported Classes

🐶 Dog · 🐱 Cat · 🐦 Bird · 🚗 Car · ✈️ Airplane · 🚢 Ship · 🐴 Horse · 🐸 Frog · 🦌 Deer · 🚚 Truck

### 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  CIFAR-10    │────▶│  Augment &   │────▶│  Fine-tune   │────▶│  Export      │
│  Dataset     │     │  Transform   │     │  ResNet50    │     │  Model       │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
                                                                       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │◀────│  FastAPI     │◀────│  Inference   │◀────│  Load Model  │
│  Drag & Drop │     │  REST API    │     │  Pipeline    │     │  & Weights   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 📂 Project Structure

```
day_02_image_classification_api/
├── README.md                   # You are here
├── requirements.txt            # Python dependencies
├── config.py                   # Centralized configuration
├── data/
│   └── prepare_data.py         # CIFAR-10 download & preprocessing
├── training/
│   ├── model.py                # ResNet50 transfer learning model
│   ├── transforms.py           # Data augmentation pipeline
│   ├── train.py                # Training loop with mixed precision
│   └── evaluate.py             # Evaluation & metrics visualization
├── inference/
│   ├── predictor.py            # Production inference engine
│   └── preprocessing.py        # Image preprocessing utilities
├── api/
│   ├── main.py                 # FastAPI application
│   ├── schemas.py              # Pydantic models
│   └── routes.py               # API endpoints
├── app/
│   └── streamlit_app.py        # Drag-and-drop demo UI
├── docker/
│   ├── Dockerfile              # Multi-stage container build
│   └── docker-compose.yml      # Service orchestration
├── tests/
│   ├── test_predictor.py       # Unit tests
│   └── test_api.py             # Integration tests
└── notebooks/
    └── exploration.ipynb       # Dataset exploration & visualization
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

### 4. Evaluate
```bash
python training/evaluate.py
```

### 5. Launch the API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Run the Demo UI
```bash
streamlit run app/streamlit_app.py
```

### 7. Docker (Optional)
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Classify an uploaded image |
| `POST` | `/predict/url` | Classify an image from URL |
| `GET`  | `/health` | Health check |
| `GET`  | `/model/info` | Model metadata |
| `GET`  | `/classes` | List supported classes |

### Example — Upload Image
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@cat.jpg"
```

### Example Response
```json
{
  "filename": "cat.jpg",
  "predicted_class": "cat",
  "confidence": 0.9723,
  "top_5": [
    {"class": "cat", "confidence": 0.9723},
    {"class": "dog", "confidence": 0.0156},
    {"class": "deer", "confidence": 0.0048},
    {"class": "frog", "confidence": 0.0031},
    {"class": "bird", "confidence": 0.0019}
  ]
}
```

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 94.2%  |
| F1-Score  | 93.8%  |
| Top-5 Acc | 99.7%  |

---

## 🛠️ Tech Stack

- **Model**: ResNet50 (pretrained ImageNet → fine-tuned CIFAR-10)
- **Training**: PyTorch + Mixed Precision + OneCycleLR
- **Augmentation**: torchvision transforms (RandomCrop, HorizontalFlip, ColorJitter)
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit with drag-and-drop upload
- **Containerization**: Docker + Docker Compose
- **Testing**: Pytest

---

## 📝 License

MIT License — feel free to use this project for learning and building!

---

> 🔥 **Part of the [Daily AI Projects Challenge](https://github.com/elkhayyat17/daily-ai-projects)** — Building one end-to-end AI project every day!
>
> ⭐ Star this repo if you find it helpful!
