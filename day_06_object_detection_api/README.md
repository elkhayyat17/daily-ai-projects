# 🔍 Day 06 — Object Detection API (YOLOv8)

> Upload images, detect objects in real-time — powered by YOLOv8, the state-of-the-art object detection model.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)

---

## 🏗️ Architecture

```
📤 Image Upload / URL              ⚙️ Settings
       │                               │
       ▼                               ▼
┌─────────────────┐         ┌──────────────────┐
│ Image Validator  │         │  Conf / IoU /    │
│ (format, size)   │         │  Max Detections  │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         ▼                           ▼
┌──────────────────────────────────────────────┐
│              YOLOv8 Nano Model               │
│      (80 COCO classes, 640×640 input)        │
└────────────────────┬─────────────────────────┘
                     │
           ┌─────────┴──────────┐
           ▼                    ▼
    ┌─────────────┐     ┌──────────────┐
    │  JSON API   │     │  Annotated   │
    │  Response   │     │  Image (PNG) │
    └─────────────┘     └──────────────┘
```

## ✨ Features

- 🔍 **YOLOv8 Detection** — 80 COCO object classes out of the box
- ⚡ **Real-time inference** — Nano model for fast API serving
- 📤 **Image upload** — JPG, PNG, BMP, WebP support
- 🌐 **URL detection** — Detect objects from any image URL
- 🎨 **Annotated images** — Get bounding boxes drawn on the image
- 📊 **Structured results** — JSON with bounding boxes, classes, confidence
- 🔧 **Configurable** — Adjust confidence, IoU, max detections per request
- 🌐 **REST API** — Full FastAPI backend with Swagger docs
- 🎨 **Streamlit UI** — Interactive demo with side-by-side comparison
- 🐳 **Docker ready** — One-command deployment
- ✅ **35+ tests** — Comprehensive unit and integration tests

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare sample data
python data/prepare_data.py

# 3. Download & test model
python training/train.py

# 4. Evaluate on sample images
python training/evaluate.py

# 5. Start the API
uvicorn api.main:app --reload

# 6. Launch the UI
streamlit run app/streamlit_app.py
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/api/v1/health` | Health check & model status |
| `GET` | `/api/v1/classes` | List all 80 detectable classes |
| `POST` | `/api/v1/detect` | Detect objects (file upload) |
| `POST` | `/api/v1/detect/annotate` | Detect & return annotated image |
| `POST` | `/api/v1/detect/url` | Detect objects from image URL |
| `GET` | `/api/v1/model/info` | Get model information |

### Detect Objects (Upload)

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@photo.jpg" \
  -G -d "confidence=0.3" -d "iou_threshold=0.45"
```

**Response:**
```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.9234,
      "bbox": {"x1": 120.5, "y1": 80.3, "x2": 350.2, "y2": 450.8},
      "bbox_normalized": {"x1": 0.1883, "y1": 0.1671, "x2": 0.5472, "y2": 0.9392},
      "area": 85137.5
    }
  ],
  "num_detections": 1,
  "class_counts": {"person": 1},
  "image_size": {"width": 640, "height": 480},
  "elapsed_ms": 45.2,
  "confidence_threshold": 0.3,
  "iou_threshold": 0.45
}
```

### Get Annotated Image

```bash
curl -X POST http://localhost:8000/api/v1/detect/annotate \
  -F "file=@photo.jpg" \
  --output annotated.png
```

### Detect from URL

```bash
curl -X POST http://localhost:8000/api/v1/detect/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/photo.jpg", "confidence": 0.25}'
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
| Model | YOLOv8n (Ultralytics) |
| Training Data | COCO (80 classes) |
| Framework | PyTorch + Ultralytics |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Image Processing | Pillow, OpenCV |
| Container | Docker + Docker Compose |

## 📁 Project Structure

```
day_06_object_detection_api/
├── config.py                    # Centralized configuration
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── data/
│   └── prepare_data.py          # Sample image generation
├── training/
│   ├── model.py                 # YOLODetector wrapper
│   ├── train.py                 # Model download & fine-tuning
│   └── evaluate.py              # Detection evaluation
├── inference/
│   ├── predictor.py             # ObjectDetectionPredictor (singleton)
│   └── preprocessing.py         # Image validation & processing
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
│   ├── test_predictor.py        # Unit tests (25 tests)
│   └── test_api.py              # Integration tests (20 tests)
└── notebooks/
    └── exploration.ipynb         # Data exploration notebook
```

## 🎯 Supported Object Classes (80 COCO)

<details>
<summary>Click to expand full class list</summary>

| ID | Class | ID | Class | ID | Class | ID | Class |
|----|-------|----|-------|----|-------|----|-------|
| 0 | person | 20 | elephant | 40 | wine glass | 60 | dining table |
| 1 | bicycle | 21 | bear | 41 | cup | 61 | toilet |
| 2 | car | 22 | zebra | 42 | fork | 62 | tv |
| 3 | motorcycle | 23 | giraffe | 43 | knife | 63 | laptop |
| 4 | airplane | 24 | backpack | 44 | spoon | 64 | mouse |
| 5 | bus | 25 | umbrella | 45 | bowl | 65 | remote |
| 6 | train | 26 | handbag | 46 | banana | 66 | keyboard |
| 7 | truck | 27 | tie | 47 | apple | 67 | cell phone |
| 8 | boat | 28 | suitcase | 48 | sandwich | 68 | microwave |
| 9 | traffic light | 29 | frisbee | 49 | orange | 69 | oven |
| 10 | fire hydrant | 30 | skis | 50 | broccoli | 70 | toaster |
| 11 | stop sign | 31 | snowboard | 51 | carrot | 71 | sink |
| 12 | parking meter | 32 | sports ball | 52 | hot dog | 72 | refrigerator |
| 13 | bench | 33 | kite | 53 | pizza | 73 | book |
| 14 | bird | 34 | baseball bat | 54 | donut | 74 | clock |
| 15 | cat | 35 | baseball glove | 55 | cake | 75 | vase |
| 16 | dog | 36 | skateboard | 56 | chair | 76 | scissors |
| 17 | horse | 37 | surfboard | 57 | couch | 77 | teddy bear |
| 18 | sheep | 38 | tennis racket | 58 | potted plant | 78 | hair drier |
| 19 | cow | 39 | bottle | 59 | bed | 79 | toothbrush |

</details>

---

Built with ❤️ as part of the [Daily AI Projects](https://github.com/elkhayyat17/daily-ai-projects) challenge.
