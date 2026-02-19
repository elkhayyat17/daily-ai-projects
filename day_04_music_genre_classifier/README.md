# 🎵 Day 04 — Music Genre Classifier

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)
![Librosa](https://img.shields.io/badge/Librosa-Audio_ML-orange?logo=soundcloud)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Random_Forest-yellow?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo_UI-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Upload any audio file. Get the music genre instantly.**
> Built with Librosa feature extraction + Random Forest classification.

---

## 🏗️ Architecture

```
Audio File (.wav / .mp3 / .ogg / .flac / .m4a)
        │
        ▼
┌───────────────────────────────────────────────────┐
│           Librosa Audio Processing                │
│                                                   │
│  ┌──────────┐  ┌────────┐  ┌───────────────────┐ │
│  │  Load &  │  │ Clip / │  │  Feature Extract  │ │
│  │ Resample │→ │  Pad   │→ │  (256-dim vector) │ │
│  │ 22050 Hz │  │ 30 sec │  │                   │ │
│  └──────────┘  └────────┘  └───────────────────┘ │
└───────────────────────────┬───────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  StandardScaler         │
              │  +  RandomForest (300)  │
              └────────────┬────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │  Genre + Confidence Score  │
              │  + Full Probability Dist.  │
              └────────────────────────────┘
```

---

## 🎼 Feature Engineering

| Feature Group          | Dims | Description                        |
|------------------------|------|------------------------------------|
| MFCC mean              |  40  | Timbral texture (mean per coeff)   |
| MFCC std               |  40  | Timbral variance                   |
| Chroma STFT mean       |  12  | Pitch class energy distribution    |
| Chroma STFT std        |  12  | Pitch class variance               |
| Mel spectrogram mean   | 128  | Perceptual frequency representation|
| Spectral centroid      |   2  | Brightness (mean + std)            |
| Spectral rolloff       |   2  | High-frequency content             |
| Spectral bandwidth     |   2  | Spectral spread                    |
| Zero-crossing rate     |   2  | Signal noisiness                   |
| RMS energy             |   2  | Loudness envelope                  |
| Spectral contrast      |   7  | Peak vs. valley spectral contrast  |
| Tonnetz                |   6  | Harmonic/tonal space               |
| Tempo                  |   1  | BPM from beat tracking             |
| **Total**              | **256** |                                 |

---

## 🎸 Supported Genres (GTZAN-style)

| # | Genre     | Emoji |
|---|-----------|-------|
| 1 | Blues     | 🎷    |
| 2 | Classical | 🎻    |
| 3 | Country   | 🤠    |
| 4 | Disco     | 🪩    |
| 5 | Hip-Hop   | 🎤    |
| 6 | Jazz      | 🎺    |
| 7 | Metal     | 🤘    |
| 8 | Pop       | 🎶    |
| 9 | Reggae    | 🌴    |
|10 | Rock      | 🎸    |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd day_04_music_genre_classifier
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Option A: synthetic demo data (default, no download needed)
python -m data.prepare_data

# Option B: real GTZAN dataset (requires kaggle CLI)
GTZAN_SOURCE=auto python -m data.prepare_data

# Option C: manual GTZAN
# Download from http://marsyas.info/downloads/datasets.html
# Extract into data/raw/ so structure is data/raw/blues/*.wav etc.
```

### 3. Train the Model

```bash
python -m training.train
# or choose a different model
python -m training.train --model svm
python -m training.train --model gradient_boost
```

### 4. Evaluate

```bash
python -m training.evaluate
# Generates artifacts/confusion_matrix.png, feature_importance.png
```

### 5. Run the API

```bash
uvicorn api.main:app --reload --port 8004
# → http://localhost:8004/docs
```

### 6. Run the Streamlit UI

```bash
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

---

## 🐳 Docker

```bash
# Build & run the full stack
cd docker
docker-compose up --build

# API  → http://localhost:8004/docs
# UI   → http://localhost:8501
```

---

## 🌐 API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_ready": true,
  "version": "0.1.0",
  "genres": ["blues", "classical", "country", "disco", "hiphop",
             "jazz", "metal", "pop", "reggae", "rock"]
}
```

### `POST /predict`

Upload an audio file (multipart/form-data):

```bash
curl -X POST http://localhost:8004/predict \
  -F "file=@my_song.mp3"
```

**Response:**
```json
{
  "genre": "jazz",
  "confidence": 0.82,
  "probabilities": {
    "blues": 0.04, "classical": 0.02, "country": 0.01,
    "disco": 0.03, "hiphop": 0.02, "jazz": 0.82,
    "metal": 0.01, "pop": 0.02, "reggae": 0.01, "rock": 0.02
  },
  "duration_seconds": 30.0,
  "model_ready": true
}
```

### `POST /reload`

Hot-reload the model from disk without restarting the server:

```bash
curl -X POST http://localhost:8004/reload
```

---

## 🧪 Tests

```bash
pytest tests/ -v
# 29+ tests across predictor, preprocessing, and API
```

---

## 📁 Project Structure

```
day_04_music_genre_classifier/
├── config.py                  ← Centralised settings
├── requirements.txt
├── api/
│   ├── main.py                ← FastAPI app factory + lifespan
│   ├── routes.py              ← /health, /predict, /reload
│   └── schemas.py             ← Pydantic models
├── app/
│   └── streamlit_app.py       ← Interactive demo UI
├── data/
│   └── prepare_data.py        ← Data download / synthetic generator
├── docker/
│   ├── Dockerfile             ← Multi-stage build
│   └── docker-compose.yml
├── inference/
│   ├── predictor.py           ← Singleton inference engine
│   └── preprocessing.py       ← Audio validation & loading
├── notebooks/
│   └── exploration.ipynb      ← Feature analysis & visualisation
├── tests/
│   ├── test_predictor.py      ← Unit tests (20+)
│   └── test_api.py            ← API integration tests (15+)
└── training/
    ├── model.py               ← sklearn Pipeline factory
    ├── train.py               ← Full training loop
    └── evaluate.py            ← Metrics, confusion matrix, plots
```

---

## 📊 Performance

> Trained on GTZAN (1 000 clips × 30 s):

| Metric     | Score |
|------------|-------|
| Accuracy   | ~85%  |
| Macro F1   | ~84%  |
| Weighted F1| ~85%  |

> *Results vary slightly by random seed and data split.*

---

## ⚙️ Environment Variables

| Variable              | Default           | Description                    |
|-----------------------|-------------------|--------------------------------|
| `MUSIC_SAMPLE_RATE`   | `22050`           | Audio resampling rate (Hz)     |
| `MUSIC_DURATION`      | `30`              | Clip duration for feature ext. |
| `MUSIC_N_MFCC`        | `40`              | Number of MFCC coefficients    |
| `MUSIC_MODEL_TYPE`    | `random_forest`   | `random_forest`/`svm`/`gradient_boost` |
| `MUSIC_N_ESTIMATORS`  | `300`             | RF trees (or GB estimators)    |
| `MUSIC_API_PORT`      | `8004`            | FastAPI port                   |
| `GTZAN_SOURCE`        | `synthetic`       | `synthetic` or `auto` (kaggle) |

---

*Part of the [Daily AI Projects](https://github.com/elkhayyat17/daily-ai-projects) challenge* 🔥
