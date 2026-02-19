"""
tests/test_api.py
=================
API integration tests for the Music Genre Classifier FastAPI application.

Run with: pytest tests/test_api.py -v
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def _make_wav_bytes(duration: float = 2.0, sr: int = 22050) -> bytes:
    """Create an in-memory WAV file (440 Hz sine wave)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(signal)}h", *signal))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_model_ready(self, client):
        data = client.get("/health").json()
        assert "model_ready" in data
        assert isinstance(data["model_ready"], bool)

    def test_health_has_genres(self, client):
        data = client.get("/health").json()
        assert "genres" in data
        assert len(data["genres"]) == 10

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_genres_contain_expected(self, client):
        data = client.get("/health").json()
        expected = {"blues", "classical", "country", "disco", "hiphop",
                    "jazz", "metal", "pop", "reggae", "rock"}
        assert expected == set(data["genres"])


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_predict_valid_wav(self, client):
        wav = _make_wav_bytes(duration=2.0)
        response = client.post(
            "/predict",
            files={"file": ("clip.wav", wav, "audio/wav")},
        )
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        wav = _make_wav_bytes(duration=2.0)
        data = client.post(
            "/predict",
            files={"file": ("clip.wav", wav, "audio/wav")},
        ).json()
        assert "genre" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "duration_seconds" in data
        assert "model_ready" in data

    def test_predict_confidence_range(self, client):
        wav = _make_wav_bytes(duration=2.0)
        data = client.post(
            "/predict",
            files={"file": ("clip.wav", wav, "audio/wav")},
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_probabilities_count(self, client):
        wav = _make_wav_bytes(duration=2.0)
        data = client.post(
            "/predict",
            files={"file": ("clip.wav", wav, "audio/wav")},
        ).json()
        assert len(data["probabilities"]) == 10

    def test_predict_unsupported_format_returns_422(self, client):
        fake_pdf = b"%PDF-1.4 fake content"
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", fake_pdf, "application/pdf")},
        )
        assert response.status_code == 422

    def test_predict_corrupt_audio_returns_422(self, client):
        response = client.post(
            "/predict",
            files={"file": ("bad.wav", b"not audio", "audio/wav")},
        )
        assert response.status_code == 422

    def test_predict_duration_positive(self, client):
        wav = _make_wav_bytes(duration=2.0)
        data = client.post(
            "/predict",
            files={"file": ("clip.wav", wav, "audio/wav")},
        ).json()
        assert data["duration_seconds"] > 0

    def test_predict_no_file_returns_422(self, client):
        response = client.post("/predict")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /reload
# ---------------------------------------------------------------------------

class TestReloadEndpoint:
    def test_reload_returns_200(self, client):
        response = client.post("/reload")
        assert response.status_code == 200

    def test_reload_response_has_success(self, client):
        data = client.post("/reload").json()
        assert "success" in data
        assert isinstance(data["success"], bool)

    def test_reload_response_has_message(self, client):
        data = client.post("/reload").json()
        assert "message" in data
        assert isinstance(data["message"], str)
