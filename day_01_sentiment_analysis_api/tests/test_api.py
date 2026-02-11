"""
API Integration Tests
Tests for FastAPI endpoints.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""
    
    def test_predict_positive(self):
        response = client.post(
            "/predict",
            json={"text": "This is absolutely amazing and wonderful!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert "probabilities" in data
    
    def test_predict_negative(self):
        response = client.post(
            "/predict",
            json={"text": "This is terrible and awful. I hate it!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
    
    def test_predict_empty_text(self):
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_text(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    def test_predict_response_structure(self):
        response = client.post(
            "/predict",
            json={"text": "This is a test sentence for validation."}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields
        assert "text" in data
        assert "sentiment" in data
        assert "confidence" in data
        assert "probabilities" in data
        
        # Check probability fields
        probs = data["probabilities"]
        assert "positive" in probs
        assert "negative" in probs
        assert "neutral" in probs
        
        # Check probabilities sum to ~1
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01


class TestBatchEndpoint:
    """Tests for the batch prediction endpoint."""
    
    def test_batch_predict(self):
        response = client.post(
            "/predict/batch",
            json={
                "texts": [
                    "I love this!",
                    "I hate this!",
                    "This is fine.",
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total" in data
        assert data["total"] == 3
    
    def test_batch_empty_list(self):
        response = client.post(
            "/predict/batch",
            json={"texts": []}
        )
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for the model info endpoint."""
    
    def test_model_info(self):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "num_labels" in data
        assert data["num_labels"] == 3
        assert "labels" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
