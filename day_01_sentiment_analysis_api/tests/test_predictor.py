"""
Unit Tests for Sentiment Predictor
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.preprocessing import clean_text, validate_input


class TestTextCleaning:
    """Tests for text preprocessing."""
    
    def test_clean_basic_text(self):
        result = clean_text("Hello World!")
        assert result == "Hello World!"
    
    def test_clean_extra_whitespace(self):
        result = clean_text("  Hello   World!  ")
        assert result == "Hello World!"
    
    def test_clean_removes_urls(self):
        result = clean_text("Check this http://example.com awesome!")
        assert "http" not in result
    
    def test_clean_removes_emails(self):
        result = clean_text("Contact user@email.com for info")
        assert "@" not in result
    
    def test_clean_removes_mentions(self):
        result = clean_text("Thanks @user for the help!")
        assert "@user" not in result
    
    def test_clean_keeps_hashtag_text(self):
        result = clean_text("Love this #amazing product")
        assert "amazing" in result
        assert "#" not in result
    
    def test_clean_handles_html_entities(self):
        result = clean_text("This is &amp; that")
        assert "&amp;" not in result
    
    def test_clean_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_clean_handles_non_string(self):
        assert clean_text(123) == ""


class TestInputValidation:
    """Tests for input validation."""
    
    def test_valid_input(self):
        is_valid, msg = validate_input("This is a valid text input")
        assert is_valid is True
        assert msg == ""
    
    def test_empty_input(self):
        is_valid, msg = validate_input("")
        assert is_valid is False
        assert "empty" in msg.lower()
    
    def test_too_short(self):
        is_valid, msg = validate_input("Hi")
        assert is_valid is False
        assert "3 characters" in msg
    
    def test_too_long(self):
        is_valid, msg = validate_input("x" * 5001)
        assert is_valid is False
        assert "5000" in msg
    
    def test_non_string_input(self):
        is_valid, msg = validate_input(12345)
        assert is_valid is False
    
    def test_whitespace_only(self):
        is_valid, msg = validate_input("   ")
        assert is_valid is False


class TestPredictor:
    """Tests for the SentimentPredictor class."""
    
    def test_predictor_import(self):
        from inference.predictor import SentimentPredictor
        predictor = SentimentPredictor()
        assert predictor is not None
        assert predictor._loaded is False
    
    def test_model_info(self):
        from inference.predictor import SentimentPredictor
        predictor = SentimentPredictor()
        info = predictor.model_info
        assert "model_name" in info
        assert "num_labels" in info
        assert info["num_labels"] == 3
    
    def test_predict_invalid_input(self):
        from inference.predictor import SentimentPredictor
        predictor = SentimentPredictor()
        predictor.load()
        
        result = predictor.predict("")
        assert "error" in result
    
    def test_predict_valid_input(self):
        from inference.predictor import SentimentPredictor
        predictor = SentimentPredictor()
        predictor.load()
        
        result = predictor.predict("This movie was absolutely fantastic and wonderful!")
        assert "sentiment" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
        assert "probabilities" in result
    
    def test_batch_predict(self):
        from inference.predictor import SentimentPredictor
        predictor = SentimentPredictor()
        predictor.load()
        
        texts = [
            "I love this product!",
            "This is terrible and awful.",
            "It's okay, nothing special.",
        ]
        results = predictor.predict_batch(texts)
        assert len(results) == 3
        for result in results:
            assert "sentiment" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
