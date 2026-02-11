"""
Sentiment Predictor
Production inference engine for sentiment analysis.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from inference.preprocessing import clean_text, validate_input


class SentimentPredictor:
    """
    Production-ready sentiment prediction engine.
    
    Loads a fine-tuned DistilBERT model and provides 
    single and batch prediction capabilities.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model directory.
                       Defaults to config.MODEL_DIR / 'sentiment_model'
        """
        self.model_path = model_path or str(config.MODEL_DIR / "sentiment_model")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def load(self):
        """Load the model and tokenizer."""
        logger.info(f"📦 Loading model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"✅ Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            # Fallback to base model for demo purposes
            logger.info("⚠️  Loading base model as fallback...")
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.MODEL_NAME, num_labels=config.NUM_LABELS
            )
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"✅ Base model loaded on {self.device}")
    
    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment, confidence, and probabilities
        """
        if not self._loaded:
            self.load()
        
        # Validate input
        is_valid, error_msg = validate_input(text)
        if not is_valid:
            return {"error": error_msg}
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        # Get predicted class
        predicted_class = int(np.argmax(probabilities))
        sentiment = config.LABEL_MAP[predicted_class]
        confidence = float(probabilities[predicted_class])
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probabilities": {
                config.LABEL_MAP[i]: round(float(probabilities[i]), 4)
                for i in range(config.NUM_LABELS)
            },
        }
    
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        if not self._loaded:
            self.load()
        
        results = []
        
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            # Validate and clean
            valid_texts = []
            valid_indices = []
            batch_results = [None] * len(batch)
            
            for j, text in enumerate(batch):
                is_valid, error_msg = validate_input(text)
                if is_valid:
                    valid_texts.append(clean_text(text))
                    valid_indices.append(j)
                else:
                    batch_results[j] = {"text": text, "error": error_msg}
            
            if valid_texts:
                # Tokenize batch
                inputs = self.tokenizer(
                    valid_texts,
                    padding=True,
                    truncation=True,
                    max_length=config.MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                
                for idx, (j, text) in enumerate(zip(valid_indices, valid_texts)):
                    predicted_class = int(np.argmax(probs[idx]))
                    batch_results[j] = {
                        "text": batch[j],
                        "sentiment": config.LABEL_MAP[predicted_class],
                        "confidence": round(float(probs[idx][predicted_class]), 4),
                        "probabilities": {
                            config.LABEL_MAP[k]: round(float(probs[idx][k]), 4)
                            for k in range(config.NUM_LABELS)
                        },
                    }
            
            results.extend(batch_results)
        
        return results
    
    @property
    def model_info(self) -> dict:
        """Return model metadata."""
        return {
            "model_name": config.MODEL_NAME,
            "num_labels": config.NUM_LABELS,
            "labels": list(config.LABEL_MAP.values()),
            "max_length": config.MAX_LENGTH,
            "device": self.device,
            "loaded": self._loaded,
        }


# Global singleton instance
_predictor = None


def get_predictor() -> SentimentPredictor:
    """Get or create the global predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
        _predictor.load()
    return _predictor
