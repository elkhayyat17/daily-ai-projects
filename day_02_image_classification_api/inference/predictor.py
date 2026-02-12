"""
Image Classification Predictor
Production inference engine for image classification.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from training.model import ImageClassifier
from training.transforms import get_inference_transforms
from inference.preprocessing import load_image_from_bytes, load_image_from_path, validate_image


class ImagePredictor:
    """
    Production-ready image classification engine.

    Loads a fine-tuned ResNet50 and provides
    single-image and batch prediction capabilities.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(config.MODEL_DIR / "best_model.pth")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.transform = get_inference_transforms()
        self._loaded = False

    def load(self):
        """Load the trained model."""
        logger.info(f"📦 Loading model from {self.model_path}")

        try:
            self.model = ImageClassifier(freeze_backbone=False)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"✅ Model loaded on {self.device}")

        except Exception as e:
            logger.warning(f"⚠️  Could not load trained model: {e}")
            logger.info("🔄 Loading pretrained ResNet50 as fallback...")

            from torchvision import models
            self.model = ImageClassifier(pretrained=True, freeze_backbone=False)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"✅ Fallback model loaded on {self.device}")

    def predict_from_bytes(self, image_bytes: bytes, filename: str = "upload") -> dict:
        """
        Predict class from raw image bytes.

        Args:
            image_bytes: Raw image file bytes
            filename: Original filename

        Returns:
            Prediction result dictionary
        """
        # Validate
        is_valid, error_msg = validate_image(image_bytes, filename)
        if not is_valid:
            return {"error": error_msg}

        image = load_image_from_bytes(image_bytes)
        return self._predict(image, filename)

    def predict_from_path(self, image_path: str) -> dict:
        """Predict class from an image file path."""
        image = load_image_from_path(image_path)
        return self._predict(image, Path(image_path).name)

    def predict_from_pil(self, image: Image.Image, filename: str = "image") -> dict:
        """Predict class from a PIL Image object."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self._predict(image, filename)

    def _predict(self, image: Image.Image, filename: str) -> dict:
        """Core prediction logic."""
        if not self._loaded:
            self.load()

        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=-1).cpu().numpy()[0]

        # Get results
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = config.CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # Top-5 predictions
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5 = [
            {
                "class": config.CLASS_NAMES[idx],
                "confidence": round(float(probabilities[idx]), 4),
                "emoji": config.CLASS_EMOJIS.get(config.CLASS_NAMES[idx], ""),
            }
            for idx in top5_indices
        ]

        return {
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "emoji": config.CLASS_EMOJIS.get(predicted_class, ""),
            "top_5": top5,
            "all_probabilities": {
                config.CLASS_NAMES[i]: round(float(probabilities[i]), 4)
                for i in range(config.NUM_CLASSES)
            },
        }

    @property
    def model_info(self) -> dict:
        """Return model metadata."""
        return {
            "model_name": config.MODEL_NAME,
            "dataset": config.DATASET_NAME,
            "num_classes": config.NUM_CLASSES,
            "class_names": config.CLASS_NAMES,
            "input_size": config.INPUT_SIZE,
            "device": self.device,
            "loaded": self._loaded,
        }


# Global singleton
_predictor = None


def get_predictor() -> ImagePredictor:
    """Get or create the global predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = ImagePredictor()
        _predictor.load()
    return _predictor
