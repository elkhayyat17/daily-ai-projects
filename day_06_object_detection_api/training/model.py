"""
Day 06 — Object Detection API: Model Architecture
YOLOv8 model wrapper for training and inference.
"""

import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class YOLODetector:
    """
    YOLOv8 object detection model wrapper.

    Handles model loading, training configuration, and export.
    Supports pre-trained COCO models and custom fine-tuning.
    """

    def __init__(
        self,
        model_name: str = config.YOLO_MODEL_NAME,
        model_path: Optional[Path] = None,
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_name: Pre-trained model name (e.g., 'yolov8n.pt').
            model_path: Path to a custom-trained model (overrides model_name).
        """
        self.model_name = model_name
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.model = None
        self._loaded = False

    def load_pretrained(self) -> bool:
        """
        Load a pre-trained YOLOv8 model from Ultralytics hub.

        Returns:
            True if model was loaded successfully.
        """
        try:
            from ultralytics import YOLO

            logger.info(f"Loading pre-trained YOLOv8 model: {self.model_name}")
            self.model = YOLO(self.model_name)
            self._loaded = True

            # Save the pre-trained model to artifacts
            pretrained_path = config.MODELS_DIR / self.model_name
            if not pretrained_path.exists():
                # The model file is downloaded by ultralytics automatically
                # We just track that it's been loaded
                logger.info(f"Pre-trained model cached by ultralytics")

            logger.success(f"✅ Pre-trained model loaded: {self.model_name}")
            return True

        except ImportError:
            logger.error("ultralytics package not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to load pre-trained model: {e}")
            return False

    def load_custom(self, path: Optional[Path] = None) -> bool:
        """
        Load a custom-trained model from disk.

        Args:
            path: Path to model weights file.

        Returns:
            True if loaded successfully.
        """
        model_path = path or self.model_path

        if not model_path.exists():
            logger.warning(f"Custom model not found at {model_path}")
            return False

        try:
            from ultralytics import YOLO

            logger.info(f"Loading custom model from: {model_path}")
            self.model = YOLO(str(model_path))
            self._loaded = True
            logger.success(f"✅ Custom model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            return False

    def load(self) -> bool:
        """
        Load model — tries custom first, then falls back to pre-trained.

        Returns:
            True if any model was loaded.
        """
        # Try custom model first
        if self.model_path.exists():
            return self.load_custom()

        # Fall back to pre-trained
        return self.load_pretrained()

    def train(
        self,
        data_yaml: str,
        epochs: int = config.TRAIN_EPOCHS,
        imgsz: int = config.TRAIN_IMAGE_SIZE,
        batch: int = config.TRAIN_BATCH_SIZE,
        lr0: float = config.TRAIN_LR,
        patience: int = config.TRAIN_PATIENCE,
        workers: int = config.TRAIN_WORKERS,
        project: str = str(config.RESULTS_DIR),
        name: str = "train",
        **kwargs,
    ) -> dict:
        """
        Train / fine-tune the YOLOv8 model.

        Args:
            data_yaml: Path to YOLO data config YAML.
            epochs: Number of training epochs.
            imgsz: Training image size.
            batch: Batch size.
            lr0: Initial learning rate.
            patience: Early stopping patience.
            workers: Number of data loading workers.
            project: Results output directory.
            name: Experiment name.

        Returns:
            Training results dict.
        """
        if self.model is None:
            self.load_pretrained()

        logger.info(f"🏋️ Starting training for {epochs} epochs...")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            patience=patience,
            workers=workers,
            project=project,
            name=name,
            exist_ok=True,
            verbose=True,
            **kwargs,
        )

        # Copy best model to standard location
        best_path = Path(project) / name / "weights" / "best.pt"
        if best_path.exists():
            shutil.copy2(best_path, config.YOLO_MODEL_PATH)
            logger.success(f"✅ Best model saved to {config.YOLO_MODEL_PATH}")

        return {
            "epochs_completed": epochs,
            "best_model_path": str(config.YOLO_MODEL_PATH),
            "results_dir": str(Path(project) / name),
        }

    def evaluate(
        self,
        data_yaml: str,
        imgsz: int = config.TRAIN_IMAGE_SIZE,
        split: str = "val",
        **kwargs,
    ) -> dict:
        """
        Evaluate the model on a validation/test set.

        Args:
            data_yaml: Path to YOLO data config YAML.
            imgsz: Image size.
            split: Dataset split to evaluate on.

        Returns:
            Evaluation metrics dict.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load() first.")

        logger.info(f"📊 Evaluating model on {split} split...")

        metrics = self.model.val(
            data=data_yaml,
            imgsz=imgsz,
            split=split,
            **kwargs,
        )

        results = {
            "mAP50": float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0,
            "mAP50_95": float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0,
            "precision": float(metrics.box.mp) if hasattr(metrics.box, "mp") else 0.0,
            "recall": float(metrics.box.mr) if hasattr(metrics.box, "mr") else 0.0,
        }

        logger.info(f"  mAP@50:    {results['mAP50']:.3f}")
        logger.info(f"  mAP@50-95: {results['mAP50_95']:.3f}")
        logger.info(f"  Precision: {results['precision']:.3f}")
        logger.info(f"  Recall:    {results['recall']:.3f}")

        return results

    def predict(
        self,
        source,
        conf: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        imgsz: int = config.IMAGE_SIZE,
        max_det: int = config.MAX_DETECTIONS,
        **kwargs,
    ):
        """
        Run detection on an image or batch.

        Args:
            source: Image path, URL, numpy array, or PIL Image.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            imgsz: Input image size.
            max_det: Maximum detections per image.

        Returns:
            Ultralytics Results object.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load() first.")

        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
            **kwargs,
        )

        return results

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export model to specified format."""
        if self.model is None:
            raise ValueError("No model loaded.")
        return self.model.export(format=format, **kwargs)

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._loaded and self.model is not None

    @property
    def model_info(self) -> dict:
        """Get model information."""
        if self.model is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_name": self.model_name,
            "task": "detect",
            "num_classes": len(config.COCO_CLASSES),
        }
