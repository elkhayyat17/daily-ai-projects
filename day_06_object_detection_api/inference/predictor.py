"""
Day 06 — Object Detection API: Production Inference Engine
Singleton YOLOv8 predictor for real-time object detection.
"""

import io
import threading
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from training.model import YOLODetector
from inference.preprocessing import (
    ImagePreprocessor,
    validate_confidence_threshold,
    validate_iou_threshold,
)


class ObjectDetectionPredictor:
    """
    Production-grade YOLOv8 object detection engine (Singleton).

    Features:
    - Pre-trained COCO model (80 classes) out of the box
    - Configurable confidence and IoU thresholds
    - Returns structured detection results
    - Annotated image generation
    - Thread-safe singleton pattern
    """

    _instance: Optional["ObjectDetectionPredictor"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.detector = YOLODetector()
        self._model_loaded = False
        logger.info("ObjectDetectionPredictor initialized.")

    def load(self) -> bool:
        """Load the detection model."""
        if self._model_loaded:
            return True

        try:
            success = self.detector.load()
            if success:
                self._model_loaded = True
                logger.success("✅ Object detection model loaded and ready.")
            else:
                logger.warning("⚠️ Could not load model.")
            return success
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def detect(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.IOU_THRESHOLD,
        max_detections: int = config.MAX_DETECTIONS,
    ) -> dict:
        """
        Run object detection on an image.

        Args:
            image: PIL Image, numpy array, file path, or URL.
            confidence: Minimum confidence threshold (0-1).
            iou_threshold: IoU threshold for NMS (0-1).
            max_detections: Maximum number of detections to return.

        Returns:
            Dict with detections, counts, and timing info.
        """
        start_time = time.time()

        # Validate thresholds
        confidence = validate_confidence_threshold(confidence)
        iou_threshold = validate_iou_threshold(iou_threshold)

        if not self._model_loaded:
            return {
                "detections": [],
                "num_detections": 0,
                "image_size": {"width": 0, "height": 0},
                "elapsed_ms": 0,
                "model_loaded": False,
                "error": "Model not loaded. Please wait for initialization.",
            }

        # Get image dimensions
        img_w, img_h = 0, 0
        if isinstance(image, Image.Image):
            img_w, img_h = image.size
        elif isinstance(image, np.ndarray):
            img_h, img_w = image.shape[:2]

        # Run detection
        try:
            results = self.detector.predict(
                source=image,
                conf=confidence,
                iou=iou_threshold,
                max_det=max_detections,
            )
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                "detections": [],
                "num_detections": 0,
                "image_size": {"width": img_w, "height": img_h},
                "elapsed_ms": round((time.time() - start_time) * 1000, 1),
                "model_loaded": True,
                "error": str(e),
            }

        # Parse results
        detections = []
        for result in results:
            if img_w == 0 and hasattr(result, "orig_shape"):
                img_h, img_w = result.orig_shape

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                cls_name = (
                    config.COCO_CLASSES[cls_id]
                    if cls_id < len(config.COCO_CLASSES)
                    else f"class_{cls_id}"
                )

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    },
                    "bbox_normalized": {
                        "x1": round(x1 / max(img_w, 1), 4),
                        "y1": round(y1 / max(img_h, 1), 4),
                        "x2": round(x2 / max(img_w, 1), 4),
                        "y2": round(y2 / max(img_h, 1), 4),
                    },
                    "area": round((x2 - x1) * (y2 - y1), 1),
                })

        # Sort by confidence
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        elapsed = (time.time() - start_time) * 1000

        # Class summary
        class_counts = {}
        for det in detections:
            name = det["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

        return {
            "detections": detections,
            "num_detections": len(detections),
            "class_counts": class_counts,
            "image_size": {"width": img_w, "height": img_h},
            "elapsed_ms": round(elapsed, 1),
            "model_loaded": True,
            "confidence_threshold": confidence,
            "iou_threshold": iou_threshold,
        }

    def detect_and_annotate(
        self,
        image: Union[Image.Image, np.ndarray],
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.IOU_THRESHOLD,
        max_detections: int = config.MAX_DETECTIONS,
        line_width: int = 3,
        font_size: int = 16,
    ) -> tuple[dict, bytes]:
        """
        Detect objects and return annotated image.

        Args:
            image: PIL Image or numpy array.
            confidence: Minimum confidence threshold.
            iou_threshold: IoU threshold for NMS.
            max_detections: Max detections.
            line_width: Bounding box line width.
            font_size: Label font size.

        Returns:
            Tuple of (detection results dict, annotated PNG image bytes).
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()

        # Run detection
        result = self.detect(
            image=pil_image,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        # Draw annotations
        annotated = self._draw_detections(
            pil_image,
            result["detections"],
            line_width=line_width,
            font_size=font_size,
        )

        # Convert to PNG bytes
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return result, png_bytes

    @staticmethod
    def _draw_detections(
        image: Image.Image,
        detections: list[dict],
        line_width: int = 3,
        font_size: int = 16,
    ) -> Image.Image:
        """Draw bounding boxes and labels on the image."""
        draw = ImageDraw.Draw(image)

        # Color palette (20 distinct colors)
        colors = [
            "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
            "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
            "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
            "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
        ]

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            cls_id = det["class_id"]
            color = colors[cls_id % len(colors)]

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Draw label
            label = f"{det['class_name']} {det['confidence']:.0%}"
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # Label background
            label_y = max(y1 - text_h - 4, 0)
            draw.rectangle(
                [x1, label_y, x1 + text_w + 6, label_y + text_h + 4],
                fill=color,
            )
            draw.text((x1 + 3, label_y + 2), label, fill="white", font=font)

        return image

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model_loaded

    @property
    def status(self) -> dict:
        """Get system status."""
        return {
            "model_loaded": self._model_loaded,
            "model_name": config.YOLO_MODEL_NAME,
            "num_classes": config.NUM_CLASSES,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "iou_threshold": config.IOU_THRESHOLD,
            "image_size": config.IMAGE_SIZE,
        }

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None
