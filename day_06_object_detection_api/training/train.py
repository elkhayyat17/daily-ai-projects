"""
Day 06 — Object Detection API: Training Pipeline
Download pre-trained model and optionally fine-tune on custom data.
"""

import json
import time
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from training.model import YOLODetector


def download_pretrained_model() -> dict:
    """
    Download and cache the pre-trained YOLOv8 model.

    For this project, we use the COCO-pretrained YOLOv8n (nano) model
    which is fast and lightweight — great for demo / API serving.

    Returns:
        Dict with model info and timing.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("🔍 Day 06 — Object Detection: Model Setup")
    logger.info("=" * 60)

    detector = YOLODetector()
    success = detector.load_pretrained()

    elapsed = time.time() - start_time

    stats = {
        "model_name": config.YOLO_MODEL_NAME,
        "loaded": success,
        "num_classes": config.NUM_CLASSES,
        "classes_sample": config.COCO_CLASSES[:10],
        "elapsed_seconds": round(elapsed, 2),
    }

    if success:
        logger.success(f"✅ Model ready in {elapsed:.1f}s")
        logger.info(f"   Model: {config.YOLO_MODEL_NAME}")
        logger.info(f"   Classes: {config.NUM_CLASSES} (COCO)")
        logger.info(f"   Input size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")

        # Run a quick test inference
        try:
            import numpy as np
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            results = detector.predict(test_img, conf=0.5)
            stats["test_inference"] = "passed"
            logger.success("✅ Test inference passed")
        except Exception as e:
            stats["test_inference"] = f"failed: {str(e)}"
            logger.warning(f"⚠️ Test inference failed: {e}")
    else:
        logger.error("❌ Failed to load model")

    # Save stats
    stats_path = config.ARTIFACTS_DIR / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats → {stats_path}")

    return stats


def fine_tune_model(data_yaml: str, epochs: int = 10, **kwargs) -> dict:
    """
    Fine-tune YOLOv8 on a custom dataset.

    Args:
        data_yaml: Path to the YOLO-format data.yaml file.
        epochs: Number of training epochs.

    Returns:
        Training results.
    """
    logger.info("🏋️ Fine-tuning YOLOv8 on custom data...")

    detector = YOLODetector()
    detector.load_pretrained()

    results = detector.train(
        data_yaml=data_yaml,
        epochs=epochs,
        **kwargs,
    )

    logger.success("✅ Fine-tuning complete!")
    return results


if __name__ == "__main__":
    logger.add(config.LOG_FILE, rotation="10 MB", level=config.LOG_LEVEL)
    stats = download_pretrained_model()
    print(json.dumps(stats, indent=2))
