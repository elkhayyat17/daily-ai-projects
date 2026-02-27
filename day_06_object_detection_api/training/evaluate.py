"""
Day 06 — Object Detection API: Evaluation
Evaluates detection model performance on sample images.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from training.model import YOLODetector
from data.prepare_data import prepare_sample_images, load_image_manifest


def evaluate_model(
    images_dir: Optional[Path] = None,
    conf_threshold: float = config.CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Evaluate the object detection model on sample images.

    Runs inference on each sample image and reports:
    - Per-image detection counts
    - Class distribution across all detections
    - Average confidence score
    - Inference timing

    Returns:
        Evaluation results dict.
    """
    logger.info("=" * 60)
    logger.info("📊 Day 06 — Object Detection: Evaluation")
    logger.info("=" * 60)

    # Load model
    detector = YOLODetector()
    if not detector.load():
        logger.error("No model available. Run train.py first!")
        return {}

    # Prepare images
    if images_dir is None:
        images_dir = config.RAW_DIR

    if not any(images_dir.glob("*.png")) and not any(images_dir.glob("*.jpg")):
        logger.info("No images found. Generating sample images...")
        prepare_sample_images()

    manifest = load_image_manifest()
    if not manifest:
        logger.error("No image manifest found.")
        return {}

    # Run evaluation
    results = {
        "images": [],
        "total_detections": 0,
        "class_distribution": {},
        "confidence_scores": [],
        "inference_times_ms": [],
    }

    for img_info in manifest:
        img_path = Path(img_info["path"])
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        start = time.time()

        try:
            preds = detector.predict(str(img_path), conf=conf_threshold)
            elapsed_ms = (time.time() - start) * 1000

            num_detections = 0
            image_classes = []
            image_confidences = []

            for result in preds:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    num_detections = len(boxes)
                    for box in boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        cls_name = config.COCO_CLASSES[cls_id] if cls_id < len(config.COCO_CLASSES) else f"class_{cls_id}"

                        image_classes.append(cls_name)
                        image_confidences.append(conf)

                        results["class_distribution"][cls_name] = \
                            results["class_distribution"].get(cls_name, 0) + 1

            results["images"].append({
                "filename": img_info["filename"],
                "num_detections": num_detections,
                "classes_detected": list(set(image_classes)),
                "avg_confidence": round(np.mean(image_confidences), 4) if image_confidences else 0.0,
                "inference_ms": round(elapsed_ms, 1),
            })

            results["total_detections"] += num_detections
            results["confidence_scores"].extend(image_confidences)
            results["inference_times_ms"].append(elapsed_ms)

            status = f"{num_detections} objects" if num_detections > 0 else "no detections"
            logger.info(
                f"  📸 {img_info['filename']}: {status} ({elapsed_ms:.0f}ms)"
            )

        except Exception as e:
            logger.error(f"  ❌ {img_info['filename']}: {e}")
            results["images"].append({
                "filename": img_info["filename"],
                "error": str(e),
            })

    # Aggregate metrics
    all_confs = results["confidence_scores"]
    all_times = results["inference_times_ms"]

    results["metrics"] = {
        "num_images": len(results["images"]),
        "total_detections": results["total_detections"],
        "avg_detections_per_image": round(
            results["total_detections"] / max(len(results["images"]), 1), 1
        ),
        "avg_confidence": round(np.mean(all_confs), 4) if all_confs else 0.0,
        "min_confidence": round(np.min(all_confs), 4) if all_confs else 0.0,
        "max_confidence": round(np.max(all_confs), 4) if all_confs else 0.0,
        "avg_inference_ms": round(np.mean(all_times), 1) if all_times else 0.0,
        "unique_classes": len(results["class_distribution"]),
        "top_classes": sorted(
            results["class_distribution"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10],
    }

    # Remove raw lists for JSON serialization
    del results["confidence_scores"]
    del results["inference_times_ms"]

    # Log summary
    m = results["metrics"]
    logger.info("─" * 60)
    logger.info("📊 Evaluation Summary")
    logger.info("─" * 60)
    logger.info(f"  Images evaluated:       {m['num_images']}")
    logger.info(f"  Total detections:       {m['total_detections']}")
    logger.info(f"  Avg detections/image:   {m['avg_detections_per_image']}")
    logger.info(f"  Avg confidence:         {m['avg_confidence']:.3f}")
    logger.info(f"  Avg inference time:     {m['avg_inference_ms']:.0f}ms")
    logger.info(f"  Unique classes:         {m['unique_classes']}")

    if m["top_classes"]:
        logger.info("  Top classes:")
        for cls_name, count in m["top_classes"][:5]:
            logger.info(f"    - {cls_name}: {count}")

    # Save results
    eval_path = config.ARTIFACTS_DIR / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"Saved evaluation results → {eval_path}")

    return results


if __name__ == "__main__":
    logger.add(config.LOG_FILE, rotation="10 MB", level=config.LOG_LEVEL)
    evaluate_model()
