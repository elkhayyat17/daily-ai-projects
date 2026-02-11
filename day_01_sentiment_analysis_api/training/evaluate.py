"""
Model Evaluation Script
Generates detailed evaluation metrics and confusion matrix.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def evaluate_model():
    """Run comprehensive model evaluation on the test set."""
    logger.info("=" * 60)
    logger.info("📊 Starting Model Evaluation")
    logger.info("=" * 60)
    
    # Load model and tokenizer
    model_path = config.MODEL_DIR / "sentiment_model"
    
    if not model_path.exists():
        logger.error(f"❌ Model not found at {model_path}. Please train first!")
        return
    
    logger.info(f"📦 Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Load test data
    test_df = pd.read_csv(config.DATA_DIR / "processed" / "test.csv")
    logger.info(f"📂 Loaded {len(test_df)} test samples")
    
    # Run predictions
    predictions = []
    true_labels = test_df["sentiment"].tolist()
    
    logger.info("🔮 Running predictions...")
    batch_size = 32
    
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df["text"].iloc[i : i + batch_size].tolist()
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(preds.tolist())
    
    # Generate classification report
    target_names = [config.LABEL_MAP[i] for i in range(config.NUM_LABELS)]
    report = classification_report(true_labels, predictions, target_names=target_names)
    
    logger.info("\n📋 Classification Report:")
    logger.info(f"\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    logger.info("📊 Confusion Matrix:")
    logger.info(f"\n{pd.DataFrame(cm, index=target_names, columns=target_names)}")
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    logger.info(f"\n🎯 Overall Accuracy: {accuracy:.4f}")
    
    # Save results
    results_dir = config.PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "evaluation_report.txt", "w") as f:
        f.write("Sentiment Analysis Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {config.MODEL_NAME}\n")
        f.write(f"Test Samples: {len(test_df)}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(pd.DataFrame(cm, index=target_names, columns=target_names)))
    
    logger.info(f"💾 Report saved to {results_dir / 'evaluation_report.txt'}")
    logger.info("✅ Evaluation Complete!")


if __name__ == "__main__":
    evaluate_model()
