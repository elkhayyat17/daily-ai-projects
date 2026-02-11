"""
Model Fine-Tuning Script
Fine-tunes DistilBERT for 3-class sentiment classification.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_data():
    """Load preprocessed data splits."""
    data_dir = config.DATA_DIR / "processed"
    
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    
    logger.info(f"📂 Loaded {len(train_df)} train and {len(val_df)} val samples")
    return train_df, val_df


def tokenize_data(df: pd.DataFrame, tokenizer):
    """Tokenize text data using the model tokenizer."""
    dataset = Dataset.from_pandas(df[["text", "sentiment"]])
    dataset = dataset.rename_column("sentiment", "labels")
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LENGTH,
        )
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    return tokenized


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision_weighted": precision_score(labels, predictions, average="weighted"),
        "recall_weighted": recall_score(labels, predictions, average="weighted"),
    }


def train():
    """Run the complete training pipeline."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Model Training Pipeline")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️  Using device: {device}")
    
    # Step 1: Load data
    train_df, val_df = load_data()
    
    # Step 2: Initialize tokenizer
    logger.info(f"📦 Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Step 3: Tokenize datasets
    logger.info("🔤 Tokenizing datasets...")
    train_dataset = tokenize_data(train_df, tokenizer)
    val_dataset = tokenize_data(val_df, tokenizer)
    
    # Step 4: Initialize model
    logger.info(f"🤖 Loading model: {config.MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.LABEL_MAP,
        label2id=config.LABEL_TO_ID,
    )
    
    # Step 5: Define training arguments
    output_dir = config.MODEL_DIR / "checkpoints"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_dir=str(config.LOGS_DIR),
        logging_steps=100,
        report_to="none",
        seed=config.SEED,
        fp16=torch.cuda.is_available(),
    )
    
    # Step 6: Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Step 7: Train!
    logger.info("🏋️ Starting training...")
    train_result = trainer.train()
    
    # Log training metrics
    logger.info(f"📊 Training Loss: {train_result.metrics['train_loss']:.4f}")
    logger.info(f"   Training Runtime: {train_result.metrics['train_runtime']:.1f}s")
    
    # Step 8: Evaluate
    logger.info("📈 Running final evaluation...")
    eval_metrics = trainer.evaluate()
    
    for key, value in eval_metrics.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.4f}")
    
    # Step 9: Save the best model
    final_model_dir = config.MODEL_DIR / "sentiment_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    logger.info(f"💾 Model saved to {final_model_dir}")
    logger.info("=" * 60)
    logger.info("✅ Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
