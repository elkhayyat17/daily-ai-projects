"""
Data Preparation Script
Downloads and preprocesses sentiment analysis datasets.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def download_dataset() -> pd.DataFrame:
    """Download the SST-2 sentiment dataset from HuggingFace."""
    logger.info("📥 Downloading SST-2 dataset from HuggingFace...")
    
    dataset = load_dataset("glue", "sst2")
    
    # Convert to DataFrame
    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])
    
    # SST-2 has binary labels (0=negative, 1=positive)
    # We'll map them and add synthetic neutral examples
    train_df = train_df.rename(columns={"sentence": "text", "label": "sentiment"})
    val_df = val_df.rename(columns={"sentence": "text", "label": "sentiment"})
    
    logger.info(f"✅ Downloaded {len(train_df)} training and {len(val_df)} validation samples")
    return train_df, val_df


def add_neutral_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a 3-class dataset by identifying neutral examples.
    Texts with mixed signals or moderate sentiment become neutral.
    """
    neutral_keywords = [
        "however", "although", "but also", "mixed", "okay", "fine",
        "average", "neither", "moderate", "so-so", "decent"
    ]
    
    # Mark some examples as neutral based on keyword heuristics
    def detect_neutral(text):
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in neutral_keywords)
    
    # Remap: 0->negative(0), 1->positive(2), detected_neutral->neutral(1)
    df["sentiment"] = df["sentiment"].map({0: 0, 1: 2})
    
    # Convert some to neutral
    neutral_mask = df["text"].apply(detect_neutral)
    df.loc[neutral_mask, "sentiment"] = 1
    
    return df


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove very short texts
    if len(text.split()) < 3:
        return ""
    
    return text.strip()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and filtering."""
    logger.info("🧹 Cleaning and preprocessing text data...")
    
    df["text"] = df["text"].apply(clean_text)
    
    # Remove empty texts
    df = df[df["text"] != ""].reset_index(drop=True)
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info(f"   Removed {initial_len - len(df)} duplicates")
    
    return df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save processed data splits to CSV."""
    output_dir = config.DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    logger.info(f"💾 Saved data splits to {output_dir}")
    logger.info(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Print class distribution
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = df["sentiment"].value_counts().to_dict()
        labels = {v: k for k, v in config.LABEL_MAP.items()}
        dist_str = " | ".join(
            f"{config.LABEL_MAP.get(k, k)}: {v}" for k, v in sorted(dist.items())
        )
        logger.info(f"   {name} distribution → {dist_str}")


def main():
    """Run the complete data preparation pipeline."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Data Preparation Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Download dataset
    train_df, val_df = download_dataset()
    
    # Step 2: Create 3-class labels
    train_df = add_neutral_class(train_df)
    val_df = add_neutral_class(val_df)
    
    # Step 3: Clean text
    train_df = preprocess_data(train_df)
    val_df = preprocess_data(val_df)
    
    # Step 4: Create test split from training data
    train_df, test_df = train_test_split(
        train_df, test_size=0.1, random_state=config.SEED, stratify=train_df["sentiment"]
    )
    
    # Step 5: Save
    save_splits(train_df, val_df, test_df)
    
    logger.info("=" * 60)
    logger.info("✅ Data Preparation Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
