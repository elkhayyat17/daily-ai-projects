"""
Data Preparation Script
Downloads CIFAR-10 and prepares train/val/test splits.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def download_cifar10():
    """Download CIFAR-10 dataset using torchvision."""
    logger.info("📥 Downloading CIFAR-10 dataset...")

    raw_dir = config.DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root=str(raw_dir), train=True, download=True
    )
    test_dataset = datasets.CIFAR10(
        root=str(raw_dir), train=False, download=True
    )

    logger.info(f"✅ Downloaded {len(train_dataset)} training and {len(test_dataset)} test images")
    return train_dataset, test_dataset


def create_validation_split(train_dataset, val_ratio=0.1):
    """Split training data into train and validation sets."""
    logger.info(f"🔀 Creating validation split (ratio={val_ratio})...")

    num_total = len(train_dataset)
    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val

    generator = torch.Generator().manual_seed(config.SEED)
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [num_train, num_val], generator=generator
    )

    logger.info(f"   Train: {num_train} | Val: {num_val}")
    return train_subset, val_subset


def compute_dataset_stats(train_dataset):
    """Compute per-channel mean and std for normalization."""
    logger.info("📊 Computing dataset statistics...")

    images = np.array([np.array(img) for img, _ in train_dataset]) / 255.0
    mean = images.mean(axis=(0, 1, 2)).tolist()
    std = images.std(axis=(0, 1, 2)).tolist()

    logger.info(f"   Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    logger.info(f"   Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    return mean, std


def analyze_class_distribution(dataset, name="Dataset"):
    """Analyze and log class distribution."""
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "dataset"):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = [label for _, label in dataset]

    from collections import Counter
    dist = Counter(targets)

    logger.info(f"📊 {name} class distribution:")
    for class_idx in sorted(dist.keys()):
        class_name = config.CLASS_NAMES[class_idx]
        emoji = config.CLASS_EMOJIS.get(class_name, "")
        count = dist[class_idx]
        logger.info(f"   {emoji} {class_name:>12}: {count:>5} ({count / len(targets) * 100:.1f}%)")


def save_metadata(mean, std, train_size, val_size, test_size):
    """Save dataset metadata to JSON."""
    metadata = {
        "dataset": config.DATASET_NAME,
        "num_classes": config.NUM_CLASSES,
        "class_names": config.CLASS_NAMES,
        "splits": {
            "train": train_size,
            "val": val_size,
            "test": test_size,
        },
        "normalization": {"mean": mean, "std": std},
        "image_size": {"original": 32, "resized": config.INPUT_SIZE},
    }

    output_path = config.DATA_DIR / "metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"💾 Metadata saved to {output_path}")


def main():
    """Run the complete data preparation pipeline."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Data Preparation Pipeline")
    logger.info("=" * 60)

    # Step 1: Download
    train_dataset, test_dataset = download_cifar10()

    # Step 2: Split
    train_subset, val_subset = create_validation_split(train_dataset)

    # Step 3: Statistics
    mean, std = compute_dataset_stats(train_dataset)

    # Step 4: Class distribution
    analyze_class_distribution(train_subset, "Train")
    analyze_class_distribution(val_subset, "Validation")
    analyze_class_distribution(test_dataset, "Test")

    # Step 5: Save metadata
    save_metadata(
        mean, std,
        train_size=len(train_subset),
        val_size=len(val_subset),
        test_size=len(test_dataset),
    )

    logger.info("=" * 60)
    logger.info("✅ Data Preparation Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
