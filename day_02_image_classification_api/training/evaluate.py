"""
Model Evaluation Script
Generates detailed metrics, confusion matrix, and per-class analysis.
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from training.model import load_model
from training.transforms import get_val_transforms


def evaluate():
    """Run comprehensive model evaluation."""
    logger.info("=" * 60)
    logger.info("📊 Starting Model Evaluation")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model_path = config.MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        logger.error(f"❌ Model not found at {model_path}. Please train first!")
        return

    model = load_model(str(model_path), device)

    # Load test data
    raw_dir = config.DATA_DIR / "raw"
    test_dataset = datasets.CIFAR10(
        root=str(raw_dir), train=False, download=True,
        transform=get_val_transforms(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4,
    )

    logger.info(f"📂 Test set: {len(test_dataset)} images")

    # Run predictions
    all_preds = []
    all_labels = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            preds = outputs.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification report
    report = classification_report(
        all_labels, all_preds, target_names=config.CLASS_NAMES, digits=4
    )
    logger.info(f"\n📋 Classification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Save results directory
    results_dir = config.PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — ResNet50 on CIFAR-10", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    logger.info(f"📊 Confusion matrix saved to {results_dir / 'confusion_matrix.png'}")

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    logger.info("\n🎯 Per-Class Accuracy:")
    for i, name in enumerate(config.CLASS_NAMES):
        emoji = config.CLASS_EMOJIS.get(name, "")
        logger.info(f"   {emoji} {name:>12}: {per_class_acc[i]:.1f}%")

    # Plot per-class accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, config.NUM_CLASSES))
    bars = ax.bar(config.CLASS_NAMES, per_class_acc, color=colors, edgecolor="black")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy — ResNet50 on CIFAR-10", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "per_class_accuracy.png", dpi=150)

    # Plot training history if available
    history_path = config.LOGS_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curves
        axes[0].plot(history["train_loss"], label="Train", linewidth=2)
        axes[0].plot(history["val_loss"], label="Validation", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curves", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[1].plot(history["train_acc"], label="Train", linewidth=2)
        axes[1].plot(history["val_acc"], label="Validation", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Accuracy Curves", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate
        axes[2].plot(history["lr"], linewidth=2, color="green")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("Learning Rate Schedule", fontweight="bold")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(results_dir / "training_curves.png", dpi=150)
        logger.info(f"📈 Training curves saved to {results_dir / 'training_curves.png'}")

    # Overall accuracy
    overall_acc = (all_preds == all_labels).mean() * 100

    # Top-5 accuracy
    top5_correct = 0
    for i in range(len(all_labels)):
        top5 = np.argsort(all_probs[i])[-5:]
        if all_labels[i] in top5:
            top5_correct += 1
    top5_acc = top5_correct / len(all_labels) * 100

    # Save text report
    with open(results_dir / "evaluation_report.txt", "w") as f:
        f.write("Image Classification Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {config.MODEL_NAME}\n")
        f.write(f"Dataset: {config.DATASET_NAME}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n")
        f.write(f"Overall Accuracy: {overall_acc:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    logger.info(f"\n🏆 Overall Accuracy: {overall_acc:.2f}%")
    logger.info(f"🏆 Top-5 Accuracy: {top5_acc:.2f}%")
    logger.info("✅ Evaluation Complete!")


if __name__ == "__main__":
    evaluate()
