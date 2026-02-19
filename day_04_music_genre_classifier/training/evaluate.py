"""
training/evaluate.py
====================
Load the trained model + feature matrix and produce a comprehensive evaluation report:
  - Accuracy, precision, recall, F1 (macro & weighted)
  - Per-class classification report
  - Confusion matrix heat-map  → artifacts/confusion_matrix.png
  - Genre probability distribution plot → artifacts/genre_distribution.png
  - Feature importance bar chart → artifacts/feature_importance.png (RF only)

Usage
-----
    python -m training.evaluate
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from config import get_settings


def _load_data(settings) -> tuple[np.ndarray, np.ndarray]:
    """Load the persisted feature CSV and return (X, y)."""
    if not settings.features_path.exists():
        logger.error(
            "Feature matrix not found at {}. Run training/train.py first.",
            settings.features_path,
        )
        sys.exit(1)

    df = pd.read_csv(settings.features_path)
    genre_index = {g: i for i, g in enumerate(settings.genres)}
    y = df["genre"].map(genre_index).values
    X = df.drop(columns=["genre"]).values.astype(np.float32)
    return X, y


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Genre", fontsize=13)
    ax.set_ylabel("True Genre", fontsize=13)
    ax.set_title("Music Genre Classifier — Confusion Matrix", fontsize=14, pad=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → {}", out_path)


def _plot_genre_distribution(y: np.ndarray, genres: tuple, out_path: Path) -> None:
    counts = pd.Series(y).map({i: g for i, g in enumerate(genres)}).value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(counts)))  # type: ignore[attr-defined]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_xlabel("Genre", fontsize=13)
    ax.set_ylabel("Number of Clips", fontsize=13)
    ax.set_title("Genre Distribution in Dataset", fontsize=14, pad=12)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Distribution plot saved → {}", out_path)


def _plot_feature_importance(pipeline, out_path: Path, top_n: int = 20) -> None:
    """Only works for Random Forest / tree-based models."""
    try:
        clf = pipeline.named_steps["clf"]
        importances = clf.feature_importances_
    except AttributeError:
        logger.info("Feature importance not available for this model type — skipping")
        return

    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(top_n), importances[indices], color="steelblue", edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f"f{i}" for i in indices], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Feature Index", fontsize=13)
    ax.set_ylabel("Importance (mean impurity decrease)", fontsize=13)
    ax.set_title(f"Top-{top_n} Feature Importances", fontsize=14, pad=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance chart saved → {}", out_path)


def evaluate() -> dict:
    """
    Full evaluation pass.  Returns a metrics dictionary.
    """
    settings = get_settings()

    if not settings.model_path.exists():
        logger.error(
            "Model not found at {}. Run training/train.py first.",
            settings.model_path,
        )
        sys.exit(1)

    pipeline = joblib.load(settings.model_path)
    logger.info("Model loaded from {}", settings.model_path)

    X, y = _load_data(settings)
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=list(settings.genres),
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    logger.info("=" * 50)
    logger.info("Test Accuracy : {:.4f}", acc)
    logger.info(
        "Macro F1      : {:.4f}", report["macro avg"]["f1-score"]
    )
    logger.info(
        "Weighted F1   : {:.4f}", report["weighted avg"]["f1-score"]
    )
    logger.info("=" * 50)
    logger.info("\n{}", classification_report(y_test, y_pred, target_names=list(settings.genres)))

    # Plots
    _plot_confusion_matrix(
        cm, list(settings.genres),
        settings.artifacts_dir / "confusion_matrix.png",
    )
    _plot_genre_distribution(
        y, settings.genres,
        settings.artifacts_dir / "genre_distribution.png",
    )
    _plot_feature_importance(
        pipeline,
        settings.artifacts_dir / "feature_importance.png",
    )

    return {
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": report,
    }


if __name__ == "__main__":
    evaluate()
