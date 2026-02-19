"""
training/model.py
=================
Defines the scikit-learn pipeline that wraps feature-scaling + classification.

Supported model types (controlled via config.model_type):
  - "random_forest"  → RandomForestClassifier  (default, no GPU required)
  - "gradient_boost" → HistGradientBoostingClassifier
  - "svm"            → SVC with RBF kernel
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import get_settings


def build_pipeline(model_type: str | None = None) -> Pipeline:
    """
    Build and return an untrained sklearn Pipeline.

    Parameters
    ----------
    model_type : str | None
        Override the model type from config. One of
        "random_forest", "gradient_boost", "svm".

    Returns
    -------
    sklearn.pipeline.Pipeline
        [StandardScaler → Classifier]
    """
    settings = get_settings()
    mtype = (model_type or settings.model_type).lower()

    clf: Any
    if mtype == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=settings.n_estimators,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=settings.random_state,
            class_weight="balanced",
        )
        logger.info("Model: RandomForestClassifier (n_estimators={})", settings.n_estimators)

    elif mtype == "gradient_boost":
        clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=settings.random_state,
        )
        logger.info("Model: GradientBoostingClassifier")

    elif mtype == "svm":
        clf = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=settings.random_state,
            class_weight="balanced",
        )
        logger.info("Model: SVC (rbf)")

    else:
        raise ValueError(
            f"Unknown model_type '{mtype}'. Choose from: random_forest, gradient_boost, svm"
        )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipeline
