"""
training/train.py
=================
Full training pipeline:
  1. Prepare raw audio data (synthetic or GTZAN)
  2. Extract audio features with librosa for every clip
  3. Persist the feature matrix to CSV for inspection
  4. Train the sklearn pipeline
  5. Save the trained model artefact with joblib

Usage
-----
    python -m training.train
    python -m training.train --model svm
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import cross_val_score, train_test_split

from config import get_settings
from data.prepare_data import prepare_data
from training.model import build_pipeline

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_features(file_path: Path, settings) -> np.ndarray | None:
    """
    Extract a rich audio feature vector from a WAV/MP3/OGG/FLAC file.

    Feature set (total ≈ 193 floats):
      - MFCC mean × n_mfcc
      - MFCC std  × n_mfcc
      - Chroma STFT mean × n_chroma
      - Chroma STFT std  × n_chroma
      - Mel spectrogram mean × 128
      - Spectral centroid  mean, std
      - Spectral rolloff   mean, std
      - Spectral bandwidth mean, std
      - Zero-crossing rate mean, std
      - RMS energy         mean, std
      - Spectral contrast  mean (7 bands)
      - Tonnetz            mean (6 dims)
      - Tempo              (1 value)
    """
    try:
        import librosa  # lazy import

        y, sr = librosa.load(str(file_path), sr=settings.sample_rate, mono=True)

        # Clip / pad to target duration
        target_len = settings.sample_rate * settings.duration
        if len(y) > target_len:
            # Take the centre segment
            start = (len(y) - target_len) // 2
            y = y[start : start + target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))

        feats: List[float] = []

        # ---- MFCC ----
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=settings.n_mfcc,
            n_fft=settings.n_fft, hop_length=settings.hop_length,
        )
        feats.extend(mfcc.mean(axis=1).tolist())
        feats.extend(mfcc.std(axis=1).tolist())

        # ---- Chroma ----
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_chroma=settings.n_chroma,
            n_fft=settings.n_fft, hop_length=settings.hop_length,
        )
        feats.extend(chroma.mean(axis=1).tolist())
        feats.extend(chroma.std(axis=1).tolist())

        # ---- Mel spectrogram (compressed to mean) ----
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=settings.n_mel,
            n_fft=settings.n_fft, hop_length=settings.hop_length,
        )
        mel_db = librosa.power_to_db(mel)
        feats.extend(mel_db.mean(axis=1).tolist())

        # ---- Spectral centroid ----
        cent = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=settings.n_fft, hop_length=settings.hop_length,
        )
        feats.extend([cent.mean(), cent.std()])

        # ---- Spectral rolloff ----
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=settings.n_fft, hop_length=settings.hop_length,
        )
        feats.extend([rolloff.mean(), rolloff.std()])

        # ---- Spectral bandwidth ----
        bw = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=settings.n_fft, hop_length=settings.hop_length,
        )
        feats.extend([bw.mean(), bw.std()])

        # ---- Zero-crossing rate ----
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=settings.hop_length)
        feats.extend([zcr.mean(), zcr.std()])

        # ---- RMS energy ----
        rms = librosa.feature.rms(y=y, hop_length=settings.hop_length)
        feats.extend([rms.mean(), rms.std()])

        # ---- Spectral contrast ----
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=settings.n_fft, hop_length=settings.hop_length,
        )
        feats.extend(contrast.mean(axis=1).tolist())

        # ---- Tonnetz ----
        harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        feats.extend(tonnetz.mean(axis=1).tolist())

        # ---- Tempo ----
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats.append(float(np.atleast_1d(tempo)[0]))

        return np.array(feats, dtype=np.float32)

    except Exception as exc:
        logger.warning("Failed to extract features from {}: {}", file_path, exc)
        return None


def build_feature_matrix(
    raw_dir: Path,
    genres: Tuple[str, ...],
    settings,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk raw_dir for genre sub-folders, extract features, return (X, y) arrays.
    """
    X_rows: List[np.ndarray] = []
    y_labels: List[int] = []
    genre_index = {g: i for i, g in enumerate(genres)}
    total = 0

    for genre in genres:
        genre_dir = raw_dir / genre
        if not genre_dir.exists():
            logger.warning("Genre folder not found: {}", genre_dir)
            continue

        files = sorted(
            f for f in genre_dir.iterdir()
            if f.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac", ".m4a")
        )
        logger.info("  Extracting features: {:10s}  ({} files)", genre, len(files))

        for fpath in files:
            feats = extract_features(fpath, settings)
            if feats is not None:
                X_rows.append(feats)
                y_labels.append(genre_index[genre])
                total += 1

    if not X_rows:
        raise RuntimeError(
            f"No features extracted from {raw_dir}. "
            "Run data/prepare_data.py first."
        )

    logger.info("Total samples extracted: {}", total)
    return np.array(X_rows, dtype=np.float32), np.array(y_labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(model_type: str | None = None) -> Path:
    """
    Full training pipeline.

    Returns the path to the saved model artefact.
    """
    settings = get_settings()
    t0 = time.time()

    # 1. Prepare data
    logger.info("Step 1/4 — Preparing data …")
    raw_dir = prepare_data()

    # 2. Extract features
    logger.info("Step 2/4 — Extracting audio features …")
    X, y = build_feature_matrix(raw_dir, settings.genres, settings)

    # 3. Persist features for inspection
    logger.info("Step 3/4 — Saving feature matrix …")
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    feat_names = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_names)
    df["genre"] = [settings.genres[i] for i in y]
    df.to_csv(settings.features_path, index=False)
    logger.info("Feature matrix saved → {}", settings.features_path)

    # 4. Train / evaluate
    logger.info("Step 4/4 — Training {} pipeline …", model_type or settings.model_type)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size,
        random_state=settings.random_state, stratify=y,
    )

    pipeline = build_pipeline(model_type)
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    logger.info("Train accuracy: {:.4f}", train_acc)
    logger.info("Test  accuracy: {:.4f}", test_acc)

    # Cross-validation on full dataset
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    logger.info(
        "5-fold CV accuracy: {:.4f} ± {:.4f}",
        cv_scores.mean(), cv_scores.std(),
    )

    # 5. Save model
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.model_path)
    logger.success(
        "Model saved → {}  (elapsed {:.1f}s)", settings.model_path, time.time() - t0
    )
    return settings.model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Music Genre Classifier")
    parser.add_argument(
        "--model",
        choices=["random_forest", "gradient_boost", "svm"],
        default=None,
        help="Override model type from config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(model_type=args.model)
