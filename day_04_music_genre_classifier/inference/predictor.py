"""
inference/predictor.py
======================
Singleton inference engine for the Music Genre Classifier.

Usage
-----
    from inference.predictor import MusicPredictor

    predictor = MusicPredictor.get_instance()
    result = predictor.predict_from_bytes(audio_bytes, filename="clip.wav")
    print(result.genre, result.confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from loguru import logger

from config import get_settings
from inference.preprocessing import AudioInput, AudioValidationError, load_audio_from_bytes, load_audio_from_path


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GenrePrediction:
    """Single-file genre prediction result."""

    genre: str                          # Predicted genre label
    confidence: float                   # Probability for the top genre
    probabilities: Dict[str, float]     # Probability per genre
    duration_seconds: float             # Audio duration that was analysed
    model_ready: bool = True            # False when model is not trained yet


# ---------------------------------------------------------------------------
# Feature extraction (reuses the same logic as training/train.py)
# ---------------------------------------------------------------------------

def _extract_features(audio: AudioInput, settings) -> np.ndarray:
    """
    Extract the same feature vector used during training.
    Raises AudioValidationError on failure.
    """
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required") from exc

    y = audio.signal
    sr = audio.sample_rate

    # Clip / pad to training duration
    target_len = settings.sample_rate * settings.duration
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    feats: List[float] = []

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=settings.n_mfcc,
        n_fft=settings.n_fft, hop_length=settings.hop_length,
    )
    feats.extend(mfcc.mean(axis=1).tolist())
    feats.extend(mfcc.std(axis=1).tolist())

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_chroma=settings.n_chroma,
        n_fft=settings.n_fft, hop_length=settings.hop_length,
    )
    feats.extend(chroma.mean(axis=1).tolist())
    feats.extend(chroma.std(axis=1).tolist())

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=settings.n_mel,
        n_fft=settings.n_fft, hop_length=settings.hop_length,
    )
    mel_db = librosa.power_to_db(mel)
    feats.extend(mel_db.mean(axis=1).tolist())

    # Spectral centroid
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=settings.n_fft, hop_length=settings.hop_length,
    )
    feats.extend([cent.mean(), cent.std()])

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=settings.n_fft, hop_length=settings.hop_length,
    )
    feats.extend([rolloff.mean(), rolloff.std()])

    # Spectral bandwidth
    bw = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=settings.n_fft, hop_length=settings.hop_length,
    )
    feats.extend([bw.mean(), bw.std()])

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=settings.hop_length)
    feats.extend([zcr.mean(), zcr.std()])

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=settings.hop_length)
    feats.extend([rms.mean(), rms.std()])

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=settings.n_fft, hop_length=settings.hop_length,
    )
    feats.extend(contrast.mean(axis=1).tolist())

    # Tonnetz
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    feats.extend(tonnetz.mean(axis=1).tolist())

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats.append(float(np.atleast_1d(tempo)[0]))

    return np.array(feats, dtype=np.float32).reshape(1, -1)


# ---------------------------------------------------------------------------
# Predictor singleton
# ---------------------------------------------------------------------------

class MusicPredictor:
    """
    Singleton inference engine.

    Loads the joblib model on first instantiation.
    Gracefully degrades when the model artefact is absent.
    """

    _instance: Optional["MusicPredictor"] = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self._pipeline = None
        self._load_model()

    @classmethod
    def get_instance(cls) -> "MusicPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if not self.settings.model_path.exists():
            logger.warning(
                "Model not found at {}. Running in fallback mode. "
                "Run training/train.py to train the model.",
                self.settings.model_path,
            )
            return
        try:
            import joblib
            self._pipeline = joblib.load(self.settings.model_path)
            logger.info("Model loaded from {}", self.settings.model_path)
        except Exception as exc:
            logger.error("Failed to load model: {}", exc)

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    def reload(self) -> bool:
        """Reload the model from disk (useful after re-training)."""
        self._pipeline = None
        self._load_model()
        return self.is_ready

    # ------------------------------------------------------------------
    # Internal prediction
    # ------------------------------------------------------------------

    def _predict_audio(self, audio: AudioInput) -> GenrePrediction:
        if not self.is_ready:
            return GenrePrediction(
                genre="unknown",
                confidence=0.0,
                probabilities={g: 0.0 for g in self.settings.genres},
                duration_seconds=audio.duration_seconds,
                model_ready=False,
            )

        feats = _extract_features(audio, self.settings)
        proba = self._pipeline.predict_proba(feats)[0]
        idx = int(np.argmax(proba))
        genre = self.settings.genres[idx]

        prob_dict = {
            self.settings.genres[i]: float(proba[i])
            for i in range(len(self.settings.genres))
        }

        logger.debug(
            "Predicted: {} ({:.1%}) | duration: {:.2f}s",
            genre, proba[idx], audio.duration_seconds,
        )

        return GenrePrediction(
            genre=genre,
            confidence=float(proba[idx]),
            probabilities=prob_dict,
            duration_seconds=audio.duration_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_from_path(self, file_path: Union[str, Path]) -> GenrePrediction:
        """Predict the genre of an audio file on disk."""
        audio = load_audio_from_path(file_path)
        return self._predict_audio(audio)

    def predict_from_bytes(
        self,
        data: bytes,
        filename: str = "upload.wav",
    ) -> GenrePrediction:
        """Predict genre from raw audio bytes (HTTP upload)."""
        audio = load_audio_from_bytes(data, filename)
        return self._predict_audio(audio)
