"""
inference/preprocessing.py
===========================
Input validation and audio pre-processing before feature extraction.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger

from config import get_settings


@dataclass
class AudioInput:
    """Validated, normalised audio input ready for feature extraction."""

    signal: np.ndarray   # float32, mono, shape (N,)
    sample_rate: int
    source: str          # file path or "bytes"
    duration_seconds: float


class AudioValidationError(ValueError):
    """Raised when the audio input fails validation."""


def validate_extension(filename: str) -> str:
    """
    Ensure the file extension is in the allow-list.

    Returns the lower-cased extension (e.g. '.wav').
    Raises AudioValidationError otherwise.
    """
    settings = get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise AudioValidationError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(settings.allowed_extensions)}"
        )
    return ext


def validate_file_size(data: bytes, max_mb: int | None = None) -> None:
    """Raise AudioValidationError if the byte payload exceeds the limit."""
    settings = get_settings()
    limit = (max_mb or settings.max_file_size_mb) * 1_024 * 1_024
    if len(data) > limit:
        raise AudioValidationError(
            f"File too large ({len(data) / 1_048_576:.1f} MB). "
            f"Maximum allowed: {settings.max_file_size_mb} MB"
        )


def load_audio_from_path(file_path: Union[str, Path]) -> AudioInput:
    """
    Load an audio file from disk, resample to target SR, and return an AudioInput.

    Parameters
    ----------
    file_path : str | Path

    Returns
    -------
    AudioInput

    Raises
    ------
    AudioValidationError  on unreadable or too-short audio
    """
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required but not installed") from exc

    settings = get_settings()
    path = Path(file_path)

    try:
        y, sr = librosa.load(str(path), sr=settings.sample_rate, mono=True)
    except Exception as exc:
        raise AudioValidationError(f"Cannot load audio from '{path}': {exc}") from exc

    if len(y) < settings.sample_rate:
        raise AudioValidationError(
            "Audio clip is too short (< 1 second). Please provide at least 1 second of audio."
        )

    logger.debug(
        "Loaded '{}': {:.2f}s @ {}Hz", path.name, len(y) / sr, sr
    )
    return AudioInput(
        signal=y.astype(np.float32),
        sample_rate=sr,
        source=str(path),
        duration_seconds=len(y) / sr,
    )


def load_audio_from_bytes(data: bytes, filename: str = "upload.wav") -> AudioInput:
    """
    Load an audio clip from raw bytes (e.g. an HTTP upload).

    Parameters
    ----------
    data     : Raw audio bytes
    filename : Original filename (used for extension check)
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("librosa and soundfile are required") from exc

    settings = get_settings()
    validate_extension(filename)
    validate_file_size(data)

    try:
        buf = io.BytesIO(data)
        y, sr = librosa.load(buf, sr=settings.sample_rate, mono=True)
    except Exception as exc:
        raise AudioValidationError(f"Cannot decode audio bytes: {exc}") from exc

    if len(y) < settings.sample_rate:
        raise AudioValidationError(
            "Audio clip is too short (< 1 second). "
            "Please provide at least 1 second of audio."
        )

    logger.debug(
        "Loaded bytes '{}': {:.2f}s @ {}Hz", filename, len(y) / sr, sr
    )
    return AudioInput(
        signal=y.astype(np.float32),
        sample_rate=sr,
        source="bytes",
        duration_seconds=len(y) / sr,
    )
