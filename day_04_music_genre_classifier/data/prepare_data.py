"""
data/prepare_data.py
====================
Downloads / synthesises music audio data for the Music Genre Classifier.

Real data  : GTZAN Genre Collection — 1 000 audio clips (30 s each, 10 genres).
             Because GTZAN requires authentication on most mirrors, this script
             supports three modes:

  1. **Auto-download** (marsyas mirror or kaggle)   — set GTZAN_SOURCE=auto
  2. **Manual path**  — point MUSIC_RAW_DIR at your own GTZAN folder
  3. **Synthetic demo** (default)                   — generates short
     librosa-synthesised clips so every pipeline step works out-of-the-box.

Usage
-----
    python -m data.prepare_data            # synthetic demo
    GTZAN_SOURCE=auto python -m data.prepare_data   # try auto-download
"""

from __future__ import annotations

import os
import shutil
import struct
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from config import get_settings

# ---------------------------------------------------------------------------
# Synthetic waveform generators — each "genre" has a characteristic signature
# ---------------------------------------------------------------------------

_GENRE_PARAMS: Dict[str, Dict] = {
    "blues": dict(tempo=90, harmonics=[1, 2, 3, 5], variation=0.3),
    "classical": dict(tempo=72, harmonics=[1, 2, 4, 8], variation=0.1),
    "country": dict(tempo=104, harmonics=[1, 3, 5], variation=0.25),
    "disco": dict(tempo=120, harmonics=[1, 2, 4], variation=0.2),
    "hiphop": dict(tempo=88, harmonics=[1, 2, 3], variation=0.35),
    "jazz": dict(tempo=100, harmonics=[1, 2, 3, 6, 9], variation=0.4),
    "metal": dict(tempo=160, harmonics=[1, 2, 3, 7, 9], variation=0.45),
    "pop": dict(tempo=116, harmonics=[1, 2, 4, 6], variation=0.15),
    "reggae": dict(tempo=80, harmonics=[1, 3, 5, 7], variation=0.3),
    "rock": dict(tempo=130, harmonics=[1, 2, 3, 5, 7], variation=0.38),
}


def _synthesise_clip(
    genre: str,
    sample_rate: int = 22050,
    duration: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    """Return a float32 waveform that loosely mimics a genre's spectral character."""
    rng = np.random.default_rng(seed)
    params = _GENRE_PARAMS[genre]
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Base frequency from tempo
    base_freq = params["tempo"] / 60.0 * 2  # ~2 Hz per beat
    signal = np.zeros_like(t)

    for h in params["harmonics"]:
        freq = base_freq * h * rng.uniform(0.98, 1.02)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = 1.0 / h * rng.uniform(0.9, 1.1)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)

    # Add slight noise for realism
    noise_level = params["variation"] * 0.05
    signal += rng.normal(0, noise_level, len(t))

    # Normalise to [-0.9, 0.9]
    peak = np.abs(signal).max() or 1.0
    signal = (signal / peak * 0.9).astype(np.float32)
    return signal


def _write_wav(path: Path, signal: np.ndarray, sample_rate: int = 22050) -> None:
    """Write a float32 numpy array as a 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (signal * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))


def generate_synthetic_dataset(
    raw_dir: Path,
    genres: Tuple[str, ...],
    clips_per_genre: int = 10,
    sample_rate: int = 22050,
    duration: float = 3.0,
) -> Dict[str, List[Path]]:
    """
    Generate synthetic WAV clips for each genre.

    Parameters
    ----------
    raw_dir : Path
        Root folder; sub-folders are created per genre.
    clips_per_genre : int
        Number of clips to create per genre (default 10 for demo).
    sample_rate : int
    duration : float
        Length of each clip in seconds.

    Returns
    -------
    Dict mapping genre -> list of created file paths.
    """
    created: Dict[str, List[Path]] = {}
    for genre in genres:
        genre_dir = raw_dir / genre
        genre_dir.mkdir(parents=True, exist_ok=True)
        created[genre] = []
        for i in range(clips_per_genre):
            clip = _synthesise_clip(genre, sample_rate=sample_rate, duration=duration, seed=i)
            out = genre_dir / f"{genre}.{i:05d}.wav"
            if not out.exists():
                _write_wav(out, clip, sample_rate=sample_rate)
            created[genre].append(out)
        logger.info("  Genre {:10s} — {:3d} clips ready", genre, clips_per_genre)
    return created


# ---------------------------------------------------------------------------
# GTZAN auto-download (best-effort)
# ---------------------------------------------------------------------------

def _try_download_gtzan(raw_dir: Path) -> bool:
    """Attempt to download GTZAN via kaggle CLI. Returns True on success."""
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "andradaolteanu/gtzan-dataset-music-genre-classification",
             "--unzip", "-p", str(raw_dir)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            logger.success("GTZAN downloaded via kaggle CLI")
            return True
        logger.warning("kaggle download failed: {}", result.stderr.strip())
    except FileNotFoundError:
        logger.warning("kaggle CLI not found — skipping auto-download")
    except Exception as exc:
        logger.warning("Auto-download error: {}", exc)
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prepare_data() -> Path:
    """
    Prepare training data.

    1. If raw_dir already contains per-genre sub-folders with WAV files → use as-is.
    2. If GTZAN_SOURCE=auto → try kaggle download.
    3. Fallback: generate synthetic demo clips.

    Returns the raw_dir path.
    """
    settings = get_settings()
    raw_dir = settings.raw_dir
    source = os.getenv("GTZAN_SOURCE", "synthetic").lower()

    # Check if real data already present
    existing = [
        d for d in raw_dir.iterdir()
        if d.is_dir() and d.name in settings.genres
    ] if raw_dir.exists() else []

    if existing:
        logger.info(
            "Found existing genre folders in {} — using them ({})", raw_dir, len(existing)
        )
        return raw_dir

    if source == "auto":
        logger.info("Attempting GTZAN auto-download …")
        if _try_download_gtzan(raw_dir):
            return raw_dir
        logger.info("Falling back to synthetic data generation")

    logger.info(
        "Generating synthetic demo dataset in {} …", raw_dir
    )
    generate_synthetic_dataset(
        raw_dir=raw_dir,
        genres=settings.genres,
        clips_per_genre=10,
        sample_rate=settings.sample_rate,
        duration=3.0,
    )
    logger.success(
        "Synthetic dataset ready — {} genres × 10 clips in {}", len(settings.genres), raw_dir
    )
    return raw_dir


if __name__ == "__main__":
    out = prepare_data()
    print(f"Data ready at: {out}")
