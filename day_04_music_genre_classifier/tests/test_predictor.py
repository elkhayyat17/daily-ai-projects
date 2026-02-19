"""
tests/test_predictor.py
=======================
Unit tests for inference/predictor.py and inference/preprocessing.py.

Run with: pytest tests/test_predictor.py -v
"""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from config import get_settings
from inference.preprocessing import (
    AudioValidationError,
    load_audio_from_bytes,
    load_audio_from_path,
    validate_extension,
    validate_file_size,
)
from inference.predictor import MusicPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration: float = 2.0, sr: int = 22050) -> bytes:
    """Create a minimal valid WAV file (440 Hz sine wave) in memory."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(signal)}h", *signal))
    return buf.getvalue()


def _write_wav_file(path: Path, duration: float = 2.0, sr: int = 22050) -> None:
    """Write a test WAV file to disk."""
    data = _make_wav_bytes(duration=duration, sr=sr)
    path.write_bytes(data)


# ---------------------------------------------------------------------------
# validate_extension
# ---------------------------------------------------------------------------

class TestValidateExtension:
    def test_wav_accepted(self):
        assert validate_extension("clip.wav") == ".wav"

    def test_mp3_accepted(self):
        assert validate_extension("track.MP3") == ".mp3"

    def test_ogg_accepted(self):
        assert validate_extension("audio.OGG") == ".ogg"

    def test_flac_accepted(self):
        assert validate_extension("sound.flac") == ".flac"

    def test_m4a_accepted(self):
        assert validate_extension("music.m4a") == ".m4a"

    def test_txt_rejected(self):
        with pytest.raises(AudioValidationError, match="Unsupported file type"):
            validate_extension("document.txt")

    def test_pdf_rejected(self):
        with pytest.raises(AudioValidationError):
            validate_extension("report.pdf")

    def test_no_extension_rejected(self):
        with pytest.raises(AudioValidationError):
            validate_extension("noextension")


# ---------------------------------------------------------------------------
# validate_file_size
# ---------------------------------------------------------------------------

class TestValidateFileSize:
    def test_small_file_ok(self):
        # 1 KB — should pass
        validate_file_size(b"x" * 1_024, max_mb=50)

    def test_large_file_rejected(self):
        big = b"x" * (51 * 1_024 * 1_024)  # 51 MB
        with pytest.raises(AudioValidationError, match="too large"):
            validate_file_size(big, max_mb=50)

    def test_exactly_at_limit_passes(self):
        at_limit = b"x" * (50 * 1_024 * 1_024)
        validate_file_size(at_limit, max_mb=50)  # should not raise


# ---------------------------------------------------------------------------
# load_audio_from_bytes
# ---------------------------------------------------------------------------

class TestLoadAudioFromBytes:
    def test_valid_wav(self):
        data = _make_wav_bytes(duration=2.0)
        audio = load_audio_from_bytes(data, "test.wav")
        assert audio.signal.dtype == np.float32
        assert audio.sample_rate == get_settings().sample_rate
        assert audio.duration_seconds > 0

    def test_too_short_raises(self):
        data = _make_wav_bytes(duration=0.1)  # < 1 second
        with pytest.raises(AudioValidationError, match="too short"):
            load_audio_from_bytes(data, "short.wav")

    def test_bad_extension_raises(self):
        with pytest.raises(AudioValidationError):
            load_audio_from_bytes(b"fake", "file.xyz")

    def test_corrupt_bytes_raises(self):
        with pytest.raises(AudioValidationError):
            load_audio_from_bytes(b"not audio data at all", "corrupt.wav")


# ---------------------------------------------------------------------------
# load_audio_from_path
# ---------------------------------------------------------------------------

class TestLoadAudioFromPath:
    def test_valid_wav_file(self, tmp_path):
        fpath = tmp_path / "test.wav"
        _write_wav_file(fpath, duration=2.0)
        audio = load_audio_from_path(fpath)
        assert audio.signal.ndim == 1
        assert len(audio.signal) > 0
        assert audio.duration_seconds > 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(AudioValidationError):
            load_audio_from_path(tmp_path / "nonexistent.wav")

    def test_too_short_file_raises(self, tmp_path):
        fpath = tmp_path / "tiny.wav"
        _write_wav_file(fpath, duration=0.2)
        with pytest.raises(AudioValidationError, match="too short"):
            load_audio_from_path(fpath)


# ---------------------------------------------------------------------------
# MusicPredictor
# ---------------------------------------------------------------------------

class TestMusicPredictor:
    def test_singleton_pattern(self):
        p1 = MusicPredictor.get_instance()
        p2 = MusicPredictor.get_instance()
        assert p1 is p2

    def test_predict_from_bytes_fallback(self):
        """When model is not trained, predict_from_bytes should return fallback."""
        predictor = MusicPredictor.get_instance()
        data = _make_wav_bytes(duration=2.0)
        result = predictor.predict_from_bytes(data, "test.wav")
        # Either model_ready=True (if trained) or fallback mode
        assert isinstance(result.genre, str)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.probabilities, dict)
        assert result.duration_seconds > 0

    def test_predict_from_bytes_probabilities_sum(self):
        """Probabilities should sum to ~1.0 when model is ready."""
        predictor = MusicPredictor.get_instance()
        if not predictor.is_ready:
            pytest.skip("Model not trained — skipping probability test")
        data = _make_wav_bytes(duration=2.0)
        result = predictor.predict_from_bytes(data, "test.wav")
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-4

    def test_predict_from_bytes_genre_in_known_genres(self):
        """Predicted genre must be one of the known genres."""
        predictor = MusicPredictor.get_instance()
        data = _make_wav_bytes(duration=2.0)
        result = predictor.predict_from_bytes(data, "test.wav")
        settings = get_settings()
        if result.model_ready:
            assert result.genre in settings.genres
        else:
            assert result.genre == "unknown"

    def test_reload_returns_bool(self):
        predictor = MusicPredictor.get_instance()
        result = predictor.reload()
        assert isinstance(result, bool)

    def test_predict_from_path(self, tmp_path):
        fpath = tmp_path / "sample.wav"
        _write_wav_file(fpath, duration=2.0)
        predictor = MusicPredictor.get_instance()
        result = predictor.predict_from_path(fpath)
        assert isinstance(result.genre, str)
        assert result.duration_seconds > 0
