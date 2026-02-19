from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root or parent directory
_this_dir = Path(__file__).resolve().parent
for _candidate in [_this_dir / ".env", _this_dir.parent / ".env"]:
    if _candidate.exists():
        load_dotenv(_candidate)
        break

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


@dataclass(frozen=True)
class Settings:
    """Centralised configuration for the Music Genre Classifier."""

    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("MUSIC_BASE_DIR", str(Path(__file__).resolve().parent))
        )
    )
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    model_path: Path = field(init=False)
    features_path: Path = field(init=False)

    project_name: str = field(
        default_factory=lambda: os.getenv(
            "MUSIC_PROJECT_NAME", "Day 04 - Music Genre Classifier"
        )
    )
    version: str = field(default_factory=lambda: os.getenv("MUSIC_VERSION", "0.1.0"))

    # Audio processing
    sample_rate: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_SAMPLE_RATE", "22050"))
    )
    duration: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_DURATION", "30"))
    )
    n_mfcc: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_N_MFCC", "40"))
    )
    n_fft: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_N_FFT", "2048"))
    )
    hop_length: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_HOP_LENGTH", "512"))
    )
    n_chroma: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_N_CHROMA", "12"))
    )
    n_mel: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_N_MEL", "128"))
    )

    # Model
    model_type: str = field(
        default_factory=lambda: os.getenv("MUSIC_MODEL_TYPE", "random_forest")
    )
    n_estimators: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_N_ESTIMATORS", "300"))
    )
    test_size: float = field(
        default_factory=lambda: float(os.getenv("MUSIC_TEST_SIZE", "0.2"))
    )
    random_state: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_RANDOM_STATE", "42"))
    )

    # API
    api_host: str = field(
        default_factory=lambda: os.getenv("MUSIC_API_HOST", "0.0.0.0")
    )
    api_port: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_API_PORT", "8004"))
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("MUSIC_LOG_LEVEL", "INFO")
    )

    # Upload limits
    max_file_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MUSIC_MAX_FILE_SIZE_MB", "50"))
    )
    allowed_extensions: tuple = field(
        default_factory=lambda: (".wav", ".mp3", ".ogg", ".flac", ".m4a")
    )

    genres: tuple = field(default_factory=lambda: tuple(GENRES))

    def __post_init__(self) -> None:
        data_dir = Path(
            os.getenv("MUSIC_DATA_DIR", str(self.base_dir / "data"))
        )
        raw_dir = Path(
            os.getenv("MUSIC_RAW_DIR", str(data_dir / "raw"))
        )
        processed_dir = Path(
            os.getenv("MUSIC_PROCESSED_DIR", str(data_dir / "processed"))
        )
        artifacts_dir = Path(
            os.getenv("MUSIC_ARTIFACTS_DIR", str(self.base_dir / "artifacts"))
        )
        model_path = artifacts_dir / "music_genre_classifier.joblib"
        features_path = processed_dir / "features.csv"

        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "raw_dir", raw_dir)
        object.__setattr__(self, "processed_dir", processed_dir)
        object.__setattr__(self, "artifacts_dir", artifacts_dir)
        object.__setattr__(self, "model_path", model_path)
        object.__setattr__(self, "features_path", features_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
