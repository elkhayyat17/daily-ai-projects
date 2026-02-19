"""
app/streamlit_app.py
====================
Interactive Streamlit demo for the Music Genre Classifier.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure parent directory is on the path when running from app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🎵 Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def _get_predictor():
    from inference.predictor import MusicPredictor
    return MusicPredictor.get_instance()


def _genre_emoji(genre: str) -> str:
    mapping = {
        "blues": "🎷",
        "classical": "🎻",
        "country": "🤠",
        "disco": "🪩",
        "hiphop": "🎤",
        "jazz": "🎺",
        "metal": "🤘",
        "pop": "🎶",
        "reggae": "🌴",
        "rock": "🎸",
    }
    return mapping.get(genre, "🎵")


def _render_waveform(signal: np.ndarray, sr: int) -> None:
    """Draw a simple waveform chart using Streamlit's line_chart."""
    import pandas as pd

    # Downsample to ≤ 5 000 points for performance
    step = max(1, len(signal) // 5_000)
    times = np.arange(0, len(signal), step) / sr
    amps = signal[::step]
    df = pd.DataFrame({"Amplitude": amps}, index=times.round(3))
    st.line_chart(df, height=180)


def _render_probability_bar(probs: dict) -> None:
    """Render a horizontal bar chart for genre probabilities."""
    import pandas as pd

    sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
    labels = [f"{_genre_emoji(g)} {g.capitalize()}" for g in sorted_probs]
    values = list(sorted_probs.values())

    df = pd.DataFrame({"Probability": values}, index=labels)
    st.bar_chart(df, height=320)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎵 Music Genre Classifier")
    st.markdown(
        """
        **Day 04 of the Daily AI Project challenge**

        Classify music audio into **10 genres** using hand-crafted
        audio features (MFCC, Chroma, Mel, Spectral, Tonnetz) +
        a Random Forest classifier.

        ---

        **Supported Genres**
        """
    )
    from config import get_settings

    settings = get_settings()
    for g in settings.genres:
        st.markdown(f"- {_genre_emoji(g)} {g.capitalize()}")

    st.divider()
    st.markdown(
        "**API** running at `http://localhost:8004`\n\n"
        "[FastAPI Docs →](http://localhost:8004/docs)"
    )

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("🎵 Music Genre Classifier")
st.caption("Upload an audio clip — get a genre prediction in seconds.")

tab_predict, tab_about = st.tabs(["🎯 Predict", "ℹ️ About"])

# ---- Predict tab -----------------------------------------------------------
with tab_predict:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("Upload Audio")
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            help="Max 50 MB. Accepts WAV, MP3, OGG, FLAC, M4A.",
        )

        if uploaded:
            st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")

            with st.expander("🔍 Waveform Preview", expanded=False):
                try:
                    import librosa
                    audio_bytes = uploaded.getvalue()
                    buf = io.BytesIO(audio_bytes)
                    y, sr = librosa.load(buf, sr=22_050, mono=True, duration=10.0)
                    _render_waveform(y, sr)
                except Exception:
                    st.info("Waveform preview unavailable for this format.")

    with col_result:
        st.subheader("Prediction")

        if uploaded:
            with st.spinner("Analysing audio…"):
                try:
                    predictor = _get_predictor()
                    audio_bytes = uploaded.getvalue()
                    result = predictor.predict_from_bytes(
                        audio_bytes, filename=uploaded.name
                    )
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.stop()

            if not result.model_ready:
                st.warning(
                    "⚠️ Model has not been trained yet.\n\n"
                    "Run `python -m training.train` to train the classifier first."
                )
            else:
                emoji = _genre_emoji(result.genre)
                st.markdown(
                    f"### {emoji} **{result.genre.upper()}**"
                )
                st.metric(
                    label="Confidence",
                    value=f"{result.confidence:.1%}",
                )
                st.metric(
                    label="Audio Duration",
                    value=f"{result.duration_seconds:.1f}s",
                )

                st.markdown("#### Genre Probabilities")
                _render_probability_bar(result.probabilities)

                # Top-3 genres
                top3 = sorted(
                    result.probabilities.items(), key=lambda x: x[1], reverse=True
                )[:3]
                st.markdown("**Top-3 Candidates**")
                for rank, (genre, prob) in enumerate(top3, 1):
                    bar = "█" * int(prob * 20)
                    st.text(f"  {rank}. {_genre_emoji(genre)} {genre:<12} {bar} {prob:.1%}")
        else:
            st.info("Upload an audio file on the left to see the prediction here.")

# ---- About tab -------------------------------------------------------------
with tab_about:
    st.subheader("How It Works")
    st.markdown(
        """
        **Audio Feature Extraction (Librosa)**

        | Feature Group      | Dimensions | Description |
        |--------------------|-----------|-------------|
        | MFCC mean + std    | 80        | Timbral texture |
        | Chroma mean + std  | 24        | Pitch class energy |
        | Mel spectrogram    | 128       | Perceptual frequency bins |
        | Spectral centroid  | 2         | Brightness |
        | Spectral rolloff   | 2         | High-frequency content |
        | Spectral bandwidth | 2         | Spectral spread |
        | Zero-crossing rate | 2         | Noisiness |
        | RMS energy         | 2         | Loudness |
        | Spectral contrast  | 7         | Peak vs. valley contrast |
        | Tonnetz            | 6         | Harmonic space |
        | Tempo              | 1         | Beats per minute |
        | **Total**          | **256**   |             |

        **Model** — Random Forest (300 trees) with Standard Scaler pre-processing.

        **Training data** — [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html):
        1 000 clips × 30 s across 10 genres.
        """
    )

    st.subheader("Architecture")
    st.code(
        """
Audio File  →  Librosa Load & Resample (22 050 Hz, mono)
           →  Feature Extraction (256-dim vector)
           →  StandardScaler  →  RandomForestClassifier
           →  Genre Label + Confidence Scores
        """,
        language="text",
    )
