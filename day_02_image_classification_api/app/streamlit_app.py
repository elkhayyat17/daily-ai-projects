"""
Streamlit Demo — Image Classification
Interactive drag-and-drop image classifier with live visualizations.
"""

import sys
import io
import requests
from pathlib import Path
from PIL import Image

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="🖼️ Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-top: 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .big-emoji {
        font-size: 4rem;
    }
    .confidence-high { color: #27ae60; }
    .confidence-mid { color: #f39c12; }
    .confidence-low { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🖼️ Image Classifier</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Powered by ResNet50 Transfer Learning — '
    'Classify images into 10 categories</p>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** ResNet50 (ImageNet → CIFAR-10)
    **Classes:** 10 categories
    **Input:** Any image (auto-resized)
    """)

    st.divider()
    st.subheader("📋 Supported Classes")
    for name in config.CLASS_NAMES:
        emoji = config.CLASS_EMOJIS.get(name, "")
        st.write(f"{emoji} {name.capitalize()}")

    st.divider()
    st.subheader("⚙️ Settings")
    input_mode = st.radio(
        "Input Method",
        ["📁 Upload Image", "🔗 Image URL", "📸 Camera"],
        index=0,
    )
    show_top_k = st.slider("Show Top-K Predictions", 1, 10, 5)

    st.divider()
    st.caption("Day 02 — Daily AI Projects Challenge")

# ── Initialize Predictor ─────────────────────────────────────
@st.cache_resource
def load_predictor():
    """Load the predictor once and cache it."""
    from inference.predictor import ImagePredictor
    predictor = ImagePredictor()
    predictor.load()
    return predictor

predictor = load_predictor()

# ── Main Content ─────────────────────────────────────────────
image = None

if "📁 Upload" in input_mode:
    uploaded_file = st.file_uploader(
        "Drop an image here or click to upload",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        filename = uploaded_file.name

elif "🔗 Image URL" in input_mode:
    url = st.text_input(
        "Enter image URL",
        placeholder="https://example.com/image.jpg",
    )
    if url:
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            filename = url.split("/")[-1].split("?")[0] or "url_image"
        except Exception as e:
            st.error(f"❌ Failed to load image: {e}")

elif "📸 Camera" in input_mode:
    camera_input = st.camera_input("Take a photo")
    if camera_input:
        image = Image.open(camera_input).convert("RGB")
        filename = "camera_capture.jpg"

# ── Prediction ───────────────────────────────────────────────
if image is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_container_width=True)
        st.caption(f"Size: {image.width} × {image.height}")

    with col2:
        with st.spinner("🔮 Classifying..."):
            result = predictor.predict_from_pil(image, filename)

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            # Main prediction
            st.subheader("🎯 Prediction")

            emoji = result["emoji"]
            pred_class = result["predicted_class"]
            confidence = result["confidence"]

            st.markdown(f"""
            <div class="prediction-box">
                <div class="big-emoji">{emoji}</div>
                <h2>{pred_class.upper()}</h2>
                <h3>{confidence * 100:.1f}% confidence</h3>
            </div>
            """, unsafe_allow_html=True)

            # Top-K bar chart
            st.subheader(f"📊 Top-{show_top_k} Predictions")
            top_k = result["top_5"][:show_top_k]

            for pred in top_k:
                pclass = pred["class"]
                pconf = pred["confidence"]
                pemoji = pred.get("emoji", "")
                st.write(f"{pemoji} **{pclass.capitalize()}**")
                st.progress(pconf, text=f"{pconf * 100:.1f}%")

    # Full probability breakdown
    with st.expander("📈 All Class Probabilities", expanded=False):
        import pandas as pd

        probs = result["all_probabilities"]
        df = pd.DataFrame(
            [(config.CLASS_EMOJIS.get(k, "") + " " + k.capitalize(), v)
             for k, v in sorted(probs.items(), key=lambda x: -x[1])],
            columns=["Class", "Probability"],
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Upload an image, paste a URL, or use your camera to get started!")

    # Show example predictions
    st.subheader("🌟 Example Classifications")
    cols = st.columns(5)
    examples = ["airplane", "automobile", "cat", "dog", "ship"]
    for col, cls in zip(cols, examples):
        with col:
            emoji = config.CLASS_EMOJIS.get(cls, "")
            st.markdown(f"<div style='text-align: center; font-size: 3rem'>{emoji}</div>",
                       unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center'><b>{cls.capitalize()}</b></div>",
                       unsafe_allow_html=True)
