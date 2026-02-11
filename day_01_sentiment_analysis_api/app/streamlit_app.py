"""
Streamlit Demo UI
Interactive web interface for sentiment analysis.
"""

import sys
from pathlib import Path
import streamlit as st
import requests
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎯 Sentiment Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .sentiment-positive {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .confidence-bar {
        height: 8px; border-radius: 4px; margin-top: 10px;
    }
    .stTextArea textarea { font-size: 16px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    
    api_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="URL of the Sentiment Analysis API"
    )
    
    st.divider()
    
    st.markdown("### 📖 About")
    st.markdown("""
    This demo uses a **fine-tuned DistilBERT** model 
    to classify text sentiment into:
    - 🟢 **Positive**
    - 🔴 **Negative**  
    - 🟣 **Neutral**
    """)
    
    st.divider()
    
    st.markdown("### 📝 Sample Texts")
    if st.button("Load positive example"):
        st.session_state.input_text = "This is absolutely amazing! I love every single thing about this product. Best purchase I've ever made!"
    if st.button("Load negative example"):
        st.session_state.input_text = "Terrible experience. The product broke after one day and customer service was unhelpful and rude."
    if st.button("Load neutral example"):
        st.session_state.input_text = "The product arrived on time. It works as described, however there are both pros and cons to consider."

# ──────────────────────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────────────────────
st.title("🎯 Real-Time Sentiment Analyzer")
st.markdown("*Powered by DistilBERT & FastAPI*")
st.divider()

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    default_text = st.session_state.get("input_text", "")
    text_input = st.text_area(
        "Enter text to analyze:",
        value=default_text,
        height=150,
        placeholder="Type or paste any text here to analyze its sentiment...",
    )

with col2:
    st.markdown("### 🔧 Options")
    use_api = st.toggle("Use API (vs direct)", value=False, 
                         help="Toggle between API calls and direct model inference")
    show_probs = st.checkbox("Show probabilities", value=True)
    
    analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────
# Prediction Logic
# ──────────────────────────────────────────────────────────────
def predict_via_api(text: str, base_url: str) -> dict:
    """Call the FastAPI endpoint."""
    response = requests.post(
        f"{base_url}/predict",
        json={"text": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def predict_direct(text: str) -> dict:
    """Run prediction directly without API."""
    try:
        from inference.predictor import get_predictor
        predictor = get_predictor()
        return predictor.predict(text)
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────
# Results Display
# ──────────────────────────────────────────────────────────────
if analyze_btn and text_input.strip():
    with st.spinner("Analyzing sentiment..."):
        try:
            if use_api:
                result = predict_via_api(text_input, api_url)
            else:
                result = predict_direct(text_input)
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                # Display results
                st.divider()
                
                sentiment = result["sentiment"]
                confidence = result["confidence"]
                
                # Sentiment card
                emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                emoji = emoji_map.get(sentiment, "🤔")
                
                st.markdown(
                    f'<div class="sentiment-{sentiment}">'
                    f'{emoji} {sentiment.upper()} — {confidence:.1%} confidence'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                
                st.markdown("")
                
                # Probability bars
                if show_probs and "probabilities" in result:
                    st.markdown("### 📊 Probability Distribution")
                    
                    probs = result["probabilities"]
                    
                    col_pos, col_neg, col_neu = st.columns(3)
                    
                    with col_pos:
                        st.metric("🟢 Positive", f"{probs.get('positive', 0):.1%}")
                        st.progress(probs.get("positive", 0))
                    
                    with col_neg:
                        st.metric("🔴 Negative", f"{probs.get('negative', 0):.1%}")
                        st.progress(probs.get("negative", 0))
                    
                    with col_neu:
                        st.metric("🟣 Neutral", f"{probs.get('neutral', 0):.1%}")
                        st.progress(probs.get("neutral", 0))
                
                # Raw JSON
                with st.expander("🔧 Raw API Response"):
                    st.json(result)
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the API server is running!")
            st.info("💡 Start the API with: `uvicorn api.main:app --reload`")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif analyze_btn:
    st.warning("⚠️ Please enter some text to analyze.")

# ──────────────────────────────────────────────────────────────
# Batch Analysis Section
# ──────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📦 Batch Analysis")

batch_input = st.text_area(
    "Enter multiple texts (one per line):",
    height=120,
    placeholder="Line 1: I love this product!\nLine 2: This is terrible.\nLine 3: It's average.",
)

if st.button("🔍 Analyze Batch", use_container_width=True):
    lines = [line.strip() for line in batch_input.strip().split("\n") if line.strip()]
    
    if lines:
        with st.spinner(f"Analyzing {len(lines)} texts..."):
            results = []
            for line in lines:
                try:
                    if use_api:
                        result = predict_via_api(line, api_url)
                    else:
                        result = predict_direct(line)
                    results.append(result)
                except Exception as e:
                    results.append({"text": line, "sentiment": "error", "confidence": 0, "error": str(e)})
            
            # Display as table
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Text": r.get("text", "")[:80] + ("..." if len(r.get("text", "")) > 80 else ""),
                    "Sentiment": r.get("sentiment", "error").upper(),
                    "Confidence": f"{r.get('confidence', 0):.1%}",
                }
                for r in results
            ])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Please enter at least one text.")

# Footer
st.divider()
st.markdown(
    "<center><sub>Built with ❤️ using DistilBERT, FastAPI & Streamlit | "
    "Day 01 of Daily AI Projects</sub></center>",
    unsafe_allow_html=True,
)
