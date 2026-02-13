from __future__ import annotations

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from parent directory .env
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(_PROJECT_DIR), ".env"))

# Ensure project root is on sys.path for imports
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")

st.title("💬 RAG Chatbot")
st.caption("Ask questions grounded in your document corpus.")

use_api = st.toggle("Use API endpoint", value=True)
api_url = st.text_input("API URL", value=os.getenv("RAG_API_URL", "http://localhost:8000"))

query = st.text_input("Your question", value="What is RAG?")

if st.button("Ask"):
    if not query or len(query.strip()) < 3:
        st.warning("Please enter a question (at least 3 characters).")
    elif use_api:
        import requests

        try:
            response = requests.post(f"{api_url}/chat", json={"query": query}, timeout=30)
            if response.status_code != 200:
                st.error(f"API error: {response.text}")
            else:
                payload = response.json()
                st.success(payload["answer"])
                if payload.get("sources"):
                    st.markdown("**Sources:**")
                    for src in payload["sources"]:
                        st.write(f"- {src['id']} (score: {src['score']:.3f})")
        except requests.ConnectionError:
            st.error("Cannot connect to the API. Make sure the FastAPI server is running.")
        except Exception as exc:
            st.error(f"Request failed: {exc}")
    else:
        with st.spinner("Loading model & searching..."):
            from inference.predictor import RAGPredictor

            predictor = RAGPredictor.get_instance()
            result = predictor.predict(query)
        st.success(result.answer)
        if result.sources:
            st.markdown("**Sources:**")
            for s in result.sources:
                st.write(f"- {s.id} (score: {s.score:.3f})")
