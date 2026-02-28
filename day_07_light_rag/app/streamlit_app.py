"""Streamlit demo for Light RAG."""

from __future__ import annotations

import os
import sys

import streamlit as st
from dotenv import load_dotenv

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(_PROJECT_DIR), ".env"))

if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Light RAG", page_icon="🪶", layout="wide")

st.title("🪶 Light RAG — Lightweight Retrieval-Augmented Generation")
st.caption(
    "Hybrid BM25 + cosine search · NumPy-based · No external vector DB"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_api = st.toggle("Use API endpoint", value=True)
    api_url = st.text_input(
        "API URL",
        value=os.getenv("LIGHTRAG_API_URL", "http://localhost:8000"),
    )
    mode = st.selectbox("Retrieval mode", ["hybrid", "semantic", "bm25"])
    top_k = st.slider("Top K results", min_value=1, max_value=20, value=5)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
query = st.text_input("🔍 Ask a question", placeholder="What is hybrid search?")

col1, col2 = st.columns([1, 1])

if col1.button("🚀 Search", use_container_width=True):
    if not query or len(query.strip()) < 3:
        st.warning("Please enter a question (at least 3 characters).")
    elif use_api:
        import requests

        try:
            resp = requests.post(
                f"{api_url}/query",
                json={"query": query, "top_k": top_k, "mode": mode},
                timeout=30,
            )
            if resp.status_code != 200:
                st.error(f"API error: {resp.text}")
            else:
                data = resp.json()
                st.success(data["answer"])
                st.markdown(f"**Mode:** `{data['mode']}`")
                if data.get("sources"):
                    st.markdown("---")
                    st.subheader("📚 Sources")
                    for src in data["sources"]:
                        with st.expander(
                            f"**{src['title']}** — score {src['score']:.4f}"
                        ):
                            st.write(src["snippet"])
                            st.caption(f"id: {src['id']}  |  doc_id: {src['doc_id']}")
        except requests.ConnectionError:
            st.error(
                "Cannot connect to the API. Start it with: "
                "`uvicorn api.main:app --reload`"
            )
        except Exception as exc:
            st.error(f"Request failed: {exc}")
    else:
        with st.spinner("Loading index & searching…"):
            from inference.predictor import LightRAGPredictor

            predictor = LightRAGPredictor.get_instance()
            result = predictor.predict(query, top_k=top_k, mode=mode)

        st.success(result.answer)
        st.markdown(f"**Mode:** `{result.mode}`")
        if result.sources:
            st.markdown("---")
            st.subheader("📚 Sources")
            for s in result.sources:
                with st.expander(f"**{s.title}** — score {s.score:.4f}"):
                    st.write(s.snippet)
                    st.caption(f"id: {s.id}  |  doc_id: {s.doc_id}")

# ---------------------------------------------------------------------------
# Ingest section
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("📥 Ingest New Documents"):
    doc_id = st.text_input("Document ID", value="custom-01")
    doc_title = st.text_input("Title", value="My Custom Document")
    doc_content = st.text_area(
        "Content",
        value="Paste your document text here…",
        height=150,
    )
    if st.button("Ingest"):
        if use_api:
            import requests

            try:
                resp = requests.post(
                    f"{api_url}/ingest",
                    json={
                        "items": [
                            {
                                "id": doc_id,
                                "title": doc_title,
                                "content": doc_content,
                            }
                        ]
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    st.success(f"Ingested {resp.json()['inserted']} document(s)")
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as exc:
                st.error(str(exc))
        else:
            from inference.predictor import LightRAGPredictor

            predictor = LightRAGPredictor.get_instance()
            try:
                n = predictor.ingest(
                    [{"id": doc_id, "title": doc_title, "content": doc_content}]
                )
                st.success(f"Ingested {n} document(s)")
            except RuntimeError as exc:
                st.error(str(exc))
