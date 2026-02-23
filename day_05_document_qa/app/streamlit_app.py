"""
Day 05 — Document Q&A: Streamlit Demo UI
Interactive document upload and question-answering interface.
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="📄 Document Q&A",
    page_icon="📄",
    layout="wide",
)


def check_health() -> dict | None:
    """Check if the API is running."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


def main():
    st.title("📄 Document Q&A with Vector Search")
    st.markdown(
        "Upload documents, ask questions, and get answers powered by "
        "**FAISS** vector search and **extractive QA**."
    )
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Health check
        health = check_health()
        if health:
            status_color = "🟢" if health.get("status") == "healthy" else "🟡"
            st.success(f"{status_color} API Connected")
            st.metric("Indexed Vectors", health.get("num_vectors", 0))
            st.caption(f"QA Model: {'✅' if health.get('qa_model_loaded') else '❌'}")
        else:
            st.error("🔴 API not available. Start with: `uvicorn api.main:app`")
            st.stop()

        st.markdown("---")
        top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)

        st.markdown("---")
        st.header("📂 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload files (TXT, PDF, MD, DOCX)",
            type=["txt", "pdf", "md", "docx", "csv", "json"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("📤 Upload & Index", type="primary"):
            with st.spinner("Processing files..."):
                for f in uploaded_files:
                    files = {"file": (f.name, f.getvalue())}
                    resp = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(
                            f"✅ {f.name}: {result.get('chunks_created', 0)} chunks indexed"
                        )
                    else:
                        st.error(f"❌ {f.name}: {resp.text}")

        st.markdown("---")
        st.header("📝 Ingest Text")
        with st.form("ingest_form"):
            doc_title = st.text_input("Document Title")
            doc_content = st.text_area("Document Content", height=150)
            submit_ingest = st.form_submit_button("📥 Ingest")

        if submit_ingest and doc_title and doc_content:
            payload = {
                "documents": [
                    {"title": doc_title, "content": doc_content, "source": "streamlit"}
                ]
            }
            with st.spinner("Ingesting..."):
                resp = requests.post(f"{API_URL}/ingest", json=payload, timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(
                        f"✅ Ingested: {result.get('chunks_created', 0)} chunks"
                    )
                else:
                    st.error(f"❌ Error: {resp.text}")

        st.markdown("---")
        if st.button("🗑️ Clear Index"):
            resp = requests.delete(f"{API_URL}/index", timeout=10)
            if resp.status_code == 200:
                st.success("Index cleared!")
            else:
                st.error("Failed to clear index.")

    # Main area — Q&A
    st.header("❓ Ask a Question")

    # Sample questions
    sample_questions = [
        "What is Python?",
        "What are the types of machine learning?",
        "How does FAISS work?",
        "What is RAG?",
        "What is the transformer architecture?",
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Type your question:",
            placeholder="e.g., What is machine learning?",
        )
    with col2:
        st.markdown("**Quick questions:**")
        for q in sample_questions[:3]:
            if st.button(q, key=f"sample_{q}"):
                question = q

    if question:
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "top_k": top_k},
                    timeout=60,
                )

                if resp.status_code == 200:
                    result = resp.json()

                    # Answer
                    st.markdown("### 💡 Answer")
                    st.info(result["answer"])

                    # Metadata
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with col_b:
                        st.metric("Mode", result["mode"].replace("_", " ").title())
                    with col_c:
                        st.metric("Time", f"{result['elapsed_ms']:.0f}ms")

                    # Sources
                    if result["sources"]:
                        st.markdown("### 📚 Sources")
                        for src in result["sources"]:
                            st.markdown(f"- 📄 {src}")

                    # Retrieved chunks
                    if result["retrieved_chunks"]:
                        st.markdown("### 🔍 Retrieved Context")
                        for i, chunk in enumerate(result["retrieved_chunks"]):
                            with st.expander(
                                f"Chunk {i+1} — {chunk['title']} (score: {chunk['score']:.3f})"
                            ):
                                st.markdown(chunk["text"])
                else:
                    st.error(f"Error: {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the server running?")
            except Exception as e:
                st.error(f"Error: {e}")

    # Footer
    st.markdown("---")
    st.caption(
        "Day 05 — Document Q&A with Vector Database | "
        "FAISS + Sentence-Transformers + Extractive QA"
    )


if __name__ == "__main__":
    main()
