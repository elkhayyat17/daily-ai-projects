from __future__ import annotations

import os

import streamlit as st

from inference.predictor import RAGPredictor

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")

st.title("💬 RAG Chatbot")
st.caption("Ask questions grounded in your document corpus.")

use_api = st.toggle("Use API endpoint", value=False)
api_url = st.text_input("API URL", value=os.getenv("RAG_API_URL", "http://localhost:8000"))

query = st.text_input("Your question", value="What is RAG?")

if st.button("Ask"):
    if use_api:
        import requests

        response = requests.post(f"{api_url}/chat", json={"query": query})
        if response.status_code != 200:
            st.error(response.text)
        else:
            payload = response.json()
            st.write(payload["answer"])
            st.write("Sources:", payload["sources"])
    else:
        predictor = RAGPredictor.get_instance()
        result = predictor.predict(query)
        st.write(result.answer)
        st.write("Sources:", [s.__dict__ for s in result.sources])
