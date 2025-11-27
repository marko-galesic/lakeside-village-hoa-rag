# app.py
import os

import streamlit as st

from rag_utils import ensure_dirs, ingest_files_in_data, answer_question_with_rag

ensure_dirs()

st.set_page_config(page_title="HOA RAG Demo")
# Upload size limit configured via .streamlit/config.toml

st.title("HOA Document RAG")

st.markdown(
    "Upload your HOA PDFs or JSON files, ingest them into the vector database, and ask a question."
)

uploaded_files = st.file_uploader(
    "Upload PDFs or JSON files",
    type=["pdf", "json"],
    accept_multiple_files=True,
)
st.caption("Supports uploads up to 1 GB per file.")

if uploaded_files:
    for f in uploaded_files:
        path = os.path.join("data", f.name)
        with open(path, "wb") as out:
            out.write(f.read())
    st.success(f"Saved {len(uploaded_files)} file(s) to data/.")

if st.button("Ingest files into vector DB"):
    with st.spinner("Reading, chunking, embedding, and indexing files..."):
        n_chunks = ingest_files_in_data()
    if n_chunks > 0:
        st.success(f"Ingested {n_chunks} chunks into Chroma.")
    else:
        st.warning("No supported files found in data/ to ingest.")

st.markdown("---")

st.subheader("Ask a question about your HOA documents")

question = st.text_input(
    "Question",
    placeholder="e.g., What are the rules about pets or noise?",
)

n_results = st.slider("Number of chunks to retrieve", 1, 10, 5)

if st.button("Run RAG Query"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Running RAG pipeline..."):
            result = answer_question_with_rag(question, n_results=n_results)

        st.markdown("### Answer")
        st.write(result["answer"])
