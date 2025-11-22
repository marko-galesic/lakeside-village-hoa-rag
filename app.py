# app.py
import os
import io

import streamlit as st
import matplotlib.pyplot as plt

from rag_utils import (
    ensure_dirs,
    ingest_pdfs_in_data,
    answer_question_with_rag,
    sample_collection_for_viz,
    project_embeddings_2d,
)

ensure_dirs()

st.set_page_config(
    page_title="HOA RAG Demo",
    layout="wide",
)


# ---------- SIDEBAR: SETUP & INGEST ----------

st.sidebar.title("HOA RAG Demo")

st.sidebar.markdown("### 1. Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Drop your HOA PDFs here",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    for f in uploaded_files:
        path = os.path.join("data", f.name)
        # save for ingest
        with open(path, "wb") as out:
            out.write(f.read())
    st.sidebar.success(f"Saved {len(uploaded_files)} file(s) to data/")

if st.sidebar.button("Ingest PDFs into vector DB"):
    with st.spinner("Reading, chunking, embedding, and indexing PDFs..."):
        n_chunks = ingest_pdfs_in_data()
    if n_chunks > 0:
        st.sidebar.success(f"Ingested {n_chunks} chunks into Chroma.")
    else:
        st.sidebar.warning("No PDFs found in data/ to ingest.")


st.sidebar.markdown("---")
st.sidebar.markdown("### 2. About this demo")
st.sidebar.write(
    """
This app is an **educational RAG demo**:

1. Your PDFs are broken into chunks and turned into embeddings (vectors).
2. Your question is also embedded.
3. The app finds the chunks **closest** to your question in vector space.
4. Those chunks are sent to GPT to generate an answer.
5. A 2D plot shows how your query sits among document chunks.
"""
)


# ---------- MAIN LAYOUT ----------

st.title("HOA Document RAG – Visual Explainer")

st.markdown(
    """
This is a local demo that shows **how retrieval-augmented generation (RAG) works**:

1. **Ingest** HOA PDFs → chunk + embed → store in a local vector DB (Chroma).
2. **Query** → embed your question.
3. **Retrieve** top matching chunks by vector similarity.
4. **Generate** an answer from those chunks with GPT.

Use the controls below to explore the pipeline.
"""
)

col_left, col_right = st.columns([3, 2])


# ---------- LEFT: QUESTION + PIPELINE ----------

with col_left:
    st.subheader("Ask a question about your HOA documents")

    question = st.text_input(
        "Question",
        placeholder="e.g., What are the rules about pets or noise?",
    )

    n_results = st.slider("Number of chunks to retrieve", 1, 10, 5)

    if st.button("Run RAG query"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Running RAG pipeline..."):
                result = answer_question_with_rag(question, n_results=n_results)

            st.markdown("### 🔄 Pipeline Overview")
            st.markdown(
                """
1. **Question → Embedding** (high-dimensional vector representing meaning)  
2. **Vector Search** → find nearest chunks in vector DB  
3. **Build Prompt** → question + top chunks as context  
4. **GPT Answer** → model answers using only that context
"""
            )

            st.markdown("### ✅ Answer")
            st.write(result["answer"])

            st.markdown("### 🔍 Retrieval Trace")
            docs = result["documents"]
            metas = result["metadatas"]
            dists = result["distances"]

            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                with st.expander(f"Result #{i+1} – {meta.get('source')} (chunk {meta.get('chunk_index')}) | distance={dist:.4f}"):
                    st.write(doc)

            # Save latest result in session_state for visualization
            st.session_state["last_question"] = question
            st.session_state["last_query_embedding"] = result["query_embedding"]


# ---------- RIGHT: VECTOR VISUALIZATION ----------

with col_right:
    st.subheader("Vector Space Visualization (2D)")

    st.markdown(
        """
This plot shows a **2D projection** of:

- A sample of document chunks (points)
- Your latest query (X marker)
"""
    )

    if "last_query_embedding" not in st.session_state:
        st.info("Run a RAG query on the left to visualize embeddings.")
    else:
        docs, metas, doc_embs = sample_collection_for_viz(max_samples=80)

        if not docs or not doc_embs:
            st.warning("No document embeddings found for visualization. Try re-running ingestion.")
        else:
            query_embs = [st.session_state["last_query_embedding"]]

            doc_2d, query_2d = project_embeddings_2d(doc_embs, query_embs)

            if len(doc_2d) == 0 or len(query_2d) == 0:
                st.warning("Not enough points to build a 2D projection yet.")
            else:
                fig, ax = plt.subplots(figsize=(5, 4))

                xs = [p[0] for p in doc_2d]
                ys = [p[1] for p in doc_2d]
                ax.scatter(xs, ys, alpha=0.6)

                qx = query_2d[0][0]
                qy = query_2d[0][1]
                ax.scatter([qx], [qy], marker="x", s=100)
                ax.text(qx, qy, "  query", fontsize=9, verticalalignment="center")

                ax.set_title("2D projection of chunks + latest query")
                ax.set_xticks([])
                ax.set_yticks([])

                st.pyplot(fig)

                with st.expander("Show sample points info"):
                    st.write("Number of sampled chunks:", len(docs))
                    st.write("Example metadata for first few points:")
                    st.write(metas[:5])


