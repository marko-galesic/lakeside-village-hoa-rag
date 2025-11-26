# rag_utils.py
import os
import json
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from openai import OpenAI

from sklearn.decomposition import PCA

load_dotenv()
client = OpenAI()

DATA_DIR = "data"
DB_DIR = "db"
COLLECTION_NAME = "hoa_docs"


# ---------- BASIC UTILITIES ----------

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)


def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _flatten_json(value: Any, prefix: str = "") -> List[str]:
    """Recursively flatten a JSON object into human-readable lines."""
    lines: List[str] = []

    if isinstance(value, dict):
        for key, val in value.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json(val, new_prefix))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            lines.extend(_flatten_json(item, new_prefix))
    else:
        key = prefix or "value"
        lines.append(f"{key}: {value}")

    return lines


def read_json_text(path: str) -> str:
    """Load JSON and flatten into text so it can be chunked and embedded."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flattened = _flatten_json(data)
    return "\n".join(flattened)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------- EMBEDDINGS ----------

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [e.embedding for e in resp.data]


def embed_single(text: str) -> List[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return resp.data[0].embedding


# ---------- CHROMA SETUP ----------

def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = chroma_client.create_collection(COLLECTION_NAME)
    return collection


# ---------- INGESTION ----------

def ingest_files_in_data() -> int:
    """
    Read all supported files in data/, chunk, embed, and store in Chroma.
    Returns number of chunks indexed.
    """
    ensure_dirs()
    collection = get_chroma_collection()

    all_ids: List[str] = []
    all_texts: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    for filename in os.listdir(DATA_DIR):
        lower_name = filename.lower()
        path = os.path.join(DATA_DIR, filename)

        if lower_name.endswith(".pdf"):
            full_text = read_pdf_text(path)
            source_type = "pdf"
        elif lower_name.endswith(".json"):
            full_text = read_json_text(path)
            source_type = "json"
        else:
            continue

        if not full_text.strip():
            continue

        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            doc_id = f"{filename}-chunk-{i}"
            all_ids.append(doc_id)
            all_texts.append(chunk)
            all_meta.append({
                "source": filename,
                "chunk_index": i,
                "source_type": source_type,
            })

    if not all_texts:
        return 0

    embeddings = embed_texts(all_texts)

    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_meta,
    )

    return len(all_texts)


# Backward compatibility alias
def ingest_pdfs_in_data() -> int:
    return ingest_files_in_data()


# ---------- QUERY + RETRIEVAL ----------

def retrieve_context(question: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Embed question, query Chroma, return retrieval results.
    """
    ensure_dirs()
    collection = get_chroma_collection()

    q_emb = embed_single(question)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "query_embedding": q_emb,
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }


def build_rag_prompt(question: str, contexts: List[str]) -> str:
    joined_context = "\n\n---\n\n".join(contexts)
    return f"""You are a helpful assistant answering questions about HOA documents (CC&Rs, bylaws, contracts, etc).

Use ONLY the context provided below to answer. If the answer is not clearly in the context, say you don't know and suggest where in the docs I might look.

Context:
{joined_context}

Question: {question}

Answer:"""


def answer_question_with_rag(question: str, n_results: int = 5) -> Dict[str, Any]:
    retrieval = retrieve_context(question, n_results=n_results)
    contexts = retrieval["documents"]
    prompt = build_rag_prompt(question, contexts)

    resp = client.chat.completions.create(
        model="gpt-5-mini",  # or "gpt-5.1"
        messages=[
            {"role": "system", "content": "You are a precise HOA document assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = resp.choices[0].message.content

    retrieval["answer"] = answer
    return retrieval


# ---------- VISUALIZATION HELPERS ----------
def sample_collection_for_viz(max_samples: int = 50) -> Tuple[List[str], List[Dict], List[List[float]]]:
    """
    Load all docs from Chroma, sample up to max_samples.
    If embeddings are not available or mismatched, recompute them for the sample.
    """
    ensure_dirs()
    collection = get_chroma_collection()
    all_docs = collection.get(include=["documents", "metadatas", "embeddings"])

    docs = all_docs.get("documents") or []
    metas = all_docs.get("metadatas") or []
    embs = all_docs.get("embeddings")

    # Normalize embeddings: None → [], NumPy array → list-of-lists
    if embs is None:
        embs = []
    else:
        try:
            import numpy as np
            if isinstance(embs, np.ndarray):
                embs = embs.tolist()
        except ImportError:
            # If numpy isn't around or not used, just assume it's already a list-like
            pass

    if not docs:
        return [], [], []

    total_docs = len(docs)

    # Down-sample docs (and embeddings if present) to max_samples
    indices = list(range(total_docs))
    if total_docs > max_samples:
        indices = indices[:max_samples]

    docs = [docs[i] for i in indices]
    metas = [metas[i] for i in indices]

    # If we have embeddings for all docs, downsample them too; otherwise, drop and recompute
    if embs and len(embs) == total_docs:
        embs = [embs[i] for i in indices]
    else:
        embs = []

    # If embeddings missing or length mismatch, recompute for this small sample
    if not embs or len(embs) != len(docs):
        embs = embed_texts(docs)

    return docs, metas, embs




def project_embeddings_2d(doc_embs: List[List[float]], query_embs: List[List[float]]):
    """
    PCA 2D projection of doc_embs + query_embs.
    Returns:
      doc_2d, query_2d (each list of [x,y])
    """
    if doc_embs is None or len(doc_embs) == 0:
        return [], []

    all_embs = doc_embs + query_embs
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_embs)

    doc_2d = all_2d[:len(doc_embs)]
    query_2d = all_2d[len(doc_embs):]
    return doc_2d, query_2d

