# ingest.py
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from openai import OpenAI

load_dotenv()
client = OpenAI()

DATA_DIR = "data"
DB_DIR = "db"
COLLECTION_NAME = "hoa_docs"
LARGE_FILE_THRESHOLD_BYTES = 5 * 1024 * 1024

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embeddings(texts, show_progress: bool = False, batch_size: int = 16, desc: str = "Embedding chunks"):
    if not texts:
        return []

    if not show_progress:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [e.embedding for e in resp.data]

    embeddings = []
    with tqdm(total=len(texts), desc=desc) as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            embeddings.extend([e.embedding for e in resp.data])
            pbar.update(len(batch))
    return embeddings


def index_chunks(collection, ids, embeddings, documents, metadatas, show_progress: bool = False, batch_size: int = 50, desc: str = "Indexing chunks"):
    if not ids:
        return

    if not show_progress:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return

    with tqdm(total=len(ids), desc=desc) as pbar:
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            pbar.update(end - i)

def main():
    # 1. Init Chroma
    client_chroma = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        collection = client_chroma.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client_chroma.create_collection(COLLECTION_NAME)

    # 2. Read PDFs and build chunks
    for filename in os.listdir(DATA_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(DATA_DIR, filename)
        is_large_file = os.path.getsize(path) > LARGE_FILE_THRESHOLD_BYTES
        print(f"Reading {path}...")
        full_text = read_pdf_text(path)
        chunks = chunk_text(full_text)
        if not chunks:
            print(f"No text found in {filename}. Skipping.")
            continue

        ids = [f"{filename}-chunk-{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        show_progress = is_large_file
        progress_suffix = " (large file)" if is_large_file else ""
        print(f"Embedding {len(chunks)} chunks from {filename}{progress_suffix}...")
        embeddings = get_embeddings(
            chunks,
            show_progress=show_progress,
            desc=f"Embedding {filename}",
        )

        print(f"Indexing {len(chunks)} chunks from {filename}{progress_suffix}...")
        index_chunks(
            collection,
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            show_progress=show_progress,
            desc=f"Indexing {filename}",
        )

    print(f"Finished indexing collection '{COLLECTION_NAME}'")

if __name__ == "__main__":
    main()
