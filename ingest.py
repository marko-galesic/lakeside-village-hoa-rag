# ingest.py
import os
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from openai import OpenAI

load_dotenv()
client = OpenAI()

DATA_DIR = "data"
DB_DIR = "db"
COLLECTION_NAME = "hoa_docs"

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

def get_embeddings(texts):
    # texts: list[str]
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [e.embedding for e in resp.data]

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
    all_texts = []
    all_ids = []
    all_meta = []

    doc_id_counter = 0
    for filename in os.listdir(DATA_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(DATA_DIR, filename)
        print(f"Reading {path}...")
        full_text = read_pdf_text(path)
        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            doc_id = f"{filename}-chunk-{i}"
            all_ids.append(doc_id)
            all_texts.append(chunk)
            all_meta.append({"source": filename, "chunk_index": i})

        doc_id_counter += 1

    if not all_texts:
        print("No PDFs found in data/. Exiting.")
        return

    print(f"Embedding {len(all_texts)} chunks...")
    embeddings = get_embeddings(all_texts)

    # 3. Add to Chroma
    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_meta,
    )

    print(f"Indexed {len(all_texts)} chunks into collection '{COLLECTION_NAME}'")

if __name__ == "__main__":
    main()
