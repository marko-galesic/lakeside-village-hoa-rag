# visualize_embeddings.py
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

DB_DIR = "db"
COLLECTION_NAME = "hoa_docs"

def get_embedding(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return resp.data[0].embedding

def main():
    # 1. Load some document chunks from Chroma
    client_chroma = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client_chroma.get_collection(COLLECTION_NAME)

    docs = collection.get()
    texts = docs["documents"]
    metadatas = docs["metadatas"]

    # Sample to avoid plotting 1000 points
    indices = list(range(len(texts)))
    random.shuffle(indices)
    indices = indices[:50]

    sampled_texts = [texts[i] for i in indices]
    sampled_labels = [
        f"{metadatas[i].get('source')}#{metadatas[i].get('chunk_index')}"
        for i in indices
    ]

    # 2. Embed sampled chunks
    print("Embedding sampled chunks...")
    doc_embeddings = []
    for t in sampled_texts:
        doc_embeddings.append(get_embedding(t))

    # 3. Add a few example questions as "query points"
    queries = [
        "What are the pet rules?",
        "When are HOA dues due?",
        "What is the noise policy at night?",
    ]
    query_embeddings = [get_embedding(q) for q in queries]

    # 4. Project to 2D using PCA
    all_embeddings = doc_embeddings + query_embeddings
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_embeddings)

    doc_2d = all_2d[:len(doc_embeddings)]
    query_2d = all_2d[len(doc_embeddings):]

    # 5. Plot
    plt.figure(figsize=(8, 6))
    # documents
    xs = [p[0] for p in doc_2d]
    ys = [p[1] for p in doc_2d]
    plt.scatter(xs, ys, alpha=0.6, label="Chunks")

    # queries
    qxs = [p[0] for p in query_2d]
    qys = [p[1] for p in query_2d]
    plt.scatter(qxs, qys, marker="x", s=80, label="Queries")

    for (qx, qy, qtext) in zip(qxs, qys, queries):
        plt.text(qx, qy, qtext, fontsize=8)

    plt.title("2D projection of HOA chunks + example queries")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

