# query.py
import sys
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

load_dotenv()
client = OpenAI()

DB_DIR = "db"
COLLECTION_NAME = "hoa_docs"

def get_query_embedding(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return resp.data[0].embedding

def build_prompt(question: str, contexts: list[str]) -> str:
    joined_context = "\n\n---\n\n".join(contexts)
    return f"""You are a helpful assistant answering questions about my HOA documents (CC&Rs, bylaws, contracts, etc).

Use ONLY the context provided below to answer. If the answer is not clearly in the context, say you don't know and suggest where in the docs I might look.

Context:
{joined_context}

Question: {question}

Answer:"""

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask a question about your HOA docs: ")

    # 1. Load Chroma
    client_chroma = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client_chroma.get_collection(COLLECTION_NAME)

    # 2. Get query embedding
    q_emb = get_query_embedding(question)

    # 3. Similarity search
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=5,
    )

    contexts = results["documents"][0]
    prompt = build_prompt(question, contexts)

    # 4. Ask GPT
    resp = client.chat.completions.create(
        model="gpt-5-mini",  # or "gpt-5.1"
        messages=[
            {"role": "system", "content": "You are a precise HOA document assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    answer = resp.choices[0].message.content
    print("\n=== Answer ===\n")
    print(answer)
    print("\n==============\n")

if __name__ == "__main__":
    main()

