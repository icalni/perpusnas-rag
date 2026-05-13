"""
query_test.py — Test the vector DB after pipeline.py has run.
Usage: python query_test.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "./chromadb"

def search(collection, model, query: str, n: int = 3):
    print(f"\nQuery: \"{query}\"")
    print("-" * 60)
    embedding = model.encode([query], normalize_embeddings=True).tolist()
    results   = collection.query(
        query_embeddings=embedding,
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        score = round(1 - dist, 3)
        print(f"\n[{i+1}] Score : {score}")
        print(f"     Source: {meta['title']}")
        print(f"     Author: {meta['author']} ({meta['year']})")
        print(f"     Topic : {meta['topic']}")
        print(f"     Text  : {doc[:200]}...")

def main():
    print("Loading model...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    print("Connecting to ChromaDB...")
    client     = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection("perpusnas_demo")
    print(f"Collection has {collection.count():,} vectors.\n")

    # Test queries — both Bahasa Indonesia and English
    queries = [
        "apa penyebab perang saudara di Amerika?",
        "causes of the American Civil War",
        "seleksi alam dan evolusi spesies",
        "natural selection Darwin evolution",
        "revolusi perancis dan penyebabnya",
        "environmental destruction and deforestation",
        "kerusakan lingkungan dan hutan",
    ]

    for q in queries:
        search(collection, model, q)

if __name__ == "__main__":
    main()
