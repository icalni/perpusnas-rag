"""
Perpusnas RAG Data Pipeline
Downloads, cleans, chunks, embeds, and stores books into ChromaDB.
Model: paraphrase-multilingual-mpnet-base-v2 (supports Bahasa Indonesia)
"""

import os
import re
import time
import json
import requests
import nltk
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Setup ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(BASE_DIR, "raw_books")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
DB_DIR     = os.path.join(BASE_DIR, "chromadb")

for d in [RAW_DIR, CHUNKS_DIR, DB_DIR]:
    os.makedirs(d, exist_ok=True)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Book catalog ───────────────────────────────────────────────────────────
BOOKS = [
    # Civil War
    {"id": 4367,  "title": "Personal Memoirs of U.S. Grant",                   "author": "Ulysses S. Grant",       "year": 1885, "topic": "Civil War"},
    {"id": 73,    "title": "The Red Badge of Courage",                          "author": "Stephen Crane",           "year": 1895, "topic": "Civil War"},
    {"id": 23,    "title": "Narrative of the Life of Frederick Douglass",       "author": "Frederick Douglass",      "year": 1845, "topic": "Civil War"},
    {"id": 58820, "title": "The Civil War",                                     "author": "James I. Robertson",      "year": 1961, "topic": "Civil War"},
    # History of Europe
    {"id": 731,   "title": "History of the Decline and Fall of the Roman Empire Vol.1", "author": "Edward Gibbon", "year": 1776, "topic": "History of Europe"},
    {"id": 1301,  "title": "The French Revolution",                             "author": "Thomas Carlyle",          "year": 1837, "topic": "History of Europe"},
    {"id": 3465,  "title": "A History of France",                               "author": "Charlotte M. Yonge",      "year": 1879, "topic": "History of Europe"},
    {"id": 2042,  "title": "The History of England Vol.1",                      "author": "David Hume",              "year": 1754, "topic": "History of Europe"},
    # Nature
    {"id": 1228,  "title": "On the Origin of Species",                          "author": "Charles Darwin",          "year": 1859, "topic": "Nature"},
    {"id": 205,   "title": "Walden",                                            "author": "Henry David Thoreau",     "year": 1854, "topic": "Nature"},
    {"id": 944,   "title": "The Voyage of the Beagle",                          "author": "Charles Darwin",          "year": 1839, "topic": "Nature"},
    {"id": 9440,  "title": "A Naturalist's Voyage Round the World",             "author": "Charles Darwin",          "year": 1860, "topic": "Nature"},
    # Environment
    {"id": 37957, "title": "Man and Nature",                                    "author": "George Perkins Marsh",    "year": 1864, "topic": "Environment"},
    {"id": 13249, "title": "Our Vanishing Wild Life",                           "author": "William T. Hornaday",     "year": 1913, "topic": "Environment"},
    {"id": 8282,  "title": "Mutual Aid: A Factor of Evolution",                 "author": "Peter Kropotkin",         "year": 1902, "topic": "Environment"},
]

# ── Step 1: Download ───────────────────────────────────────────────────────
def download_book(book: dict) -> str | None:
    path = os.path.join(RAW_DIR, f"{book['id']}.txt")
    if os.path.exists(path):
        return path

    urls = [
        f"https://www.gutenberg.org/files/{book['id']}/{book['id']}-0.txt",
        f"https://www.gutenberg.org/files/{book['id']}/{book['id']}.txt",
        f"https://www.gutenberg.org/cache/epub/{book['id']}/pg{book['id']}.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 5000:
                with open(path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(r.text)
                return path
        except Exception:
            continue
    return None

# ── Step 2: Clean ──────────────────────────────────────────────────────────
def clean_text(raw: str) -> str:
    # Strip Gutenberg header
    start_markers = [
        r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG",
        r"\*\*\*START OF THE PROJECT GUTENBERG",
        r"END OF THE PROJECT GUTENBERG EBOOK",
    ]
    for marker in start_markers:
        match = re.search(marker, raw, re.IGNORECASE)
        if match:
            raw = raw[match.end():]
            break

    # Strip Gutenberg footer
    end_markers = [
        r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG",
        r"End of (the )?Project Gutenberg",
    ]
    for marker in end_markers:
        match = re.search(marker, raw, re.IGNORECASE)
        if match:
            raw = raw[:match.start()]
            break

    # Normalize whitespace
    raw = re.sub(r"\r\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.strip()

# ── Step 3: Chunk ──────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        words = sentence.split()
        wcount = len(words)

        if current_len + wcount > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep overlap
            overlap_words = []
            overlap_count = 0
            for s in reversed(current):
                sw = s.split()
                if overlap_count + len(sw) <= overlap:
                    overlap_words.insert(0, s)
                    overlap_count += len(sw)
                else:
                    break
            current = overlap_words
            current_len = overlap_count

        current.append(sentence)
        current_len += wcount

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c.strip()) > 100]

# ── Step 4: Embed + Store ──────────────────────────────────────────────────
def build_vector_db(all_chunks: list[dict], model: SentenceTransformer):
    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection("perpusnas_demo")
    except Exception:
        pass

    collection = client.create_collection(
        name="perpusnas_demo",
        metadata={"hnsw:space": "cosine"}
    )

    texts    = [c["text"]   for c in all_chunks]
    ids      = [c["id"]     for c in all_chunks]
    metas    = [c["meta"]   for c in all_chunks]

    # Embed in batches
    BATCH = 64
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH), desc="  Embedding"):
        batch = texts[i:i+BATCH]
        embs  = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(embs.tolist())

    # Store in batches
    STORE_BATCH = 500
    for i in range(0, len(texts), STORE_BATCH):
        collection.add(
            ids        = ids[i:i+STORE_BATCH],
            embeddings = all_embeddings[i:i+STORE_BATCH],
            documents  = texts[i:i+STORE_BATCH],
            metadatas  = metas[i:i+STORE_BATCH],
        )

    return collection

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Perpusnas RAG Data Pipeline")
    print("=" * 60)

    # Load multilingual embedding model
    # Supports Bahasa Indonesia + English cross-lingual retrieval
    print("\n[1/4] Loading multilingual embedding model...")
    print("      Using: paraphrase-multilingual-mpnet-base-v2")
    print("      (Supports 50+ languages including Bahasa Indonesia)")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    print("      Model loaded.")

    # Download
    print(f"\n[2/4] Downloading {len(BOOKS)} books from Project Gutenberg...")
    downloaded = []
    for book in tqdm(BOOKS, desc="  Downloading"):
        path = download_book(book)
        if path:
            downloaded.append((book, path))
        else:
            print(f"      SKIP: {book['title']} (download failed)")
        time.sleep(0.5)  # be polite to Gutenberg servers

    print(f"      Downloaded: {len(downloaded)}/{len(BOOKS)} books")

    # Clean + Chunk
    print(f"\n[3/4] Cleaning and chunking {len(downloaded)} books...")
    all_chunks = []
    stats = []

    for book, path in tqdm(downloaded, desc="  Chunking"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        clean  = clean_text(raw)
        chunks = chunk_text(clean, chunk_size=400, overlap=50)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{book['id']}_chunk_{i:04d}"
            all_chunks.append({
                "id":   chunk_id,
                "text": chunk,
                "meta": {
                    "gutenberg_id": str(book["id"]),
                    "title":        book["title"],
                    "author":       book["author"],
                    "year":         str(book["year"]),
                    "topic":        book["topic"],
                    "chunk_index":  str(i),
                }
            })

        stats.append({
            "title":       book["title"],
            "topic":       book["topic"],
            "char_count":  len(clean),
            "chunk_count": len(chunks),
        })
        print(f"      {book['title'][:45]:<45} → {len(chunks):>4} chunks")

    total_chunks = len(all_chunks)
    print(f"\n      Total chunks: {total_chunks:,}")

    # Save chunk stats
    with open(os.path.join(BASE_DIR, "pipeline_stats.json"), "w") as f:
        json.dump({"books": stats, "total_chunks": total_chunks}, f, indent=2)

    # Embed + Store
    print(f"\n[4/4] Embedding and storing {total_chunks:,} chunks into ChromaDB...")
    collection = build_vector_db(all_chunks, model)
    print(f"      Stored {collection.count():,} vectors in ChromaDB.")

    # Quick sanity test
    print("\n── Sanity Test ──────────────────────────────────────────")
    test_query = "causes of the American Civil War"
    test_emb   = model.encode([test_query], normalize_embeddings=True).tolist()
    results    = collection.query(query_embeddings=test_emb, n_results=3)

    print(f"  Query: \"{test_query}\"")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        score = 1 - results["distances"][0][i]
        print(f"\n  [{i+1}] Score: {score:.3f}")
        print(f"       Source: {meta['title']} — {meta['author']} ({meta['year']})")
        print(f"       Topic:  {meta['topic']}")
        print(f"       Text:   {doc[:120]}...")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  DB saved to: {DB_DIR}")
    print(f"  Total vectors: {collection.count():,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
