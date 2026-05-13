"""
precompute.py — Run this ONCE on your Mac.
Saves all embeddings to embeddings.npz so the server never needs the model.
"""

import os
import json
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_DIR     = os.path.join(BASE_DIR, "chromadb")
OUTPUT     = os.path.join(BASE_DIR, "embeddings.npz")
META_FILE  = os.path.join(BASE_DIR, "metadata.json")

print("Loading embedding model...")
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

print("Loading ChromaDB...")
client     = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("perpusnas_demo")
total      = collection.count()
print(f"Total vectors: {total:,}")

# Pull everything out of ChromaDB
print("Extracting all vectors and metadata...")
BATCH  = 500
all_embeddings = []
all_documents  = []
all_metadatas  = []
all_ids        = []

for offset in range(0, total, BATCH):
    result = collection.get(
        limit   = BATCH,
        offset  = offset,
        include = ["embeddings", "documents", "metadatas"]
    )
    all_embeddings.extend(result["embeddings"])
    all_documents.extend(result["documents"])
    all_metadatas.extend(result["metadatas"])
    all_ids.extend(result["ids"])
    print(f"  Extracted {min(offset+BATCH, total):,}/{total:,}")

# Save embeddings as numpy (compact, fast to load)
emb_array = np.array(all_embeddings, dtype=np.float32)
np.savez_compressed(OUTPUT, embeddings=emb_array)
print(f"Saved embeddings to {OUTPUT} ({emb_array.shape})")

# Save metadata + documents as JSON
metadata = {
    "ids":       all_ids,
    "documents": all_documents,
    "metadatas": all_metadatas,
}
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False)
print(f"Saved metadata to {META_FILE}")

# Quick sanity check
print("\nSanity check...")
test_query = "causes of the American Civil War"
q_emb = model.encode([test_query], normalize_embeddings=True)[0]
scores = emb_array @ q_emb
top3   = np.argsort(scores)[::-1][:3]
for i, idx in enumerate(top3):
    print(f"  [{i+1}] {scores[idx]:.3f} — {all_metadatas[idx]['title']} ({all_metadatas[idx]['year']})")

print("\nDone! You can now deploy without the embedding model.")
print(f"Files to commit: embeddings.npz ({emb_array.nbytes/1e6:.1f}MB) + metadata.json")
