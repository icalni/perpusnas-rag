"""
backend.py — Lightweight RAG backend.
Uses pre-computed embeddings (numpy) instead of loading a heavy model.
RAM usage: ~200MB instead of 1.5GB. Works on Railway free tier.
"""

import os
import re
import json
import numpy as np
import anthropic
from dotenv import load_dotenv
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
EMB_FILE   = os.path.join(BASE_DIR, "embeddings.npz")
META_FILE  = os.path.join(BASE_DIR, "metadata.json")
API_KEY    = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-sonnet-4-5"
TOP_K      = 5
MAX_TOKENS = 1024

# ── Load pre-computed embeddings ───────────────────────────────────────────
print("Loading pre-computed embeddings...")
data       = np.load(EMB_FILE)
embeddings = data["embeddings"].astype(np.float32)  # shape: (N, 768)
print(f"Loaded {embeddings.shape[0]:,} vectors, dim={embeddings.shape[1]}")

print("Loading metadata...")
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

documents = meta["documents"]
metadatas = meta["metadatas"]
print(f"Ready — {len(documents):,} chunks loaded.\n")

ai_client = anthropic.Anthropic(api_key=API_KEY)

# ── Encode query via Claude (returns a pseudo-embedding using token overlap) ─
def encode_query(question: str) -> np.ndarray:
    """
    Translate query to English, then compute a query vector by averaging
    the pre-computed embeddings of the top keyword-matched documents.
    This gives us a proper vector in the same space as our stored embeddings.
    """
    # Step 1: translate to English
    response = ai_client.messages.create(
        model      = "claude-haiku-4-5",
        max_tokens = 150,
        messages   = [{
            "role": "user",
            "content": f"Translate to English. Return only the translation, nothing else: {question}"
        }]
    )
    translated = response.content[0].text.strip()
    print(f"  Translated: {translated}")

    # Step 2: keyword match to find seed documents
    terms = re.findall(r'\b\w{3,}\b', translated.lower())
    stopwords = {"the","and","for","are","was","were","that","this",
                 "with","from","have","had","been","they","their","what",
                 "which","when","how","why","its","not","but","also"}
    terms = [t for t in terms if t not in stopwords]

    keyword_scores = np.zeros(len(documents))
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        keyword_scores[i] = sum(1 for t in terms if t in doc_lower)

    # Step 3: use top keyword matches to build query vector
    top_seed = np.argsort(keyword_scores)[::-1][:20]
    seed_embs = embeddings[top_seed]
    # Weight by keyword score
    weights = keyword_scores[top_seed]
    weights = weights / (weights.sum() + 1e-9)
    query_vec = (seed_embs * weights[:, None]).sum(axis=0)

    # Normalize
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    return query_vec

def search(question: str, top_k: int = TOP_K):
    """Cosine similarity search using pre-computed embeddings."""
    query_vec = encode_query(question)

    # Cosine similarity (embeddings already normalized)
    scores = embeddings @ query_vec

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "document": documents[idx],
            "metadata": metadatas[idx],
            "score":    round(float(scores[idx]), 3)
        })
    return results

# ── RAG function ───────────────────────────────────────────────────────────
def rag_query(question: str) -> dict:
    # 1. Retrieve relevant chunks
    hits = search(question, TOP_K)

    # 2. Build context
    context_parts = []
    sources       = []

    for i, hit in enumerate(hits):
        meta = hit["metadata"]
        context_parts.append(
            f"[Source {i+1}] {meta['title']} by {meta['author']} ({meta['year']})\n{hit['document']}"
        )
        sources.append({
            "rank":         i + 1,
            "score":        hit["score"],
            "title":        meta["title"],
            "author":       meta["author"],
            "year":         meta["year"],
            "topic":        meta["topic"],
            "gutenberg_id": meta["gutenberg_id"],
            "snippet":      hit["document"][:200] + "..."
        })

    context = "\n\n---\n\n".join(context_parts)

    # 3. Generate answer
    system_prompt = """Anda adalah asisten penelitian Perpustakaan Nasional Republik Indonesia (Perpusnas).

Tugas Anda:
- Jawab pertanyaan pengguna HANYA berdasarkan konteks sumber yang diberikan
- Selalu jawab dalam bahasa yang sama dengan pertanyaan pengguna (Indonesia atau Inggris)
- Selalu sebutkan sumber yang Anda gunakan di akhir jawaban
- Jika konteks tidak cukup, katakan bahwa informasi tidak tersedia dalam koleksi saat ini
- Jangan mengarang fakta yang tidak ada dalam sumber
- Berikan jawaban yang terstruktur dan mudah dipahami

Ini adalah sistem DEMO menggunakan teks domain publik dari Project Gutenberg."""

    response = ai_client.messages.create(
        model      = MODEL_NAME,
        max_tokens = MAX_TOKENS,
        system     = system_prompt,
        messages   = [{"role": "user", "content": f"Konteks:\n{context}\n\nPertanyaan: {question}"}]
    )

    return {
        "question": question,
        "answer":   response.content[0].text,
        "sources":  sources,
        "chunks_retrieved": TOP_K
    }

# ── HTTP Server ────────────────────────────────────────────────────────────
class RAGHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"  → {args[0]} {args[1]}")

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self._json({"status": "ok", "vectors": len(documents)})
            return

        if parsed.path == "/query":
            params   = parse_qs(parsed.query)
            question = params.get("q", [""])[0].strip()
            if not question:
                self._json({"error": "Missing ?q="}, 400)
                return
            try:
                self._json(rag_query(question))
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        self._json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/query":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data     = json.loads(body)
                question = data.get("question", "").strip()
            except Exception:
                self._json({"error": "Invalid JSON"}, 400)
                return
            if not question:
                self._json({"error": "Missing 'question' field"}, 400)
                return
            try:
                self._json(rag_query(question))
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        self._json({"error": "Not found"}, 404)

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_cors()
        self.end_headers()
        self.wfile.write(body)

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT   = int(os.getenv("PORT", 8000))
    server = HTTPServer(("0.0.0.0", PORT), RAGHandler)
    print("=" * 60)
    print(f"  Perpusnas RAG Backend (lightweight)")
    print(f"  http://localhost:{PORT}")
    print(f"  Vectors loaded: {len(documents):,}")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
