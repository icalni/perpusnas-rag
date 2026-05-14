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

# ── Translate query to English using Claude ────────────────────────────────
def translate_query(question: str) -> str:
    """Translate Indonesian query to English for better keyword matching."""
    response = ai_client.messages.create(
        model      = "claude-haiku-4-5",
        max_tokens = 150,
        messages   = [{
            "role": "user",
            "content": f"Translate this to English. Return only the translation, nothing else: {question}"
        }]
    )
    return response.content[0].text.strip()

def search(question: str, top_k: int = TOP_K):
    """
    Search using TF-IDF style scoring with translated query.
    Scores documents by weighted keyword overlap with boosting for
    exact phrase matches and topic-relevant terms.
    """
    # Translate to English if needed
    translated = translate_query(question)
    print(f"  Translated: {translated}")

    # Build term list from both original and translated query
    combined = f"{question} {translated}".lower()
    # Remove common stop words
    stopwords = {"the","a","an","of","in","is","are","was","were","to","for",
                 "and","or","but","with","from","by","at","on","as","it","its",
                 "this","that","what","how","why","when","which","who","does",
                 "did","do","be","been","have","has","had","not","no","apa",
                 "yang","di","dan","ke","dari","ini","itu","dengan","untuk",
                 "adalah","pada","dalam","se","akan","atau"}
    terms = [t for t in re.findall(r'\b\w+\b', combined) if t not in stopwords and len(t) > 2]

    if not terms:
        terms = combined.split()

    print(f"  Search terms: {terms}")

    # Score each document
    scores = np.zeros(len(documents))
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        meta      = metadatas[i]
        score     = 0.0

        for term in terms:
            if term in doc_lower:
                # Base score per hit
                count  = doc_lower.count(term)
                score += min(count * 0.15, 0.6)  # cap per term

        # Boost if topic matches query terms
        topic_lower = meta.get("topic", "").lower()
        for term in terms:
            if term in topic_lower:
                score += 0.3

        # Boost title matches
        title_lower = meta.get("title", "").lower()
        for term in terms:
            if term in title_lower:
                score += 0.2

        scores[i] = score

    # Normalize
    max_score = scores.max()
    if max_score > 0:
        scores = scores / max_score

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
