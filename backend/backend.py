"""
Perpusnas RAG Backend
Run: python3 backend.py
API will be available at http://localhost:8000
"""

import os
import json
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import anthropic
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ── Config ─────────────────────────────────────────────────────────────────
load_dotenv()

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_DIR      = os.path.join(BASE_DIR, "chromadb")
API_KEY     = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME  = "claude-sonnet-4-5"
TOP_K       = 5
MAX_TOKENS  = 1024

# ── Load model & DB ────────────────────────────────────────────────────────
print("Loading embedding model...")
embedder   = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

print("Connecting to ChromaDB...")
client     = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("perpusnas_demo")
print(f"Ready — {collection.count():,} vectors loaded.\n")

ai_client  = anthropic.Anthropic(api_key=API_KEY)

# ── Core RAG function ──────────────────────────────────────────────────────
def rag_query(question: str) -> dict:

    # 1. Embed the question
    embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    # 2. Retrieve top-K relevant chunks
    results = collection.query(
        query_embeddings=embedding,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 3. Build context string
    context_parts = []
    sources       = []

    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances)):
        score = round(1 - dist, 3)
        context_parts.append(
            f"[Source {i+1}] {meta['title']} by {meta['author']} ({meta['year']})\n{chunk}"
        )
        sources.append({
            "rank":         i + 1,
            "score":        score,
            "title":        meta["title"],
            "author":       meta["author"],
            "year":         meta["year"],
            "topic":        meta["topic"],
            "gutenberg_id": meta["gutenberg_id"],
            "snippet":      chunk[:200] + "..."
        })

    context = "\n\n---\n\n".join(context_parts)

    # 4. Build system prompt
    system_prompt = """Anda adalah asisten penelitian Perpustakaan Nasional Republik Indonesia (Perpusnas).

Tugas Anda:
- Jawab pertanyaan pengguna HANYA berdasarkan konteks sumber yang diberikan
- Selalu jawab dalam bahasa yang sama dengan pertanyaan pengguna (Indonesia atau Inggris)
- Selalu sebutkan sumber yang Anda gunakan di akhir jawaban
- Jika konteks tidak cukup untuk menjawab, katakan dengan jelas bahwa informasi tidak tersedia dalam koleksi saat ini
- Jangan mengarang fakta yang tidak ada dalam sumber
- Berikan jawaban yang terstruktur dan mudah dipahami

Format jawaban:
1. Jawaban utama (2-4 paragraf)
2. Sumber: sebutkan judul dan pengarang yang digunakan

Ingat: Ini adalah sistem DEMO menggunakan teks domain publik dari Project Gutenberg."""

    user_message = f"""Konteks dari koleksi Perpusnas:

{context}

Pertanyaan: {question}"""

    # 5. Generate answer
    response = ai_client.messages.create(
        model      = MODEL_NAME,
        max_tokens = MAX_TOKENS,
        system     = system_prompt,
        messages   = [{"role": "user", "content": user_message}]
    )

    answer = response.content[0].text

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources,
        "model":    MODEL_NAME,
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

        # Health check
        if parsed.path == "/health":
            self._json({"status": "ok", "vectors": collection.count()})
            return

        # Query via GET: /query?q=your+question
        if parsed.path == "/query":
            params = parse_qs(parsed.query)
            question = params.get("q", [""])[0].strip()

            if not question:
                self._json({"error": "Missing query parameter ?q="}, 400)
                return

            print(f"\nQuery: {question}")
            try:
                result = rag_query(question)
                self._json(result)
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        self._json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/query":
            length   = int(self.headers.get("Content-Length", 0))
            body     = self.rfile.read(length)
            try:
                data     = json.loads(body)
                question = data.get("question", "").strip()
            except Exception:
                self._json({"error": "Invalid JSON"}, 400)
                return

            if not question:
                self._json({"error": "Missing 'question' field"}, 400)
                return

            print(f"\nQuery: {question}")
            try:
                result = rag_query(question)
                self._json(result)
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
    PORT   = 8000
    server = HTTPServer(("0.0.0.0", PORT), RAGHandler)
    print("=" * 60)
    print(f"  Perpusnas RAG Backend running")
    print(f"  http://localhost:{PORT}")
    print(f"")
    print(f"  Endpoints:")
    print(f"    GET  /health")
    print(f"    GET  /query?q=your+question")
    print(f"    POST /query  {{\"question\": \"your question\"}}")
    print(f"")
    print(f"  Try it:")
    print(f"    curl \"http://localhost:{PORT}/query?q=apa+penyebab+perang+saudara\"")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
