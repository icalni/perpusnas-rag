# Perpusnas RAG — Data Pipeline

Mengunduh, membersihkan, memotong, dan mengindeks 15 buku domain publik
dari Project Gutenberg ke dalam vector database (ChromaDB) untuk sistem RAG Perpusnas.

---

## Topik Koleksi Demo

| Topik           | Jumlah Buku | Estimasi Chunks |
|-----------------|-------------|-----------------|
| Civil War       | 4           | ~11.600         |
| History of Europe | 4         | ~14.300         |
| Nature          | 4           | ~9.800          |
| Environment     | 3           | ~4.200          |
| **Total**       | **15**      | **~39.900**     |

---

## Cara Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK data (sekali saja)

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 3. Jalankan pipeline

```bash
python pipeline.py
```

Pipeline akan:
1. Download model embedding multilingual (pertama kali ~400MB)
2. Download 15 buku dari Project Gutenberg (~50MB total)
3. Bersihkan header/footer Gutenberg dari setiap buku
4. Potong teks menjadi chunk 400 kata dengan overlap 50 kata
5. Embed semua chunk menggunakan model multilingual
6. Simpan ke ChromaDB (folder `chromadb/`)
7. Jalankan sanity test — query bahasa Inggris terhadap koleksi

**Estimasi waktu:** 15–30 menit (tergantung koneksi & CPU)

---

## Hasil Output

```
perpusnas_pipeline/
├── pipeline.py          ← script utama
├── requirements.txt
├── pipeline_stats.json  ← statistik setelah pipeline selesai
├── raw_books/           ← teks mentah dari Gutenberg (.txt)
├── chunks/              ← (opsional) chunk per buku
└── chromadb/            ← vector database (gunakan ini di backend)
```

---

## Menggunakan Vector DB di Backend

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Load
client     = chromadb.PersistentClient(path="./chromadb")
collection = client.get_collection("perpusnas_demo")
model      = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Query (bisa Bahasa Indonesia atau Inggris)
query     = "apa penyebab perang saudara di Amerika?"
embedding = model.encode([query], normalize_embeddings=True).tolist()

results = collection.query(
    query_embeddings=embedding,
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    score = round(1 - dist, 3)
    print(f"[{score}] {meta['title']} ({meta['year']}) — {meta['topic']}")
    print(f"       {doc[:150]}...")
    print()
```

---

## Model Embedding

**paraphrase-multilingual-mpnet-base-v2**
- Mendukung 50+ bahasa termasuk Bahasa Indonesia
- Dimensi vektor: 768
- Cross-lingual: query Bahasa Indonesia → dokumen Inggris ✓
- Lisensi: Apache 2.0 (bebas digunakan)

---

## Catatan Penting

- Semua buku adalah domain publik dari Project Gutenberg
- Pipeline ini hanya untuk **versi demo** — pada produksi gunakan koleksi resmi Perpusnas
- Tambahkan disclaimer di UI bahwa jawaban bersifat informatif, bukan otoritatif
