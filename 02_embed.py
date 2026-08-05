"""
Step 2: Embeddings + vector store

WHY AN EMBEDDING MODEL AT ALL:
An embedding model turns text into a fixed-length vector of numbers such that
texts with similar MEANING end up close together in that vector space (by
cosine similarity / distance), even if they don't share exact words. That's
what lets "how do I validate a query param" retrieve a chunk about
"Query parameter validations" even though the wording differs.

MODEL CHOICE: all-MiniLM-L6-v2 (via sentence-transformers)
- Runs fully local and free (no API key, no per-call cost)
- 384-dimensional vectors, ~80MB model, fast on CPU
- The standard lightweight baseline for small-scale RAG/retrieval work -
  trades a bit of accuracy for speed/size versus bigger models (bge-base,
  e5-large). Good enough at 132 chunks; would revisit for a larger corpus
  or if retrieval quality testing showed it missing relevant chunks.

WHY A VECTOR DATABASE (Chroma) INSTEAD OF JUST A LIST OF VECTORS:
We need "given a query vector, find the N closest chunk vectors" fast.
Chroma stores the vectors, the original text, and metadata together, and
handles the nearest-neighbor search + persistence (saves to disk) for us.
For 132 chunks, a plain in-memory list + numpy would work too - Chroma
is chosen because it's what we'll actually use in the FastAPI app later,
and because it's zero-infra (runs embedded in-process, no server to run).
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks.jsonl")
DB_DIR = "data/chroma"
COLLECTION_NAME = "fastapi_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks():
    chunks = []
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(json.loads(line))
    return chunks


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=DB_DIR)
    # Fresh run each time: drop and recreate so re-running this script
    # never leaves stale/duplicate vectors behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"], "heading": c["heading"]} for c in chunks],
    )
    print(f"Stored {collection.count()} vectors in Chroma at {DB_DIR}/ (collection: {COLLECTION_NAME})")

    # Sanity-check retrieval with a couple of manual queries before we ever
    # touch generation - if retrieval doesn't return sensible chunks, no
    # amount of prompt engineering downstream will fix that.
    sanity_queries = [
        "how do I validate a query parameter",
        "how does dependency injection work in FastAPI",
    ]
    for q in sanity_queries:
        q_embedding = model.encode([q]).tolist()
        results = collection.query(query_embeddings=q_embedding, n_results=3)
        print(f"\nQuery: {q!r}")
        for i, (doc_id, doc, meta) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        ):
            preview = doc[:120].replace("\n", " ")
            print(f"  {i+1}. [{doc_id}] {meta['heading']!r} -> {preview}...")


if __name__ == "__main__":
    main()
