"""
Document ingestion - Embedding and vector storage.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.shared import get_config, get_logger, track_operation

config = get_config()
logger = get_logger(__name__)

CHUNKS_PATH = Path("data/chunks.jsonl")
DB_DIR = "data/chroma"
COLLECTION_NAME = "httpx_kb"


def load_chunks():
    """Load chunks from JSONL file."""
    chunks = []
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(json.loads(line))
    return chunks


@track_operation("embed_chunks")
def embed_chunks(chunks: list, model_name: str | None = None, device: str | None = None):
    """Generate embeddings for chunks."""
    model_name = model_name or config.EMBEDDING_MODEL
    device = device or config.EMBEDDING_DEVICE
    
    logger.info("loading_embedding_model", model=model_name, device=device)
    model = SentenceTransformer(model_name, device=device)
    
    texts = [c["text"] for c in chunks]
    logger.info("embedding_chunks", count=len(texts))
    
    embeddings = model.encode(
        texts, 
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()
    
    return embeddings


@track_operation("store_vectors")
def store_vectors(chunks: list, embeddings: list):
    """Store vectors in Chroma."""
    client = chromadb.PersistentClient(path=DB_DIR)
    
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("deleted_existing_collection")
    except chromadb.errors.NotFoundError:
        logger.debug("collection_not_found", collection=COLLECTION_NAME)
    except chromadb.errors.ChromaError as e:
        logger.warning("delete_collection_failed", error=str(e))
    
    collection = client.create_collection(COLLECTION_NAME)
    
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "heading": c["heading"]} for c in chunks],
    )
    
    logger.info("vectors_stored", count=collection.count(), collection=COLLECTION_NAME)
    return collection


@track_operation("sanity_check")
def sanity_check(collection, model):
    """Run sanity check queries."""
    sanity_queries = [
        "how does httpx handle connection pooling and limits",
        "what does the Client.request method do internally",
        "how is authentication implemented in httpx",
        "how do I stream a response",
        "what exceptions can httpx raise on a timeout",
    ]
    
    for q in sanity_queries:
        q_embedding = model.encode([q]).tolist()
        results = collection.query(query_embeddings=q_embedding, n_results=3)
        
        logger.info("sanity_check_query", query=q)
        for i, (doc_id, doc, meta) in enumerate(zip(
            results["ids"][0], 
            results["documents"][0], 
            results["metadatas"][0]
        )):
            preview = doc[:120].replace("\n", " ")
            logger.info(
                "sanity_check_result",
                rank=i + 1,
                id=doc_id,
                heading=meta["heading"],
                preview=preview,
            )


def main():
    """Main embedding entry point."""
    chunks = load_chunks()
    logger.info("loaded_chunks", count=len(chunks), path=str(CHUNKS_PATH))
    
    embeddings = embed_chunks(chunks)
    
    collection = store_vectors(chunks, embeddings)
    
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)
    sanity_check(collection, model)
    
    logger.info("embedding_pipeline_complete")


if __name__ == "__main__":
    main()