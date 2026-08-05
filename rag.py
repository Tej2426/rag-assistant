"""
Shared RAG pipeline (retrieval + generation) - used by both the CLI
(03_generate.py) and the FastAPI app (app.py), so the logic lives in one
place instead of being copy-pasted into each entry point.
"""

import os

import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_DIR = "data/chroma"
COLLECTION_NAME = "fastapi_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
N_RESULTS = 4

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about FastAPI \
using ONLY the documentation excerpts below. If the excerpts don't contain \
the answer, say you don't know instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


class RagPipeline:
    """Loads the embedding model + Chroma collection once, then answers
    questions cheaply. Loading the model per-request would add multi-second
    latency to every call - it's loaded once at startup instead."""

    def __init__(self):
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, question: str):
        query_embedding = self.embed_model.encode([question]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=N_RESULTS)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return list(zip(docs, metas))

    def build_prompt(self, question: str, chunks):
        context = "\n\n---\n\n".join(f"[{meta['heading']}]\n{doc}" for doc, meta in chunks)
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(self, prompt: str) -> str:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def answer(self, question: str):
        chunks = self.retrieve(question)
        prompt = self.build_prompt(question, chunks)
        reply = self.generate(prompt)
        return reply, chunks
