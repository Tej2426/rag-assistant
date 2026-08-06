"""
RAG Pipeline - Core retrieval and generation logic.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass

import chromadb
import httpx
from sentence_transformers import SentenceTransformer

from src.shared import get_config, get_logger, track_operation

config = get_config()
logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata."""
    id: str
    text: str
    heading: str
    source: str
    score: float
    chars: int


@dataclass
class RagResponse:
    """Response from the RAG pipeline."""
    answer: str
    sources: list[RetrievedChunk]
    latency_ms: int
    model: str
    request_id: str


PROMPT_TEMPLATE = """You are a helpful assistant answering questions about the httpx Python library using ONLY the documentation and source code excerpts below. Excerpts may be prose documentation or Python source code - read code excerpts as ground truth for how the library actually behaves. If the excerpts don't contain the answer, say you don't know instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


class RagPipeline:
    """Main RAG pipeline: retrieval + generation."""
    
    def __init__(self):
        self.embed_model: SentenceTransformer | None = None
        self.chroma_client: chromadb.PersistentClient | None = None
        self.collection = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize embedding model and Chroma connection."""
        if self._initialized:
            return
        
        logger.info("initializing_pipeline")
        
        # Load embedding model (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        self.embed_model = await loop.run_in_executor(
            None, 
            lambda: SentenceTransformer(config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)
        )
        
        # Connect to Chroma
        self.chroma_client = chromadb.PersistentClient(path="data/chroma")
        self.collection = self.chroma_client.get_collection(config.CHROMA_COLLECTION)
        
        self._initialized = True
        logger.info("pipeline_initialized", chunks=self.collection.count())
    
    @track_operation("retrieve")
    def retrieve(self, question: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a question."""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized")
        
        # Embed query
        query_embedding = self.embed_model.encode([question]).tolist()[0]
        
        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, meta, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score (Chroma uses L2 distance)
                score = 1.0 / (1.0 + distance)
                
                chunks.append(RetrievedChunk(
                    id=results["ids"][0][i],
                    text=doc,
                    heading=meta.get("heading", "(intro)"),
                    source=meta.get("source", "unknown"),
                    score=score,
                    chars=len(doc),
                ))
        
        logger.info("retrieval_completed", question=question[:50], num_chunks=len(chunks))
        return chunks
    
    def build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Build the prompt with retrieved context."""
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[{chunk.heading}]\n{chunk.text}")
        
        context = "\n\n---\n\n".join(context_parts)
        return PROMPT_TEMPLATE.format(context=context, question=question)
    
    @track_operation("generate")
    async def generate(self, prompt: str, model: str | None = None, temperature: float = 0.1) -> str:
        """Generate answer using the Groq API (OpenAI-compatible chat completions).

        Retries on 429 (rate limit) with backoff - a transient provider
        rate limit shouldn't surface as a 500 to the user when waiting a
        couple seconds and retrying almost always succeeds."""
        model = model or config.GROQ_MODEL
        max_retries = 3

        async with httpx.AsyncClient(timeout=config.GROQ_TIMEOUT) as client:
            for attempt in range(max_retries):
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                    },
                )
                if response.status_code == 429 and attempt < max_retries - 1:
                    retry_after = float(response.headers.get("retry-after", 2 ** attempt))
                    logger.warning("groq_rate_limited", attempt=attempt + 1, retry_after=retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
    
    async def answer(
        self, 
        question: str, 
        top_k: int = 4,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> RagResponse:
        """Full RAG pipeline: retrieve -> generate."""
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        logger.info("answering_question", question=question[:100], request_id=request_id)
        
        # Retrieve
        chunks = self.retrieve(question, top_k)
        
        # Build prompt
        prompt = self.build_prompt(question, chunks)
        
        # Generate
        answer = await self.generate(prompt, model, temperature)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        response = RagResponse(
            answer=answer.strip(),
            sources=chunks,
            latency_ms=latency_ms,
            model=model or config.GROQ_MODEL,
            request_id=request_id,
        )
        
        logger.info("answer_generated", request_id=request_id, latency_ms=latency_ms, num_sources=len(chunks))
        return response
    
    def is_healthy(self) -> bool:
        """Check if pipeline is healthy."""
        return self._initialized and self.collection is not None
    
    def chroma_healthy(self) -> bool:
        """Check Chroma connection."""
        try:
            self.collection.count()
            return True
        except chromadb.errors.ChromaError:
            return False
    
    async def groq_healthy(self) -> bool:
        """Check Groq API is reachable and configured."""
        if not config.GROQ_API_KEY:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                )
                return resp.status_code == 200
        except httpx.HTTPError:
            return False