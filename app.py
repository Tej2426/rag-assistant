"""
Step 4: FastAPI wrapper

Turns the RAG pipeline into a real HTTP service - the shape Project 3
(multi-agent system) will call this project as a tool through later.

WHY LOAD THE PIPELINE AT STARTUP, NOT PER-REQUEST:
Loading the embedding model and connecting to Chroma takes real time
(model weights, disk I/O). Doing that once in a lifespan handler, and
reusing the same RagPipeline instance for every request, is what makes
each individual request fast.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from rag import RagPipeline

pipeline: RagPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RagPipeline()
    yield


app = FastAPI(title="RAG Document Q&A Assistant", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    source: str
    heading: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    reply, chunks = pipeline.answer(request.question)
    sources = [
        SourceChunk(source=meta["source"], heading=meta["heading"], text=doc)
        for doc, meta in chunks
    ]
    return QueryResponse(answer=reply, sources=sources)
