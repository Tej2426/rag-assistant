"""
API Routes for RAG Assistant.
"""


import json
from datetime import datetime

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.rag.pipeline import RagPipeline
from src.shared import get_logger, limiter, verify_api_key

router = APIRouter()
logger = get_logger(__name__)


# =============================================================================
# Dependency Injection
# =============================================================================

_pipeline_instance: RagPipeline | None = None


def set_pipeline_instance(pipeline: RagPipeline):
    """Set the pipeline instance (called from main.py)."""
    global _pipeline_instance
    _pipeline_instance = pipeline


def get_pipeline() -> RagPipeline:
    """Get the pipeline instance."""
    if _pipeline_instance is None:
        raise RuntimeError("Pipeline not initialized")
    return _pipeline_instance


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(4, ge=1, le=20)
    model: str | None = Field(None, description="Groq model to use")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    stream: bool = Field(False)


class SourceResponse(BaseModel):
    id: str
    text: str
    heading: str
    source: str
    score: float
    chars: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    latency_ms: int
    model: str
    request_id: str


class EvalRunRequest(BaseModel):
    dataset: str = Field("default", description="Evaluation dataset name")
    metrics: list[str] = Field(
        default=["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    )


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    sources: list[str]
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = None


# =============================================================================
# Query Endpoint
# =============================================================================

@router.post("/query", response_model=QueryResponse, tags=["query"])
@limiter.limit("30/minute")
async def query(
    request: Request,
    query_req: QueryRequest,
    api_key: str = Depends(verify_api_key),
    rag: RagPipeline = Depends(get_pipeline),
):
    """Query the RAG system."""
    logger.info("api_query", question=query_req.question[:100])
    
    response = await rag.answer(
        question=query_req.question,
        top_k=query_req.top_k,
        model=query_req.model,
        temperature=query_req.temperature,
    )
    
    return QueryResponse(
        answer=response.answer,
        sources=[
            SourceResponse(
                id=s.id,
                text=s.text,
                heading=s.heading,
                source=s.source,
                score=s.score,
                chars=s.chars,
            )
            for s in response.sources
        ],
        latency_ms=response.latency_ms,
        model=response.model,
        request_id=response.request_id,
    )


# =============================================================================
# HTMX Partial Endpoints (for UI)
# =============================================================================

@router.post("/query/ui", response_class=HTMLResponse, tags=["ui"])
@limiter.limit("30/minute")
async def query_ui(
    request: Request,
    question: str = Form(...),
    top_k: int = Form(4),
    model: str = Form("llama-3.3-70b-versatile"),
    temperature: float = Form(0.1),
    show_sources: bool = Form(True),
    rag: RagPipeline = Depends(get_pipeline),
):
    """HTMX endpoint for playground query."""
    response = await rag.answer(
        question=question,
        top_k=top_k,
        model=model,
        temperature=temperature,
    )
    
    from src.main import templates
    return templates.TemplateResponse(request, "_response.html", {
        "request": request,
        "answer": response.answer,
        "sources": response.sources if show_sources else [],
        "latency_ms": response.latency_ms,
        "model": response.model,
    })


# =============================================================================
# Evaluation Endpoints
# =============================================================================

@router.post("/eval/run", response_class=HTMLResponse, tags=["eval"])
@limiter.limit("5/minute")
async def run_evaluation(
    request: Request,
    dataset: str = Form("default"),
    rag: RagPipeline = Depends(get_pipeline),
):
    """Run evaluation and return results."""
    from src.eval.runner import run_evaluation
    
    logger.info("api_eval_run", dataset=dataset)
    
    results = await run_evaluation(rag, dataset)
    
    from src.main import templates
    return templates.TemplateResponse(request, "_eval_results.html", {
        "request": request,
        "results": results,
    })


@router.get("/eval/run/stream", tags=["eval"])
@limiter.limit("5/minute")
async def run_evaluation_stream_endpoint(
    request: Request,
    dataset: str = Query("default"),
    rag: RagPipeline = Depends(get_pipeline),
):
    """Server-Sent Events stream of eval progress - one event per question
    per stage, so the eval dashboard can show a live progress bar instead
    of a blank wait during a run that takes well over a minute."""
    from src.eval.runner import run_evaluation_stream

    logger.info("api_eval_run_stream", dataset=dataset)

    async def event_source():
        async for event in run_evaluation_stream(rag, dataset):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.get("/eval/results", tags=["eval"])
async def get_evaluation_results(
    dataset: str = Query("default"),
    rag: RagPipeline = Depends(get_pipeline),
):
    """Get latest evaluation results."""
    import json
    from pathlib import Path
    
    eval_file = Path(f"eval/results/{dataset}.json")
    if not eval_file.exists():
        return {"results": [], "aggregate": {}}
    
    with open(eval_file) as f:
        return json.load(f)


# =============================================================================
# Ingestion Endpoints (Admin)
# =============================================================================

@router.post("/ingest/chunk", tags=["ingest"])
@limiter.limit("10/minute")
async def ingest_chunk(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Re-run chunking pipeline."""
    from src.ingestion.chunker import main as chunk_main
    
    logger.info("api_ingest_chunk")
    chunk_main()
    
    return {"status": "ok", "message": "Chunking completed"}


@router.post("/ingest/embed", tags=["ingest"])
@limiter.limit("10/minute")
async def ingest_embed(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Re-run embedding pipeline."""
    from src.ingestion.embedder import main as embed_main
    
    logger.info("api_ingest_embed")
    embed_main()
    
    return {"status": "ok", "message": "Embedding completed"}


# =============================================================================
# Feedback Endpoint
# =============================================================================

@router.post("/feedback", tags=["feedback"])
@limiter.limit("60/minute")
async def submit_feedback(
    request: Request,
    feedback: FeedbackRequest,
    api_key: str = Depends(verify_api_key),
):
    """Submit user feedback for evaluation dataset."""
    import json
    from pathlib import Path
    
    feedback_dir = Path("eval/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    
    feedback_file = feedback_dir / f"feedback_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": feedback.question,
        "answer": feedback.answer,
        "sources": feedback.sources,
        "rating": feedback.rating,
        "comment": feedback.comment,
    }
    
    with open(feedback_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    logger.info("feedback_received", rating=feedback.rating)
    return {"status": "ok", "message": "Feedback recorded"}