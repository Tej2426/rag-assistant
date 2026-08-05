"""
RAG Assistant - Main application entry point.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.api.routes import set_pipeline_instance
from src.rag.pipeline import RagPipeline
from src.shared import (
    create_app,
    get_config,
    get_logger,
)
from src.shared.models import HealthCheck

config = get_config()
logger = get_logger(__name__)

# Global pipeline instance
pipeline: RagPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize pipeline on startup."""
    global pipeline
    
    logger.info("application_starting", version=config.APP_VERSION)
    
    # Initialize pipeline
    pipeline = RagPipeline()
    await pipeline.initialize()
    
    # Set pipeline instance for API routes
    set_pipeline_instance(pipeline)
    
    logger.info("pipeline_initialized", collection=pipeline.collection.name)
    
    yield
    
    logger.info("application_shutting_down")


# Create app with shared factory
app = create_app(
    title="RAG Document Q&A Assistant",
    description="Production-ready RAG system for querying documentation with grounded answers and citations.",
    version=config.APP_VERSION,
    lifespan=lifespan,
    static_dir="ui/static" if Path("ui/static").exists() else None,
    templates_dir="ui/templates",
)

# Templates
templates = Jinja2Templates(directory="ui/templates")


# =============================================================================
# UI Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def index(request: Request):
    """Landing page."""
    return templates.TemplateResponse(request, "rag_layout.html", {
        "request": request,
        "PROJECT_NAME": config.APP_NAME,
        "APP_VERSION": config.APP_VERSION,
        "page_title": "httpx Codebase & Docs Q&A Assistant",
        "page_description": "Ask questions about the httpx source code and documentation, and get grounded answers with citations.",
    })


@app.get("/playground", response_class=HTMLResponse, tags=["ui"])
async def playground(request: Request):
    """Interactive RAG playground."""
    example_questions = [
        "How do I send a GET request with query parameters using httpx?",
        "What does the Client.request method actually do internally?",
        "How is authentication implemented in httpx's source code?",
        "What's the difference between httpx.Client and httpx.AsyncClient?",
        "How does httpx handle connection pooling and limits?",
        "What exceptions can be raised on a timeout?",
        "How do I use a custom transport?",
        "How do I stream a large response body?",
    ]
    
    return templates.TemplateResponse(request, "playground.html", {
        "request": request,
        "PROJECT_NAME": config.APP_NAME,
        "APP_VERSION": config.APP_VERSION,
        "example_questions": example_questions,
        "default_question": "",
        "sources": None,
    })


@app.get("/eval", response_class=HTMLResponse, tags=["ui"])
async def evaluation_dashboard(request: Request):
    """Evaluation dashboard."""
    import json
    from pathlib import Path
    
    eval_file = Path("eval/results/latest.json")
    metrics = {
        "context_precision": 0.0,
        "context_recall": 0.0,
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
    }
    results = []
    last_run = "Never"
    
    if eval_file.exists():
        with open(eval_file) as f:
            data = json.load(f)
            metrics = data.get("aggregate", metrics)
            results = data.get("results", [])
            last_run = data.get("timestamp", "Unknown")
    
    return templates.TemplateResponse(request, "eval_dashboard.html", {
        "request": request,
        "PROJECT_NAME": config.APP_NAME,
        "APP_VERSION": config.APP_VERSION,
        "metrics": metrics,
        "results": results,
        "last_run": last_run,
    })


# =============================================================================
# API Routes
# =============================================================================

from src.api.routes import router as api_router

app.include_router(api_router, prefix="/api")


# =============================================================================
# Health & Metrics
# =============================================================================

@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """Health check endpoint."""
    checks = {}
    if pipeline:
        checks["pipeline"] = pipeline.is_healthy()
        checks["chroma"] = pipeline.chroma_healthy()
        checks["groq"] = await pipeline.groq_healthy()
    
    return HealthCheck(
        status="healthy" if all(checks.values()) else "degraded",
        service=config.APP_NAME,
        version=config.APP_VERSION,
        checks=checks,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)