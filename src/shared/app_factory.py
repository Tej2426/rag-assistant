"""
Shared base FastAPI application factory.
All projects use this to create their app with consistent middleware, routes, and setup.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .auth import setup_rate_limiting, verify_api_key
from .config import get_config
from .observability import (
    ObservabilityMiddleware,
    configure_logging,
    get_logger,
    metrics_endpoint,
)

config = get_config()
logger = get_logger(__name__)


def create_app(
    title: str,
    description: str = "",
    version: str = "0.1.0",
    lifespan=None,
    include_metrics: bool = True,
    include_auth: bool = True,
    static_dir: str | None = None,
    templates_dir: str | None = None,
) -> FastAPI:
    """Create a FastAPI app with standard configuration."""
    
    # Configure logging first
    configure_logging(title.lower().replace(" ", "-"), config.LOG_LEVEL)
    
    @asynccontextmanager
    async def default_lifespan(app: FastAPI):
        logger.info("application_starting", app_name=title)
        yield
        logger.info("application_shutting_down", app_name=title)
    
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan or default_lifespan,
        docs_url="/docs" if config.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if config.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if config.ENVIRONMENT != "production" else None,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=config.CORS_ALLOW_CREDENTIALS,
        allow_methods=config.CORS_ALLOW_METHODS,
        allow_headers=config.CORS_ALLOW_HEADERS,
    )
    
    # Observability middleware
    app.add_middleware(ObservabilityMiddleware, service_name=title.lower().replace(" ", "-"))
    
    # Rate limiting
    setup_rate_limiting(app)
    
    # Static files
    if static_dir:
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Templates
    templates = None
    if templates_dir:
        templates = Jinja2Templates(directory=templates_dir)
        app.state.templates = templates
    
    # Metrics endpoint
    if include_metrics and config.PROMETHEUS_ENABLED:
        @app.get("/metrics", tags=["monitoring"], include_in_schema=False)
        async def metrics(request):
            return await metrics_endpoint(request)
    
    # Auth dependency for protected routes
    if include_auth:
        app.dependency_overrides[verify_api_key] = verify_api_key
    
    logger.info("application_created", app_name=title, version=version)
    return app


def mount_subapp(parent: FastAPI, path: str, subapp: FastAPI):
    """Mount a sub-application at a path."""
    parent.mount(path, subapp)
    logger.info("subapp_mounted", path=path, subapp=subapp.title)