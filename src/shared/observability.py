"""
Shared observability module - structured logging, metrics, tracing.
All projects import this for consistent observability.
"""

import os
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps

import structlog
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

# =============================================================================
# Structured Logging Setup
# =============================================================================

def configure_logging(service_name: str, log_level: str = "INFO") -> None:
    """Configure structlog for the service."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if os.getenv("LOG_FORMAT") == "json" 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set standard library logging level
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# =============================================================================
# Prometheus Metrics
# =============================================================================

# Standard HTTP metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently in progress",
    ["method", "endpoint"]
)

# Business metrics (customize per project)
business_operations_total = Counter(
    "business_operations_total",
    "Total business operations",
    ["operation", "status"]
)

business_operation_duration = Histogram(
    "business_operation_duration_seconds",
    "Business operation duration in seconds",
    ["operation"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)


# =============================================================================
# FastAPI Middleware
# =============================================================================

class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and metrics."""

    def __init__(self, app, service_name: str):
        super().__init__(app)
        self.service_name = service_name
        self.logger = get_logger(f"{service_name}.http")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Add request ID to context
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        # Track in-progress
        http_requests_in_progress.labels(
            method=request.method, 
            endpoint=request.url.path
        ).inc()
        
        # Log request
        self.logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_host=request.client.host if request.client else None,
        )
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            self.logger.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
            )
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            http_requests_in_progress.labels(
                method=request.method, 
                endpoint=request.url.path
            ).dec()
            
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code
            ).inc()
            
            http_request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Log response
            self.logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=round(duration * 1000, 2),
            )
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-MS"] = str(round(duration * 1000, 2))
        
        return response


# =============================================================================
# Metrics Endpoint
# =============================================================================

async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


# =============================================================================
# Decorators for Business Operations
# =============================================================================

def track_operation(operation_name: str):
    """Decorator to track business operation metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            logger = get_logger(f"operation.{operation_name}")
            try:
                result = await func(*args, **kwargs)
                business_operations_total.labels(
                    operation=operation_name, status="success"
                ).inc()
                logger.info("operation_succeeded", operation=operation_name)
                return result
            except Exception as e:
                business_operations_total.labels(
                    operation=operation_name, status="error"
                ).inc()
                logger.exception("operation_failed", operation=operation_name, error=str(e))
                raise
            finally:
                business_operation_duration.labels(
                    operation=operation_name
                ).observe(time.time() - start)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            logger = get_logger(f"operation.{operation_name}")
            try:
                result = func(*args, **kwargs)
                business_operations_total.labels(
                    operation=operation_name, status="success"
                ).inc()
                logger.info("operation_succeeded", operation=operation_name)
                return result
            except Exception as e:
                business_operations_total.labels(
                    operation=operation_name, status="error"
                ).inc()
                logger.exception("operation_failed", operation=operation_name, error=str(e))
                raise
            finally:
                business_operation_duration.labels(
                    operation=operation_name
                ).observe(time.time() - start)
        
        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# =============================================================================
# Context Managers
# =============================================================================

@contextmanager
def track_timing(operation: str, logger: structlog.stdlib.BoundLogger | None = None):
    """Context manager to time an operation."""
    log = logger or get_logger(f"timing.{operation}")
    start = time.time()
    try:
        yield
        log.info("timing_completed", operation=operation, duration_ms=round((time.time() - start) * 1000, 2))
    except Exception as e:
        log.exception("timing_failed", operation=operation, error=str(e))
        raise