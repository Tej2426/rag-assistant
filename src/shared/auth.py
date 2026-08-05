"""
Shared authentication & authorization module.
API key based auth with rate limiting.
"""

from collections.abc import Callable
from functools import wraps

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from .config import get_config

config = get_config()

# API Key security scheme
api_key_header = APIKeyHeader(name=config.API_KEY_HEADER, auto_error=False)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


async def verify_api_key(
    request: Request,
    api_key: str | None = Depends(api_key_header),
) -> str:
    """Verify API key if configured."""
    if config.API_KEY is None:
        # No API key configured - allow all (development mode)
        return "development"
    
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication on an endpoint."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # The actual verification happens via Depends(verify_api_key)
        # This decorator is for documentation purposes
        return await func(*args, **kwargs)
    return wrapper


def rate_limit(limit: str = "100/minute"):
    """Decorator for rate limiting specific endpoints."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return limiter.limit(limit)(wrapper)
    return decorator


# Rate limit exceeded handler
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
        headers={"Retry-After": str(exc.retry_after)},
    )


def setup_rate_limiting(app):
    """Setup rate limiting on FastAPI app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)