"""
Shared exceptions and error handling.
"""

from typing import Any

import structlog
from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)


class AppException(Exception):
    """Base application exception."""
    
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class InputValidationError(AppException):
    """Input validation error."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, "VALIDATION_ERROR", status.HTTP_400_BAD_REQUEST, details)


class NotFoundError(AppException):
    """Resource not found error."""
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            f"{resource} not found: {identifier}",
            "NOT_FOUND",
            status.HTTP_404_NOT_FOUND,
            {"resource": resource, "identifier": identifier},
        )


class UnauthorizedError(AppException):
    """Authentication error."""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, "UNAUTHORIZED", status.HTTP_401_UNAUTHORIZED)


class ForbiddenError(AppException):
    """Authorization error."""
    def __init__(self, message: str = "Access denied"):
        super().__init__(message, "FORBIDDEN", status.HTTP_403_FORBIDDEN)


class RateLimitError(AppException):
    """Rate limit exceeded error."""
    def __init__(self, retry_after: int):
        super().__init__(
            "Rate limit exceeded",
            "RATE_LIMITED",
            status.HTTP_429_TOO_MANY_REQUESTS,
            {"retry_after": retry_after},
        )


class ExternalServiceError(AppException):
    """External service (LLM, DB, etc.) error."""
    def __init__(self, service: str, message: str):
        super().__init__(
            f"{service} error: {message}",
            "EXTERNAL_SERVICE_ERROR",
            status.HTTP_502_BAD_GATEWAY,
            {"service": service},
        )


# =============================================================================
# Exception Handlers
# =============================================================================

async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle application exceptions."""
    logger.warning(
        "app_exception",
        code=exc.code,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "details": {},
            }
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.warning(
        "validation_exception",
        errors=exc.errors(),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
            }
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(
        "unhandled_exception",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {},
            }
        },
    )


def register_exception_handlers(app):
    """Register all exception handlers on a FastAPI app."""
    from fastapi.exceptions import RequestValidationError
    
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)