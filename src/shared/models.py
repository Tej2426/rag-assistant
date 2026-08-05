"""
Shared Pydantic models and response types.
"""

from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Generic Response Models
# =============================================================================

T = TypeVar("T")


class ResponseMeta(BaseModel):
    """Metadata for API responses."""
    request_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0"


class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response envelope."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "data": {"key": "value"},
            "meta": {"request_id": "abc123", "timestamp": "2024-01-01T00:00:00Z", "version": "1.0"}
        }
    })
    success: bool = True
    data: T
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class ErrorDetail(BaseModel):
    """Error detail structure."""
    code: str
    message: str
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    """Standard error response envelope."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": False,
            "error": {"code": "NOT_FOUND", "message": "Resource not found", "details": {}},
            "meta": {"request_id": "abc123", "timestamp": "2024-01-01T00:00:00Z", "version": "1.0"}
        }
    })
    success: bool = False
    error: ErrorDetail
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


# =============================================================================
# Pagination
# =============================================================================

class PaginationParams(BaseModel):
    """Pagination query parameters."""
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response envelope."""
    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


def paginate(items: list[T], params: PaginationParams) -> PaginatedResponse[T]:
    """Create a paginated response from a list."""
    total = len(items)
    start = (params.page - 1) * params.page_size
    end = start + params.page_size
    paginated_items = items[start:end]
    total_pages = (total + params.page_size - 1) // params.page_size
    
    return PaginatedResponse(
        items=paginated_items,
        total=total,
        page=params.page,
        page_size=params.page_size,
        total_pages=total_pages,
        has_next=params.page < total_pages,
        has_prev=params.page > 1,
    )


# =============================================================================
# Common Domain Models
# =============================================================================

class TimestampedModel(BaseModel):
    """Base model with timestamps."""
    model_config = ConfigDict(from_attributes=True)
    created_at: datetime
    updated_at: datetime | None = None


class IDModel(BaseModel):
    """Base model with ID."""
    id: UUID = Field(default_factory=uuid4)


class SourceReference(BaseModel):
    """Reference to a source document/chunk."""
    source_id: str
    source_type: str  # "document", "chunk", "url", etc.
    title: str | None = None
    url: str | None = None
    relevance_score: float | None = None


# =============================================================================
# Health & Status
# =============================================================================

class HealthCheck(BaseModel):
    """Health check response."""
    status: str  # "healthy", "degraded", "unhealthy"
    service: str
    version: str
    checks: dict[str, bool] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ServiceInfo(BaseModel):
    """Service information."""
    name: str
    version: str
    description: str
    environment: str
    uptime_seconds: float
    dependencies: dict[str, str] = {}  # name -> version/status