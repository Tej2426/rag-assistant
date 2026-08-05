"""
Shared utilities package for AI Portfolio Projects.
"""

from .app_factory import create_app, mount_subapp
from .auth import limiter, rate_limit, require_auth, setup_rate_limiting, verify_api_key
from .config import BaseConfig, get_config
from .observability import (
    ObservabilityMiddleware,
    business_operation_duration,
    business_operations_total,
    configure_logging,
    get_logger,
    http_request_duration,
    http_requests_total,
    metrics_endpoint,
    track_operation,
    track_timing,
)

__all__ = [
    # App factory
    "create_app",
    "mount_subapp",
    # Config
    "BaseConfig",
    "ObservabilityMiddleware",
    "business_operation_duration",
    "business_operations_total",
    # Observability
    "configure_logging",
    "get_config",
    "get_logger",
    "http_request_duration",
    "http_requests_total",
    "limiter",
    "metrics_endpoint",
    "rate_limit",
    "require_auth",
    "setup_rate_limiting",
    "track_operation",
    "track_timing",
    # Auth
    "verify_api_key",
]