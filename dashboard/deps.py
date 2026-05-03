"""FastAPI dependencies: rate limiter and API-key authentication."""

from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi import _rate_limit_exceeded_handler as rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import API_KEY

__all__ = ["limiter", "rate_limit_exceeded_handler", "RateLimitExceeded", "_require_auth"]

limiter = Limiter(key_func=get_remote_address)


def _require_auth(request: Request) -> None:
    """Dependency that enforces the X-API-Key header when API_KEY is set."""
    if not API_KEY:
        return
    if request.headers.get("X-API-Key", "") != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")
