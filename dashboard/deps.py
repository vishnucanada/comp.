"""FastAPI dependencies: API-key authentication."""

from fastapi import HTTPException, Request

from .config import API_KEY

__all__ = ["_require_auth"]


def _require_auth(request: Request) -> None:
    """Dependency that enforces the X-API-Key header when API_KEY is set."""
    if not API_KEY:
        return
    if request.headers.get("X-API-Key", "") != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")
