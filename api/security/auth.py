"""
MythosForge API — Authentication and authorization.

Uses simple API key authentication via Bearer token or X-API-Key header.
Disabled by default — enable with MYTHOSFORGE_AUTH_ENABLED=true.
"""

from __future__ import annotations

import secrets
import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.settings import settings

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


def _validate_api_key(request: Request) -> Optional[str]:
    """Extract and validate API key from request headers."""
    if not settings.auth_enabled:
        return None

    # Check X-API-Key header
    api_key = request.headers.get("x-api-key")

    # Check Authorization: Bearer <key>
    if not api_key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            api_key = auth[7:].strip()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing_api_key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if settings.api_keys and api_key not in settings.api_keys:
        logger.warning("Invalid API key attempt from remote")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_api_key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return api_key


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[str]:
    """Dependency: require authentication if auth is enabled."""
    if not settings.auth_enabled:
        return None
    return _validate_api_key(request)


async def optional_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[str]:
    """Dependency: extract auth if present, but don't require it."""
    if not settings.auth_enabled:
        return None
    try:
        return _validate_api_key(request)
    except HTTPException:
        return None


def generate_api_key() -> str:
    """Generate a new random API key for configuration."""
    return f"mf_{secrets.token_urlsafe(32)}"
