"""
MythosForge API — Safe error handlers.

Error responses never leak internal details (stack traces, file paths, etc.)
to the client.  Full details are logged server-side for debugging.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("mythosforge.api.errors")


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id", "unknown")


class AppError(Exception):
    """Base application error with safe public detail."""

    def __init__(self, detail: str, status_code: int = 500):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


class NotFoundError(AppError):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=404)


class ValidationError(AppError):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=422)


async def not_found_handler(request: Request, exc: NotFoundError):
    rid = _request_id(request)
    logger.warning("NotFound %s %s [%s]: %s", request.method, request.url.path, rid, exc.detail)
    return JSONResponse(
        status_code=404,
        content={"detail": exc.detail, "request_id": rid},
    )


async def validation_error_handler(request: Request, exc: ValidationError):
    rid = _request_id(request)
    logger.warning("Validation %s %s [%s]: %s", request.method, request.url.path, rid, exc.detail)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.detail, "request_id": rid},
    )


async def app_error_handler(request: Request, exc: AppError):
    rid = _request_id(request)
    logger.error("AppError %s %s [%s]: %s", request.method, request.url.path, rid, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": rid},
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = _request_id(request)
    error_id = uuid.uuid4().hex[:12]
    logger.exception(
        "Unhandled %s %s [%s] error_id=%s: %s",
        request.method, request.url.path, rid, error_id, exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "internal_server_error",
            "error_id": error_id,
            "request_id": rid,
        },
    )
