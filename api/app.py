"""
MythosForge API — Main application (v0.5.0).

Architecture:
- Settings loaded from environment (Pydantic Settings)
- Restricted CORS (allowlist, not wildcard)
- Safe error handlers (no internal detail leaks)
- Request ID tracking + structured logging
- Optional API key authentication
- In-process metrics collection
- Layered routers: public catalog, inference, system

All catalog routes live in routers/public.py.
Inference routes live in routers/inference.py.
"""

import time
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api import __version__, __api_version__
from api.settings import settings
from api.errors import (
    AppError, NotFoundError,
    not_found_handler, app_error_handler, unhandled_exception_handler,
)
from api.deps import build_health_response
from api.observability.metrics import metrics
from api.models import (
    APIRootResponse, HealthResponse,
)
from api.routers.public import router as public_router
from api.routers.inference import router as inference_router

# ─── Logging ──────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format=settings.log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mythosforge.api")

# ─── Lifespan ─────────────────────────────────────────

_start_time = time.perf_counter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "MythosForge API v%s starting | env=%s | auth=%s | metrics=%s",
        __version__, settings.env, settings.auth_enabled, settings.metrics_enabled,
    )
    yield
    logger.info("MythosForge API shutting down.")


# ─── App ──────────────────────────────────────────────

app = FastAPI(
    title="MythosForge API",
    description=(
        "## MythosForge — Recurrent-Depth Transformer Research Lab\n\n"
        "REST API para el laboratorio de investigacion MythosForge.\n\n"
        "### Arquitectura\n"
        "Prelude -> **Recurrent Block** (xN loops) -> Coda\n\n"
        "### Componentes\n"
        "- **MLA / GQA**: Atencion conmutable (DeepSeek-V2 / GQA)\n"
        "- **MoE**: Mixture of Experts con routed + shared experts\n"
        "- **LTI**: Inyeccion estable Linear Time-Invariant, A in (0,1)\n"
        "- **ACT**: Adaptive Computation Time — halting por posicion\n"
        "- **LoRA**: Adaptador per-iteracion con senal de profundidad\n\n"
        "### Seguridad\n"
        f"Autenticacion: {'habilitada' if settings.auth_enabled else 'deshabilitada (dev)'}\n\n"
        "### Aviso\n"
        "OpenMythos es una reconstruccion teorica independiente, no afiliada a Anthropic.\n"
        "MythosForge no afirma equivalencia con ningun sistema propietario.\n\n"
        "---\n\n"
        "*Creado por [Smouj013](https://github.com/smouj) con agentes de IA — Abril 2026*"
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    license_info={
        "name": "MIT",
        "url": "https://github.com/smouj/MythosForge/blob/main/LICENSE",
    },
    contact={
        "name": "Smouj013",
        "url": "https://github.com/smouj",
    },
)

# ─── CORS (restricted) ──────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# ─── Exception handlers (safe — no detail leaks) ────

app.add_exception_handler(NotFoundError, not_found_handler)
app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# ─── Include routers ──────────────────────────────────

app.include_router(inference_router)
app.include_router(public_router)


# ─── Middleware: Request ID + logging + metrics ──────

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    start = time.perf_counter()

    # Assign request ID if not present
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid

    # Process request
    response = await call_next(request)

    # Metrics
    duration_ms = (time.perf_counter() - start) * 1000
    path = request.url.path
    metrics.inc("http_requests_total")
    metrics.inc(f"http_requests_total_{response.status_code}")
    metrics.observe_latency(path, duration_ms)

    # Structured log
    logger.info(
        "%s %s -> %d (%.1fms) [%s]",
        request.method, path, response.status_code, duration_ms, rid,
    )

    # Add request ID to response
    response.headers["X-Request-ID"] = rid

    return response


# ═══════════════════════════════════════════════════════
# SYSTEM ROUTES (kept in app.py — thin layer)
# ═══════════════════════════════════════════════════════

@app.get("/", response_model=APIRootResponse, tags=["Root"])
async def api_root():
    """API root — Returns links to documentation and available endpoints."""
    return APIRootResponse(
        name="MythosForge API",
        version=__version__,
        api_version=__api_version__,
        description="REST API para el laboratorio de investigacion MythosForge — Recurrent-Depth Transformers",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        endpoints={
            "info": "/api/v1/info",
            "architecture": "/api/v1/architecture",
            "components": "/api/v1/components",
            "validation": "/api/v1/validation",
            "roadmap": "/api/v1/roadmap",
            "references": "/api/v1/references",
            "i18n": "/api/v1/i18n/{lang}",
            "inference": "/api/v1/inference",
            "metrics": "/api/v1/metrics",
            "health": "/api/v1/health",
        },
    )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — Returns service status and dependency availability."""
    return build_health_response(time.perf_counter() - _start_time)


@app.get("/api/v1/metrics", tags=["System"])
async def get_metrics():
    """Application metrics — counters, gauges, and latency histograms."""
    if not settings.metrics_enabled:
        return {"detail": "metrics_disabled", "metrics_enabled": False}
    return metrics.snapshot()
