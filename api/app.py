"""
MythosForge API — Main application (v0.4.0).

Refactored with:
- Environment-based settings (Pydantic Settings)
- Restricted CORS (allowlist, not wildcard)
- Safe error handlers (no internal detail leaks)
- Request ID tracking
- Structured logging
- Optional API key authentication
- Basic metrics collection
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
    not_found_handler, validation_error_handler,
    app_error_handler, unhandled_exception_handler,
)
from api.deps import build_health_response
from api.observability.metrics import metrics
from api.models import (
    APIRootResponse, HealthResponse, ProjectInfo, ArchitectureResponse,
    ComponentsListResponse, ComponentDetail, ValidationResponse,
    RoadmapResponse, ReferencesResponse, Reference,
    I18nResponse,
)
from api.data import (
    PROJECT_INFO, ARCHITECTURE, COMPONENTS,
    INFERENCE_CHECKS, REPO_STATUS, ROADMAP, REFERENCES, I18N_TRANSLATIONS,
)
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
        "## 🔥 MythosForge — Recurrent-Depth Transformer Research Lab\n\n"
        "REST API para el laboratorio de investigación MythosForge.\n\n"
        "### Arquitectura\n"
        "Prelude → **Recurrent Block** (×N loops) → Coda\n\n"
        "### Componentes\n"
        "- **MLA / GQA**: Atención conmutable (DeepSeek-V2 / GQA)\n"
        "- **MoE**: Mixture of Experts con routed + shared experts\n"
        "- **LTI**: Inyección estable Linear Time-Invariant, A ∈ (0,1)\n"
        "- **ACT**: Adaptive Computation Time — halting por posición\n"
        "- **LoRA**: Adaptador per-iteración con señal de profundidad\n\n"
        "### Seguridad\n"
        f"Autenticación: {'habilitada' if settings.auth_enabled else 'deshabilitada (dev)'}\n\n"
        "### Aviso\n"
        "OpenMythos es una reconstrucción teórica independiente, no afiliada a Anthropic.\n"
        "MythosForge no afirma equivalencia con ningún sistema propietario.\n\n"
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
        "%s %s → %d (%.1fms) [%s]",
        request.method, path, response.status_code, duration_ms, rid,
    )

    # Add request ID to response
    response.headers["X-Request-ID"] = rid

    return response


# ═══════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════

@app.get("/", response_model=APIRootResponse, tags=["Root"])
async def api_root():
    """API root — Returns links to documentation and available endpoints."""
    return APIRootResponse(
        name="MythosForge API",
        version=__version__,
        api_version=__api_version__,
        description="REST API para el laboratorio de investigación MythosForge — Recurrent-Depth Transformers",
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


# ─── Health ───────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — Returns service status and dependency availability."""
    return build_health_response(time.perf_counter() - _start_time)


# ─── Metrics ──────────────────────────────────────────

@app.get("/api/v1/metrics", tags=["System"])
async def get_metrics():
    """Application metrics — counters, gauges, and latency histograms."""
    if not settings.metrics_enabled:
        return {"detail": "metrics_disabled", "metrics_enabled": False}
    return metrics.snapshot()


# ─── Project Info ─────────────────────────────────────

@app.get("/api/v1/info", response_model=ProjectInfo, tags=["Project"])
async def get_info():
    """Get complete project information — name, description, credits, features, and disclaimers."""
    return PROJECT_INFO


# ─── Architecture ────────────────────────────────────

@app.get("/api/v1/architecture", response_model=ArchitectureResponse, tags=["Architecture"])
async def get_architecture():
    """Get full architecture details — flow, recurrent components, and the recurrent rule."""
    return ArchitectureResponse(**ARCHITECTURE)


# ─── Components ───────────────────────────────────────

@app.get("/api/v1/components", response_model=ComponentsListResponse, tags=["Components"])
async def get_components():
    """Get all architectural components — MLA/GQA, MoE, LTI, ACT, LoRA, Prelude+Coda."""
    return ComponentsListResponse(
        components=COMPONENTS,
        total=len(COMPONENTS),
    )


@app.get(
    "/api/v1/components/{slug}",
    response_model=ComponentDetail,
    tags=["Components"],
    responses={404: {"description": "Component not found"}},
)
async def get_component(slug: str):
    """Get details for a specific architectural component by slug."""
    for comp in COMPONENTS:
        if comp.slug == slug:
            return comp
    raise NotFoundError(
        f"Component '{slug}' not found. Available: {[c.slug for c in COMPONENTS]}"
    )


# ─── Validation ───────────────────────────────────────

@app.get("/api/v1/validation", response_model=ValidationResponse, tags=["Validation"])
async def get_validation():
    """Get all validation results — inference checks, repo status, and key findings."""
    return ValidationResponse(
        environment="CPU · PyTorch 2.11.0+cpu",
        inference_checks=INFERENCE_CHECKS,
        repo_status=REPO_STATUS,
        key_finding=(
            "test_spectral_radius_stable_after_large_grad_step falla por redondeo float32. "
            "Parche propuesto resuelve sin alterar la arquitectura."
        ),
        key_finding_en=(
            "test_spectral_radius_stable_after_large_grad_step fails due to float32 rounding. "
            "Proposed patch resolves without altering the architecture."
        ),
        patch_available=True,
        patch_url="https://github.com/smouj/MythosForge/blob/main/src/openmythos_lti_patch.diff",
    )


# ─── Roadmap ──────────────────────────────────────────

@app.get("/api/v1/roadmap", response_model=RoadmapResponse, tags=["Roadmap"])
async def get_roadmap():
    """Get the full project roadmap — 7 phases from freeze to publication."""
    completed = sum(1 for p in ROADMAP if p.status.value == "completed")
    return RoadmapResponse(
        phases=ROADMAP,
        total=len(ROADMAP),
        completed=completed,
    )


# ─── References ───────────────────────────────────────

@app.get("/api/v1/references", response_model=ReferencesResponse, tags=["References"])
async def get_references():
    """Get all academic references — papers, code repos, and documentation."""
    return ReferencesResponse(
        references=REFERENCES,
        total=len(REFERENCES),
    )


@app.get(
    "/api/v1/references/{ref_id}",
    response_model=Reference,
    tags=["References"],
    responses={404: {"description": "Reference not found"}},
)
async def get_reference(ref_id: str):
    """Get a specific academic reference by ID (e.g., R1, R2, R4...)."""
    for ref in REFERENCES:
        if ref.id.upper() == ref_id.upper():
            return ref
    raise NotFoundError(
        f"Reference '{ref_id}' not found. Available: {[r.id for r in REFERENCES]}"
    )


# ─── i18n ─────────────────────────────────────────────

@app.get("/api/v1/i18n/{lang}", response_model=I18nResponse, tags=["i18n"])
async def get_translations(lang: str):
    """Get all translations for a language (es or en). Returns ~30 translation keys."""
    translations = I18N_TRANSLATIONS.get(lang.lower())
    if not translations:
        raise NotFoundError(
            f"Language '{lang}' not found. Available: {list(I18N_TRANSLATIONS.keys())}"
        )
    return I18nResponse(
        language=lang.lower(),
        total_keys=len(translations),
        translations=translations,
    )
