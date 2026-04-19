"""
MythosForge API — Main application.

FastAPI server serving real project data, OpenAPI docs, and optional OpenMythos inference.
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import __version__, __api_version__
from api.models import (
    APIRootResponse, ProjectInfo, ArchitectureResponse,
    ComponentsListResponse, ComponentDetail, ValidationResponse,
    RoadmapResponse, ReferencesResponse, Reference,
    I18nResponse, HealthResponse,
)
from api.data import (
    PROJECT_INFO, ARCHITECTURE, COMPONENTS,
    INFERENCE_CHECKS, REPO_STATUS, ROADMAP, REFERENCES, I18N_TRANSLATIONS,
)
from api.routers.inference import router as inference_router

# ─── Logging ──────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mythosforge.api")

# ─── Lifespan ─────────────────────────────────────────

_start_time = time.perf_counter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"MythosForge API v{__version__} starting...")
    yield
    logger.info("MythosForge API shutting down.")


# ─── App ──────────────────────────────────────────────

app = FastAPI(
    title="MythosForge API",
    description=(
        "## 🔥 MythosForge — Recurrent-Depth Transformer Research Lab\n\n"
        "REST API real para el laboratorio de investigación MythosForge basado en "
        "[OpenMythos](https://github.com/kyegomez/OpenMythos).\n\n"
        "### Arquitectura\n"
        "Prelude → **Recurrent Block** (×N loops) → Coda\n\n"
        "### Componentes\n"
        "- **MLA / GQA**: Atención conmutable (DeepSeek-V2 / GQA)\n"
        "- **MoE**: Mixture of Experts con routed + shared experts\n"
        "- **LTI**: Inyección estable Linear Time-Invariant, A ∈ (0,1)\n"
        "- **ACT**: Adaptive Computation Time — halting por posición\n"
        "- **LoRA**: Adaptador per-iteración con señal de profundidad\n\n"
        "### Endpoints\n"
        "- Datos del proyecto, arquitectura, componentes, validación, roadmap, referencias\n"
        "- Traducciones i18n (ES/EN)\n"
        "- Inferencia real con OpenMythos (requiere PyTorch)\n\n"
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

# ─── CORS ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Include routers ──────────────────────────────────

app.include_router(inference_router)


# ─── Middleware: Request logging ──────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration_ms:.1f}ms)")
    return response


# ─── Exception handler ───────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


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
            "health": "/api/v1/health",
        },
    )


# ─── Health ───────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — Returns service status and dependency availability."""
    try:
        import torch
        pt = True
    except ImportError:
        pt = False

    try:
        import open_mythos
        om = True
    except ImportError:
        try:
            import mythos
            om = True
        except ImportError:
            om = False

    return HealthResponse(
        status="healthy",
        version=__version__,
        api_version=__api_version__,
        pytorch_available=pt,
        openmythos_available=om,
        uptime_seconds=round(time.perf_counter() - _start_time, 2),
    )


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
    raise _not_found(f"Component '{slug}' not found. Available: {[c.slug for c in COMPONENTS]}")


# ─── Validation ───────────────────────────────────────

@app.get("/api/v1/validation", response_model=ValidationResponse, tags=["Validation"])
async def get_validation():
    """Get all validation results — inference checks, repo status, and key findings."""
    return ValidationResponse(
        environment="CPU · PyTorch 2.10.0+cpu",
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
    raise _not_found(f"Reference '{ref_id}' not found. Available: {[r.id for r in REFERENCES]}")


# ─── i18n ─────────────────────────────────────────────

@app.get("/api/v1/i18n/{lang}", response_model=I18nResponse, tags=["i18n"])
async def get_translations(lang: str):
    """Get all translations for a language (es or en). Returns ~30 translation keys."""
    translations = I18N_TRANSLATIONS.get(lang.lower())
    if not translations:
        raise _not_found(f"Language '{lang}' not found. Available: {list(I18N_TRANSLATIONS.keys())}")
    return I18nResponse(
        language=lang.lower(),
        total_keys=len(translations),
        translations=translations,
    )


# ─── Helpers ──────────────────────────────────────────

def _not_found(detail: str):
    from fastapi import HTTPException
    return HTTPException(status_code=404, detail=detail)
