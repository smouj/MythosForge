"""
MythosForge API — Public catalog router.

All read-only catalog endpoints: project info, architecture, components,
validation, roadmap, references, and i18n translations.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.models import (
    ProjectInfo, ArchitectureResponse,
    ComponentsListResponse, ComponentDetail, ValidationResponse,
    RoadmapResponse, ReferencesResponse, Reference, I18nResponse,
)
from api.data import (
    PROJECT_INFO, ARCHITECTURE, COMPONENTS,
    INFERENCE_CHECKS, REPO_STATUS, ROADMAP, REFERENCES, I18N_TRANSLATIONS,
)
from api.errors import NotFoundError

router = APIRouter(prefix="/api/v1", tags=["Catalog"])


# ─── Project Info ─────────────────────────────────────

@router.get("/info", response_model=ProjectInfo, tags=["Project"])
async def get_info():
    """Get complete project information — name, description, credits, features, and disclaimers."""
    return PROJECT_INFO


# ─── Architecture ────────────────────────────────────

@router.get("/architecture", response_model=ArchitectureResponse, tags=["Architecture"])
async def get_architecture():
    """Get full architecture details — flow, recurrent components, and the recurrent rule."""
    return ArchitectureResponse(**ARCHITECTURE)


# ─── Components ───────────────────────────────────────

@router.get("/components", response_model=ComponentsListResponse, tags=["Components"])
async def get_components():
    """Get all architectural components — MLA/GQA, MoE, LTI, ACT, LoRA, Prelude+Coda."""
    return ComponentsListResponse(
        components=COMPONENTS,
        total=len(COMPONENTS),
    )


@router.get(
    "/components/{slug}",
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

@router.get("/validation", response_model=ValidationResponse, tags=["Validation"])
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

@router.get("/roadmap", response_model=RoadmapResponse, tags=["Roadmap"])
async def get_roadmap():
    """Get the full project roadmap — 7 phases from freeze to publication."""
    completed = sum(1 for p in ROADMAP if p.status.value == "completed")
    return RoadmapResponse(
        phases=ROADMAP,
        total=len(ROADMAP),
        completed=completed,
    )


# ─── References ───────────────────────────────────────

@router.get("/references", response_model=ReferencesResponse, tags=["References"])
async def get_references():
    """Get all academic references — papers, code repos, and documentation."""
    return ReferencesResponse(
        references=REFERENCES,
        total=len(REFERENCES),
    )


@router.get(
    "/references/{ref_id}",
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

@router.get("/i18n/{lang}", response_model=I18nResponse, tags=["i18n"])
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
