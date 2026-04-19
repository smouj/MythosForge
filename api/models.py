"""
Pydantic v2 schemas for MythosForge API.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


# ─── Enums ────────────────────────────────────────────

class ComponentSlug(str, Enum):
    MLA = "mla-gqa"
    MOE = "moe"
    LTI = "lti"
    ACT = "act"
    LORA = "lora"
    PRELUDE_CODA = "prelude-coda"


class AttentionType(str, Enum):
    GQA = "gqa"
    MLA = "mla"


class PhaseStatus(str, Enum):
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


# ─── Shared Models ───────────────────────────────────

class Link(BaseModel):
    url: str
    label: str


class HealthResponse(BaseModel):
    status: str = Field(description="Service health status")
    version: str
    api_version: str
    pytorch_available: bool
    openmythos_available: bool
    uptime_seconds: float


# ─── Info ─────────────────────────────────────────────

class ProjectInfo(BaseModel):
    name: str
    full_name: str
    version: str
    api_version: str
    description: str
    description_en: str
    license: str
    repository: str
    github_pages: str
    creator: str
    created_with: str
    date: str
    based_on: str
    disclaimer: str
    features: list[str]
    not_claims: list[str]


# ─── Architecture ────────────────────────────────────

class ArchBlock(BaseModel):
    name: str
    description: str
    execution: str


class ArchComponent(BaseModel):
    name: str
    short_name: str
    description: str
    description_en: str


class ArchitectureResponse(BaseModel):
    flow: list[ArchBlock]
    recurrent_components: list[ArchComponent]
    recurrent_rule: str
    recurrent_rule_desc: str
    recurrent_rule_desc_en: str
    max_loop_iters: int


# ─── Components ───────────────────────────────────────

class ComponentDetail(BaseModel):
    slug: str
    name: str
    name_en: str
    description: str
    description_en: str
    color: str
    references: list[Link]
    key_insight: Optional[str] = None


class ComponentsListResponse(BaseModel):
    components: list[ComponentDetail]
    total: int


# ─── Validation ───────────────────────────────────────

class ValidationCheck(BaseModel):
    name: str
    name_en: str
    status: CheckStatus
    note: Optional[str] = None
    note_en: Optional[str] = None


class RepoStatusItem(BaseModel):
    feature: str
    feature_en: str
    available: bool


class ValidationResponse(BaseModel):
    environment: str
    inference_checks: list[ValidationCheck]
    repo_status: list[RepoStatusItem]
    key_finding: str
    key_finding_en: str
    patch_available: bool
    patch_url: str


# ─── Roadmap ──────────────────────────────────────────

class RoadmapPhase(BaseModel):
    phase: int
    name: str
    name_en: str
    description: str
    description_en: str
    status: PhaseStatus


class RoadmapResponse(BaseModel):
    phases: list[RoadmapPhase]
    total: int
    completed: int


# ─── References ───────────────────────────────────────

class Reference(BaseModel):
    id: str
    title: str
    url: str
    type: str


class ReferencesResponse(BaseModel):
    references: list[Reference]
    total: int


# ─── i18n ─────────────────────────────────────────────

class I18nResponse(BaseModel):
    language: str
    total_keys: int
    translations: dict[str, str]


# ─── Inference ────────────────────────────────────────

class InferenceRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Input text prompt for generation",
        examples=["The future of recurrent transformers is"]
    )
    max_tokens: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Maximum tokens to generate"
    )
    attn_type: AttentionType = Field(
        default=AttentionType.GQA,
        description="Attention type: gqa (faster) or mla (more memory efficient)"
    )
    n_loops: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of recurrent loop iterations"
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )


class InferenceConfig(BaseModel):
    attn_type: str
    n_loops: int
    max_tokens: int
    temperature: float
    device: str


class InferenceTimings(BaseModel):
    model_load_seconds: Optional[float] = None
    prefill_seconds: Optional[float] = None
    generation_seconds: Optional[float] = None
    total_seconds: Optional[float] = None


class InferenceResponse(BaseModel):
    success: bool
    prompt: str
    generated_text: str
    generated_tokens: list[str]
    num_generated: int
    config: InferenceConfig
    timings: InferenceTimings
    note: Optional[str] = None
    error: Optional[str] = None


# ─── API Root ─────────────────────────────────────────

class APIRootResponse(BaseModel):
    name: str
    version: str
    api_version: str
    description: str
    docs_url: str
    redoc_url: str
    openapi_url: str
    endpoints: dict[str, str]
