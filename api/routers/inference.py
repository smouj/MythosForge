"""
Inference router — Real inference using the mythosforge package.

Uses the built-in mythosforge PyTorch implementation (no external dependencies
beyond PyTorch).  GQA and MLA are available out of the box.

The model runs on CPU with a small config for demo purposes.  Token IDs are
returned since mythosforge does not include a trained tokenizer (Phase 3).
"""

import os
import sys
import time
import logging
from fastapi import APIRouter, HTTPException

from api.models import (
    InferenceRequest, InferenceResponse, InferenceConfig, InferenceTimings
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/inference", tags=["Inference"])

# ─── Ensure mythosforge is importable ────────────────

# Add src/ to path so that `from mythosforge import ...` works
_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ─── Dependency detection (lazy) ─────────────────────

_pytorch_available = False
_mythosforge_available = False
_import_error: str | None = None


def _check_dependencies():
    """Lazily check if PyTorch and mythosforge are importable."""
    global _pytorch_available, _mythosforge_available, _import_error
    if _pytorch_available or _import_error:
        return

    try:
        import torch
        _pytorch_available = True
        logger.info(f"PyTorch {torch.__version__} detected")
    except ImportError:
        _import_error = (
            "PyTorch is not installed. Install it with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
        logger.warning(_import_error)
        return

    try:
        from mythosforge import OpenMythos, MythosConfig
        _mythosforge_available = True
        logger.info("mythosforge package loaded successfully")
    except ImportError as e:
        _import_error = (
            f"mythosforge package not found: {e}\n"
            "Make sure src/mythosforge/ exists in the repository root."
        )
        logger.warning(_import_error)


# ─── Model cache (singleton per process) ─────────────

_model_cache: dict[str, object] = {}


def _get_model(attn_type: str = "gqa"):
    """Load or retrieve a cached mythosforge model."""
    if attn_type in _model_cache:
        return _model_cache[attn_type]

    from mythosforge import OpenMythos, MythosConfig

    if attn_type == "mla":
        cfg = MythosConfig(
            vocab_size=256,
            dim=64,
            n_heads=4,
            n_kv_heads=4,
            max_seq_len=128,
            max_loop_iters=8,
            prelude_layers=1,
            coda_layers=1,
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=1,
            expert_dim=64,
            lora_rank=4,
            attn_type="mla",
            kv_lora_rank=8,
            q_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
        )
    else:
        cfg = MythosConfig(
            vocab_size=256,
            dim=48,
            n_heads=4,
            n_kv_heads=4,
            max_seq_len=128,
            max_loop_iters=8,
            prelude_layers=1,
            coda_layers=1,
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=1,
            expert_dim=64,
            lora_rank=4,
            attn_type="gqa",
        )

    model = OpenMythos(cfg)
    model.eval()

    counts = model.count_parameters()
    logger.info(
        f"Loaded {attn_type.upper()} model: {counts['total']:,} parameters"
    )

    _model_cache[attn_type] = model
    return model


# ─── Endpoints ────────────────────────────────────────

@router.post(
    "",
    response_model=InferenceResponse,
    summary="Run model inference",
    description=(
        "Generate token IDs using the built-in mythosforge recurrent-depth "
        "transformer. Supports both GQA and MLA attention. "
        "Only requires PyTorch — no external ML packages needed."
    ),
    responses={
        503: {"description": "PyTorch not installed"},
    },
)
async def run_inference(request: InferenceRequest):
    _check_dependencies()

    config = InferenceConfig(
        attn_type=request.attn_type.value,
        n_loops=request.n_loops,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        device="cpu",
    )
    timings = InferenceTimings()

    if not _pytorch_available:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "PyTorch not installed",
                "instructions": _import_error,
                "config": config.model_dump(),
            },
        )

    if not _mythosforge_available:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "mythosforge not available",
                "instructions": _import_error,
                "config": config.model_dump(),
            },
        )

    total_start = time.perf_counter()

    try:
        import torch
        from mythosforge import OpenMythos

        # Load model
        load_start = time.perf_counter()
        model = _get_model(request.attn_type.value)
        assert isinstance(model, OpenMythos)
        timings.model_load_seconds = round(time.perf_counter() - load_start, 4)

        # Tokenize prompt: simple ordinal mapping into vocab range
        vocab = model.config.vocab_size
        prompt_ids = [ord(c) % vocab for c in request.prompt]
        prompt_ids = prompt_ids[: model.config.max_seq_len - request.max_tokens - 1]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        if input_ids.shape[1] == 0:
            input_ids = torch.tensor([[0]], dtype=torch.long)

        # Prefill
        prefill_start = time.perf_counter()
        with torch.no_grad():
            logits = model(input_ids, n_loops=request.n_loops)
        timings.prefill_seconds = round(time.perf_counter() - prefill_start, 4)

        # Generate
        gen_start = time.perf_counter()
        generated = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            n_loops=request.n_loops,
            temperature=request.temperature,
            top_k=50,
        )
        timings.generation_seconds = round(time.perf_counter() - gen_start, 4)

        # Extract generated token IDs
        gen_ids = generated[0, input_ids.shape[1]:].tolist()
        timings.total_seconds = round(time.perf_counter() - total_start, 4)

        # Build parameter info
        counts = model.count_parameters()
        A = model.injection.get_A().detach()
        lti_min = round(A.min().item(), 6)
        lti_max = round(A.max().item(), 6)

        return InferenceResponse(
            success=True,
            prompt=request.prompt,
            generated_text=(
                f"[{len(gen_ids)} tokens generados con {request.attn_type.value.upper()} "
                f"— modo demo mythosforge v0.1.0]"
            ),
            generated_tokens=[f"tok_{tid}" for tid in gen_ids],
            num_generated=len(gen_ids),
            config=config,
            timings=timings,
            note=(
                f"Parametros: {counts['total']:,} | "
                f"LTI A: [{lti_min}, {lti_max}] | "
                f"Bucle: {request.n_loops} iters | "
                f"mythosforge no incluye tokenizer (Phase 3 del roadmap). "
                f"Los tokens son IDs de vocabulario mapeados desde ordinales."
            ),
        )

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        timings.total_seconds = round(time.perf_counter() - total_start, 4)
        return InferenceResponse(
            success=False,
            prompt=request.prompt,
            generated_text="",
            generated_tokens=[],
            num_generated=0,
            config=config,
            timings=timings,
            error=str(e),
        )


@router.get(
    "/status",
    summary="Check inference dependencies",
    description="Check if PyTorch and mythosforge are available for inference.",
)
async def inference_status():
    _check_dependencies()
    return {
        "pytorch_available": _pytorch_available,
        "mythosforge_available": _mythosforge_available,
        "cached_models": list(_model_cache.keys()),
        "setup_instructions": _import_error,
        "backend": "mythosforge v0.1.0 (built-in PyTorch)",
    }
