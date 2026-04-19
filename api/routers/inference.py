"""
Inference router — Real OpenMythos model inference when PyTorch is available.
Graceful degradation with clear setup instructions when dependencies are missing.
"""

import time
import logging
from fastapi import APIRouter, HTTPException

from api.models import (
    InferenceRequest, InferenceResponse, InferenceConfig, InferenceTimings
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/inference", tags=["Inference"])

# ─── Dependency detection (lazy, safe) ────────────────

_pytorch_available = False
_openmythos_available = False
_openmythos_error = None

def _check_dependencies():
    """Lazily check if PyTorch and OpenMythos are importable."""
    global _pytorch_available, _openmythos_available, _openmythos_error
    if _pytorch_available or _openmythos_error:
        return  # Already checked

    try:
        import torch
        _pytorch_available = True
        logger.info(f"PyTorch {torch.__version__} detected on device: {torch.device('cpu')}")
    except ImportError:
        _openmythos_error = (
            "PyTorch is not installed. Install it with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
        logger.warning(_openmythos_error)
        return

    try:
        # Try to import OpenMythos model components
        from open_mythos import MythosForCausalLM, MythosConfig
        _openmythos_available = True
        logger.info("OpenMythos model components detected successfully")
    except ImportError:
        try:
            # Alternative import paths
            from mythos import MythosForCausalLM, MythosConfig
            _openmythos_available = True
            logger.info("OpenMythos model components detected (mythos import)")
        except ImportError:
            _openmythos_error = (
                "OpenMythos is not installed. Clone and install it:\n"
                "  git clone https://github.com/kyegomez/OpenMythos.git\n"
                "  cd OpenMythos && pip install -r requirements.txt\n"
                "  pip install -e ."
            )
            logger.warning(_openmythos_error)


# ─── Model cache (singleton per process) ─────────────

_model_cache = {}

def _get_model(attn_type: str = "gqa"):
    """Load or retrieve cached model instance."""
    if attn_type in _model_cache:
        return _model_cache[attn_type]

    try:
        import torch
        if attn_type == "mla":
            from open_mythos import MythosForCausalLM, MythosConfig
        else:
            from open_mythos import MythosForCausalLM, MythosConfig

        config = MythosConfig(
            d_model=256,
            n_heads=4,
            n_kv_heads=2,
            d_ff=512,
            n_layers=2,
            max_seq_len=512,
            max_loop_iters=32,
            vocab_size=3200,
            attn_type=attn_type,
            use_moe=True,
            num_experts=4,
            num_shared_experts=1,
            top_k=2,
            use_lti=True,
            use_act=True,
            use_lora=True,
            lora_rank=4,
        )
        model = MythosForCausalLM(config)
        model.eval()
        _model_cache[attn_type] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {str(e)}")


# ─── Endpoints ────────────────────────────────────────

@router.post(
    "",
    response_model=InferenceResponse,
    summary="Run model inference",
    description=(
        "Generate text using the OpenMythos Recurrent-Depth Transformer. "
        "Requires PyTorch and OpenMythos to be installed. "
        "If not available, returns setup instructions."
    ),
    responses={
        503: {"description": "PyTorch or OpenMythos not installed"},
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
                "instructions": _openmythos_error,
                "config": config.model_dump(),
            },
        )

    if not _openmythos_available:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "OpenMythos not installed",
                "instructions": _openmythos_error,
                "config": config.model_dump(),
            },
        )

    total_start = time.perf_counter()

    try:
        import torch

        # Load model
        load_start = time.perf_counter()
        model = _get_model(request.attn_type.value)
        timings.model_load_seconds = round(time.perf_counter() - load_start, 4)

        # Tokenize (simple character-level since we don't have a real tokenizer)
        # OpenMythos doesn't include a tokenizer, so we use simple token mapping
        vocab = model.config.vocab_size
        input_ids = torch.tensor([list(range(min(len(request.prompt), 64))) % vocab], dtype=torch.long)

        # Prefill
        prefill_start = time.perf_counter()
        with torch.no_grad():
            logits = model(input_ids, n_loops=request.n_loops)
        timings.prefill_seconds = round(time.perf_counter() - prefill_start, 4)

        # Generation loop
        gen_start = time.perf_counter()
        generated_ids = []
        current_ids = input_ids

        with torch.no_grad():
            for _ in range(request.max_tokens):
                logits = model(current_ids, n_loops=request.n_loops)
                next_logits = logits[:, -1, :] / max(request.temperature, 1e-8)

                if request.temperature < 0.01:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                generated_ids.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token], dim=1)

        timings.generation_seconds = round(time.perf_counter() - gen_start, 4)
        timings.total_seconds = round(time.perf_counter() - total_start, 4)

        return InferenceResponse(
            success=True,
            prompt=request.prompt,
            generated_text=f"[{len(generated_ids)} tokens generated — OpenMythos demo mode (no tokenizer)]",
            generated_tokens=[f"tok_{tid}" for tid in generated_ids],
            num_generated=len(generated_ids),
            config=config,
            timings=timings,
            note=(
                "OpenMythos no incluye tokenizer entrenado. Los tokens generados son índices de vocabulario. "
                "Para salida textual legible, se requiere Phase 3 (Tokenizer y datos) del roadmap."
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
    description="Check if PyTorch and OpenMythos are available for inference.",
)
async def inference_status():
    _check_dependencies()
    return {
        "pytorch_available": _pytorch_available,
        "openmythos_available": _openmythos_available,
        "cached_models": list(_model_cache.keys()),
        "setup_instructions": _openmythos_error,
    }
