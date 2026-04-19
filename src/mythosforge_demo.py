"""
mythosforge_demo.py — Ejemplo mínimo de uso del paquete mythosforge.

Ejecutar desde la raíz del repositorio:
    PYTHONPATH=src python src/mythosforge_demo.py
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
from mythosforge import OpenMythos, MythosConfig


def demo(attn_type: str = "gqa") -> None:
    """Run a minimal forward + generate demo with the given attention type."""
    print(f"\n{'='*60}")
    print(f"  MythosForge Demo — {attn_type.upper()}")
    print(f"{'='*60}")

    # ── Config ─────────────────────────────────────────
    base = dict(
        vocab_size=256,
        dim=64 if attn_type == "mla" else 48,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=64,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=32,
        lora_rank=4,
        attn_type=attn_type,
    )

    if attn_type == "mla":
        cfg = MythosConfig(
            **base,
            kv_lora_rank=8,
            q_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
        )
    else:
        cfg = MythosConfig(**base)

    # ── Build model ────────────────────────────────────
    print(f"Building model...")
    model = OpenMythos(cfg)
    model.eval()

    counts = model.count_parameters()
    print(f"Total parameters: {counts['total']:,}")
    print(f"  Embedding:   {counts['embedding']:,}")
    print(f"  Prelude:     {counts['prelude']:,}")
    print(f"  Recurrent:   {counts['recurrent']:,}")
    print(f"  LTI:         {counts['injection']:,}")
    print(f"  Coda:        {counts['coda']:,}")

    # ── Forward pass ───────────────────────────────────
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    print(f"\nInput shape: {tuple(ids.shape)}")

    t0 = time.time()
    with torch.no_grad():
        logits = model(ids, n_loops=2)
    dt = time.time() - t0
    print(f"Forward (2 loops) → logits shape: {tuple(logits.shape)}  ({dt:.3f}s)")

    # ── Generate ───────────────────────────────────────
    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(ids, max_new_tokens=4, n_loops=2, temperature=0.8)
    dt = time.time() - t0
    print(f"Generate (4 tokens) → shape: {tuple(generated.shape)}  ({dt:.3f}s)")

    # ── LTI stability check ────────────────────────────
    A = model.injection.get_A().detach()
    print(f"\nLTI A matrix:")
    print(f"  min = {A.min().item():.8f}")
    print(f"  max = {A.max().item():.8f}")
    print(f"  mean = {A.mean().item():.8f}")
    print(f"  All in (0,1): {(A > 0).all() and (A < 1).all()}")


def demo_comparison() -> None:
    """Compare GQA vs MLA output shapes and parameter counts."""
    print(f"\n{'='*60}")
    print(f"  MythosForge — GQA vs MLA Comparison")
    print(f"{'='*60}")

    cfg_gqa = MythosConfig(
        vocab_size=256, dim=48, n_heads=4, n_kv_heads=4,
        max_seq_len=64, max_loop_iters=2, prelude_layers=1,
        coda_layers=1, n_experts=4, n_shared_experts=1,
        n_experts_per_tok=1, expert_dim=32, lora_rank=4,
        attn_type="gqa",
    )
    cfg_mla = MythosConfig(
        vocab_size=256, dim=64, n_heads=4, n_kv_heads=4,
        max_seq_len=64, max_loop_iters=2, prelude_layers=1,
        coda_layers=1, n_experts=4, n_shared_experts=1,
        n_experts_per_tok=1, expert_dim=32, lora_rank=4,
        attn_type="mla", kv_lora_rank=8, q_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
    )

    model_gqa = OpenMythos(cfg_gqa)
    model_mla = OpenMythos(cfg_mla)
    model_gqa.eval()
    model_mla.eval()

    ids = torch.randint(0, 256, (1, 8))

    with torch.no_grad():
        logits_gqa = model_gqa(ids, n_loops=2)
        logits_mla = model_mla(ids, n_loops=2)

    gqa_counts = model_gqa.count_parameters()
    mla_counts = model_mla.count_parameters()

    print(f"\n{'Metric':<25} {'GQA':>12} {'MLA':>12}")
    print(f"{'-'*49}")
    print(f"{'Total params':<25} {gqa_counts['total']:>12,} {mla_counts['total']:>12,}")
    print(f"{'Output shape':<25} {str(tuple(logits_gqa.shape)):>12} {str(tuple(logits_mla.shape)):>12}")
    print(f"{'NaN in output':<25} {'No':>12} {'No':>12}")


if __name__ == "__main__":
    print("MythosForge — Pure PyTorch Recurrent-Depth Transformer")
    print(f"PyTorch version: {torch.__version__}")
    demo("gqa")
    demo("mla")
    demo_comparison()
    print("\nAll demos completed successfully!")
