"""
openmythos_quickstart.py
Ejemplo mínimo y verificable para arrancar OpenMythos en CPU o GPU.

Uso esperado desde la raíz del repo clonado:
    python openmythos_quickstart.py
"""

import time
import torch
from open_mythos.main import OpenMythos, MythosConfig


def run(attn_type: str) -> None:
    base = dict(
        vocab_size=256,
        dim=64 if attn_type == "mla" else 48,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=64,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=16,
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

    model = OpenMythos(cfg)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    ids = torch.randint(0, cfg.vocab_size, (1, 8))

    t0 = time.time()
    with torch.no_grad():
        logits = model(ids, n_loops=2)
        generated = model.generate(ids, max_new_tokens=2, n_loops=2)
        A = model.recurrent.injection.get_A().detach()
    dt = time.time() - t0

    print(f"\n[{attn_type.upper()}]")
    print(f"Parámetros: {total:,}")
    print(f"Forma logits: {tuple(logits.shape)}")
    print(f"Forma generación: {tuple(generated.shape)}")
    print(f"Rango A discreta: min={A.min().item():.8f}, max={A.max().item():.8f}")
    print(f"Tiempo total: {dt:.3f}s")


if __name__ == "__main__":
    print("Iniciando verificación mínima de OpenMythos...")
    run("gqa")
    run("mla")
    print("\nListo.")
