"""
mythosforge — Pure PyTorch implementation of the OpenMythos recurrent-depth transformer.

This package provides a self-contained, research-grade implementation of the
MythosForge architecture with the following components:

* **Switchable Attention** — GQA or MLA, selected at configuration time.
* **Mixture of Experts** — routed + shared experts (DeepSeekMoE-style).
* **Stable LTI Recurrence** — Linear Time-Invariant state with A in (0,1) by construction.
* **Adaptive Computation Time** — per-position halting (Graves, 2016).
* **Per-iteration LoRA** — depth-modulated low-rank adaptation.

Quickstart
----------
>>> from mythosforge import OpenMythos, MythosConfig
>>> cfg = MythosConfig(vocab_size=256, dim=64, n_heads=4, n_kv_heads=4,
...                     max_seq_len=64, max_loop_iters=2, prelude_layers=1,
...                     coda_layers=1, n_experts=4, n_shared_experts=1,
...                     n_experts_per_tok=1, expert_dim=16, lora_rank=4,
...                     attn_type="gqa")
>>> model = OpenMythos(cfg)
>>> model.eval()
>>> ids = torch.randint(0, 256, (1, 8))
>>> with torch.no_grad():
...     logits = model(ids, n_loops=2)
>>> logits.shape
torch.Size([1, 8, 256])

Switching to MLA attention:
>>> cfg_mla = MythosConfig(vocab_size=256, dim=64, n_heads=4, n_kv_heads=4,
...                        max_seq_len=64, max_loop_iters=2, prelude_layers=1,
...                        coda_layers=1, n_experts=4, n_shared_experts=1,
...                        n_experts_per_tok=1, expert_dim=16, lora_rank=4,
...                        attn_type="mla", kv_lora_rank=8, q_lora_rank=16,
...                        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8)
>>> model_mla = OpenMythos(cfg_mla)

References
----------
* OpenMythos: https://github.com/kyegomez/OpenMythos
* Loop, Think, & Generalize: https://arxiv.org/abs/2604.07822
* Parcae: https://arxiv.org/abs/2604.12946
* DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
* GQA: https://arxiv.org/abs/2305.13245
* DeepSeekMoE: https://arxiv.org/abs/2401.06066
* ACT: https://arxiv.org/abs/1603.08983
"""

__version__ = "0.1.0"
__author__ = "Smouj013"

from .config import MythosConfig
from .model import OpenMythos

__all__ = [
    "OpenMythos",
    "MythosConfig",
]
