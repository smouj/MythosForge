"""
MythosConfig — Central configuration for the OpenMythos recurrent-depth transformer.

All architectural hyperparameters are defined here.  Every component reads from
this single dataclass so that switching between GQA / MLA, adjusting expert
counts, loop budget, or LTI parameters is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MythosConfig:
    """Immutable configuration for MythosForge / OpenMythos.

    Attributes
    ----------
    vocab_size : int
        Tokenizer vocabulary size (default 32000).
    dim : int
        Hidden dimension used throughout the model.
    n_heads : int
        Number of query heads for attention.
    n_kv_heads : int
        Number of key/value heads.  When equal to *n_heads* this is MHA;
        smaller values give GQA.
    max_seq_len : int
        Maximum sequence length the model can process.
    max_loop_iters : int
        Upper bound on recurrent loop iterations during a single forward pass.
    prelude_layers : int
        Dense transformer layers executed once before the recurrent block.
    coda_layers : int
        Dense transformer layers executed once after the recurrent block.
    attn_type : {"gqa", "mla"}
        Which attention mechanism to instantiate inside the recurrent block.
        - ``"gqa"`` — Grouped-Query Attention (Shazeer, 2019).
        - ``"mla"`` — Multi-head Latent Attention (DeepSeek-V2).
    n_experts : int
        Number of routed experts in the MoE FFN.
    n_shared_experts : int
        Number of always-active shared experts (summed with routed output).
    n_experts_per_tok : int
        Top-K experts selected per token by the router.
    expert_dim : int
        Internal FFN dimension of each expert.
    lora_rank : int
        Rank of the per-iteration LoRA adapter injected inside the recurrent block.
    act_eps : float
        Epsilon for the ACT halting sigmoid to avoid division by zero.
    rope_base : float
        Base frequency for Rotary Position Embeddings.
    rms_norm_eps : float
        Epsilon for RMSNorm layers.

    MLA-specific (ignored when attn_type="gqa")
    --------------------------------------------
    kv_lora_rank : int
        Rank of the KV compression latent in MLA.
    q_lora_rank : int
        Rank of the query compression latent in MLA (0 = no compression).
    qk_rope_head_dim : int
        Dimension of the RoPE portion of Q/K in MLA.
    qk_nope_head_dim : int
        Dimension of the non-RoPE portion of Q/K in MLA.
    v_head_dim : int
        Dimension of V in MLA.
    """

    # ── Model ──────────────────────────────────────────
    vocab_size: int = 32000
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: int = 8
    max_seq_len: int = 2048
    max_loop_iters: int = 8

    # ── Prelude / Coda ─────────────────────────────────
    prelude_layers: int = 2
    coda_layers: int = 2

    # ── Attention ──────────────────────────────────────
    attn_type: Literal["gqa", "mla"] = "gqa"

    # ── MoE ────────────────────────────────────────────
    n_experts: int = 8
    n_shared_experts: int = 2
    n_experts_per_tok: int = 2
    expert_dim: int = 1024

    # ── LoRA ───────────────────────────────────────────
    lora_rank: int = 8

    # ── ACT ────────────────────────────────────────────
    act_eps: float = 1e-3

    # ── RoPE / Norm ────────────────────────────────────
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5

    # ── MLA-specific ───────────────────────────────────
    kv_lora_rank: int = 64
    q_lora_rank: int = 128
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
