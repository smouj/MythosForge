"""
mythosforge.attention — Switchable attention mechanisms (GQA and MLA).

The :class:`Attention` factory returns either a :class:`GroupedQueryAttention` or a
:class:`MultiLatentAttention` depending on ``config.attn_type``.  Both expose the
same forward signature so they can be swapped at configuration time without any
other code change.

References
----------
* GQA — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
  Checkpoints" (Shazeer, 2019)  https://arxiv.org/abs/2305.13245
* MLA — "A Long March from Mamba to DeepSeek-V2" (DeepSeek-AI, 2024)
  https://arxiv.org/abs/2405.04434
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MythosConfig
from .utils import RMSNorm, RotaryEmbedding, apply_rotary_emb, repeat_kv


# ═══════════════════════════════════════════════════════════
# Grouped-Query Attention (GQA)
# ═══════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with RoPE.

    Parameters
    ----------
    config : MythosConfig
        Must have ``attn_type="gqa"``.
    """

    def __init__(self, config: MythosConfig):
        super().__init__()
        assert config.n_heads % config.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.rope = RotaryEmbedding(
            self.head_dim, max_seq_len=config.max_seq_len, base=config.rope_base
        )
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor ``(batch, seq_len, dim)``
        mask : Optional[Tensor] ``(batch, 1, seq_len, seq_len)`` — causal mask.

        Returns
        -------
        Tensor ``(batch, seq_len, dim)``
        """
        B, S, _ = x.shape
        h = self.norm(x)

        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(S)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Expand KV for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out) + x  # residual


# ═══════════════════════════════════════════════════════════
# Multi-head Latent Attention (MLA) — DeepSeek-V2 style
# ═══════════════════════════════════════════════════════════

class MultiLatentAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2).

    Instead of projecting directly to full-rank K/V, MLA first compresses the
    hidden state into a low-rank KV latent ``c_kv`` of rank ``kv_lora_rank``.
    The full K and V are then reconstructed on-the-fly via learnable up-projections.

    Optionally, queries can also be compressed via a ``q_lora_rank`` latent.

    The head dimension is split into two parts:
    * **nope** (non-RoPE) — standard attention portion
    * **rope** — portion that receives rotary embeddings

    References
    ----------
    DeepSeek-V2: https://arxiv.org/abs/2405.04434
    """

    def __init__(self, config: MythosConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.q_total_dim = self.n_heads * self.qk_head_dim
        self.kv_total_dim = (
            self.n_heads * self.qk_nope_head_dim + self.n_heads * self.v_head_dim
        )
        self.scale = self.qk_head_dim ** -0.5

        # KV compression
        self.kv_down_proj = nn.Linear(config.dim, self.kv_lora_rank, bias=False)

        # KV up-projections → K_nope (shared across heads via broadcast) and V
        self.kv_up_k = nn.Linear(self.kv_lora_rank, self.n_heads * self.qk_nope_head_dim, bias=False)
        self.kv_up_v = nn.Linear(self.kv_lora_rank, self.n_heads * self.v_head_dim, bias=False)

        # Q projection (direct or compressed)
        if self.q_lora_rank > 0:
            self.q_down_proj = nn.Linear(config.dim, self.q_lora_rank, bias=False)
            self.q_up_proj = nn.Linear(self.q_lora_rank, self.q_total_dim, bias=False)
        else:
            self.q_down_proj = None
            self.q_up_proj = None
            self.q_proj = nn.Linear(config.dim, self.q_total_dim, bias=False)

        # RoPE for the RoPE portion
        self.rope = RotaryEmbedding(
            self.qk_rope_head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        # Output projection
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, config.dim, bias=False)
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor ``(batch, seq_len, dim)``
        mask : Optional[Tensor] ``(batch, 1, seq_len, seq_len)`` — causal mask.

        Returns
        -------
        Tensor ``(batch, seq_len, dim)``
        """
        B, S, D = x.shape
        h = self.norm(x)

        # ── KV branch ───────────────────────────────────
        c_kv = self.kv_down_proj(h)  # (B, S, kv_lora_rank)
        k_nope = self.kv_up_k(c_kv).view(B, S, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
        v = self.kv_up_v(c_kv).view(B, S, self.n_heads, self.v_head_dim).transpose(1, 2)

        # ── Q branch ────────────────────────────────────
        if self.q_down_proj is not None:
            c_q = self.q_down_proj(h)  # (B, S, q_lora_rank)
            q_full = self.q_up_proj(c_q).view(B, S, self.n_heads, self.qk_head_dim).transpose(1, 2)
        else:
            q_full = self.q_proj(h).view(B, S, self.n_heads, self.qk_head_dim).transpose(1, 2)

        # Split Q into nope + rope parts
        q_nope = q_full[..., :self.qk_nope_head_dim]
        q_rope = q_full[..., self.qk_nope_head_dim:]

        # RoPE portion of K — derived from the KV latent + a RoPE-specific up-projection
        # We generate a RoPE-only K using the rope portion of c_kv content.
        # This is a learned projection for the RoPE part.
        # Fallback: reuse k_nope's structure — produce a rope_k from the KV latent.
        if not hasattr(self, "kv_up_rope"):
            self.kv_up_rope = nn.Linear(
                self.kv_lora_rank, self.n_heads * self.qk_rope_head_dim, bias=False
            ).to(x.device, dtype=x.dtype)
        k_rope = self.kv_up_rope(c_kv).view(
            B, S, self.n_heads, self.qk_rope_head_dim
        ).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(S)
        q_rope = apply_rotary_emb(q_rope, cos, sin)
        k_rope = apply_rotary_emb(k_rope, cos, sin)

        # Concatenate nope + rope for full Q and K
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, H, S, qk_head_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)   # (B, H, S, qk_head_dim)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        # Attend to V (v_head_dim per head)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out) + x  # residual


# ═══════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════

def Attention(config: MythosConfig) -> nn.Module:  # noqa: N802 — PascalCase factory
    """Return the appropriate attention module based on *config.attn_type*.

    Parameters
    ----------
    config : MythosConfig

    Returns
    -------
    nn.Module — either :class:`GroupedQueryAttention` or :class:`MultiLatentAttention`.
    """
    if config.attn_type == "gqa":
        return GroupedQueryAttention(config)
    elif config.attn_type == "mla":
        return MultiLatentAttention(config)
    else:
        raise ValueError(f"Unknown attn_type: {config.attn_type!r}.  Use 'gqa' or 'mla'.")
