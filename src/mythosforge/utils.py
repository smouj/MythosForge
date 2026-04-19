"""
mythosforge.utils — Shared primitives: RMSNorm, SwiGLU, RotaryEmbedding, helpers.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────
# RMS Normalization
# ────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (Zhang & Sennrich, 2019).

    More efficient than LayerNorm — no mean subtraction, no bias — widely
    used in LLaMA, Mistral, DeepSeek, and related architectures.
    """

    __constants__ = ["eps", "dim"]

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# ────────────────────────────────────────────────────────
# SwiGLU Feed-Forward
# ────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU FFN (Shazeer, 2020).

    .. math::
        \\text{SwiGLU}(x) = (x W_1 \\odot \\text{SiLU}(x V)) W_2

    Used in LLaMA, Mistral, DeepSeek and many modern transformers.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.v(x))


# ────────────────────────────────────────────────────────
# Rotary Position Embeddings (RoPE)
# ────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Generates rotation matrices of shape ``(seq_len, head_dim // 2)``
    used to apply rotary invariance to Q and K tensors.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (max_seq_len, dim/2)
        cos, sin = freqs.cos(), freqs.sin()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self._max_seq_len = max_seq_len

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({self._max_seq_len})"
            )
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to *x*.

    Parameters
    ----------
    x : Tensor, shape ``(batch, n_heads, seq_len, head_dim)``
    cos, sin : Tensor, shape ``(seq_len, head_dim // 2)``

    Returns
    -------
    Tensor of same shape as *x* with rotary applied.
    """
    # x: (B, H, S, D) → split halves → (B, H, S, D/2)
    x1, x2 = x.chunk(2, dim=-1)
    # Broadcast cos/sin: (1, 1, S, D/2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    # Rotate
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


# ────────────────────────────────────────────────────────
# Misc helpers
# ────────────────────────────────────────────────────────

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat K/V along the head dimension for GQA.

    If *n_rep* == 1, returns *x* unchanged.

    Parameters
    ----------
    x : Tensor ``(batch, n_kv_heads, seq_len, head_dim)``
    n_rep : int — how many query heads share each KV head.

    Returns
    -------
    Tensor ``(batch, n_kv_heads * n_rep, seq_len, head_dim)``
    """
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    x = x.unsqueeze(3).expand(B, H, S, n_rep, D)
    return x.reshape(B, H * n_rep, S, D)


def _default_init(tensor: torch.Tensor, scale: float = 0.02) -> None:
    """Kaiming-style init used as default for linear layers."""
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
    tensor.data.mul_(scale / tensor.data.std())
