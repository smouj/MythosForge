"""
mythosforge.moe — Mixture of Experts with routed + shared experts.

Architecture inspired by DeepSeekMoE (https://arxiv.org/abs/2401.06066):

* **Routed experts**: a sparse gating network selects top-K experts per token.
* **Shared experts**: always active, summed with the routed output.
* **Top-K routing**: load-balanced via auxiliary loss (not applied here —
  left to the training loop).

This design maximises the performance-per-compute ratio inside the recurrent
block where every FLOP matters because the block is executed N times.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MythosConfig
from .utils import RMSNorm, SwiGLU


# ═══════════════════════════════════════════════════════════
# Single Expert (SwiGLU FFN)
# ═══════════════════════════════════════════════════════════

class Expert(nn.Module):
    """A single SwiGLU feed-forward expert.

    Parameters
    ----------
    dim : int — input / output dimension.
    hidden_dim : int — intermediate (expert) dimension.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


# ═══════════════════════════════════════════════════════════
# Top-K Router
# ═══════════════════════════════════════════════════════════

class TopKRouter(nn.Module):
    """Learnable top-K gating network.

    Projects the input to ``n_experts`` logits, applies softmax, and selects
    the top-K scoring experts per token.  Returns the gating weights for the
    selected experts.

    Parameters
    ----------
    dim : int
    n_experts : int
    top_k : int
    """

    def __init__(self, dim: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to top-K experts.

        Parameters
        ----------
        x : Tensor ``(batch, seq_len, dim)``

        Returns
        -------
        gates : Tensor ``(batch, seq_len, top_k)`` — softmax weights for selected experts.
        indices : Tensor ``(batch, seq_len, top_k)`` — long tensor of expert indices.
        """
        logits = self.gate(x)  # (B, S, E)
        scores = F.softmax(logits, dim=-1)
        top_k_vals, top_k_idx = scores.topk(self.top_k, dim=-1)
        # Re-normalise top-k scores
        gates = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-9)
        return gates, top_k_idx


# ═══════════════════════════════════════════════════════════
# MoE Layer (routed + shared)
# ═══════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    """Full MoE FFN with routed + shared experts.

    Parameters
    ----------
    config : MythosConfig
    """

    def __init__(self, config: MythosConfig):
        super().__init__()
        self.dim = config.dim
        self.n_experts = config.n_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.n_experts_per_tok

        # Routed experts
        self.experts = nn.ModuleList([
            Expert(config.dim, config.expert_dim) for _ in range(config.n_experts)
        ])
        self.router = TopKRouter(config.dim, config.n_experts, self.top_k)

        # Shared experts (always active, separate parameters)
        if self.n_shared > 0:
            self.shared_experts = nn.ModuleList([
                Expert(config.dim, config.expert_dim) for _ in range(self.n_shared)
            ])
            # Gating weights for shared experts (learnable scalar per shared expert)
            self.shared_gate = nn.Parameter(torch.ones(self.n_shared))
        else:
            self.shared_experts = None
            self.shared_gate = None

        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with top-K routing + shared expert summation.

        Parameters
        ----------
        x : Tensor ``(batch, seq_len, dim)``

        Returns
        -------
        Tensor ``(batch, seq_len, dim)`` — residual connection included.
        """
        B, S, D = x.shape
        residual = x
        h = self.norm(x)

        # ── Routed experts ──────────────────────────────
        gates, indices = self.router(h)  # (B, S, top_k), (B, S, top_k)

        # Flatten for batched expert execution
        h_flat = h.view(-1, D)  # (B*S, D)
        gates_flat = gates.view(-1, self.top_k)  # (B*S, top_k)
        idx_flat = indices.view(-1, self.top_k)  # (B*S, top_k)

        # Dispatch to experts and combine
        routed_out = torch.zeros_like(h_flat)  # (B*S, D)
        for k in range(self.top_k):
            expert_idx = idx_flat[:, k]  # (B*S,)
            expert_weights = gates_flat[:, k].unsqueeze(-1)  # (B*S, 1)
            for e in range(self.n_experts):
                mask = expert_idx == e
                if mask.any():
                    token_slice = h_flat[mask]
                    expert_out = self.experts[e](token_slice)
                    routed_out[mask] += expert_weights[mask] * expert_out

        routed_out = routed_out.view(B, S, D)

        # ── Shared experts ──────────────────────────────
        shared_out = torch.zeros_like(h)
        if self.shared_experts is not None:
            shared_weights = F.softmax(self.shared_gate, dim=0)
            for i, expert in enumerate(self.shared_experts):
                shared_out = shared_out + shared_weights[i] * expert(h)

        return residual + routed_out + shared_out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_experts={self.n_experts}, "
            f"n_shared={self.n_shared}, top_k={self.top_k}"
        )
