"""
mythosforge.block — Recurrent Block with LoRA, ACT, and LTI.

The RecurrentBlock is the core unit that gets executed N times during a single
forward pass.  It contains:

1. **Switchable Attention** (GQA or MLA)
2. **MoE FFN** (routed + shared experts)
3. **LoRA Adapter** — per-iteration low-rank adaptation driven by a sinusoidal
   depth signal so that each loop iteration behaves slightly differently.
4. **ACT Halting** — per-position stopping criterion.
5. **LTI Injection** — stable recurrent state accumulation.

Each iteration:
    h_{t+1} = ACT(h_t) · [A · h_t + B · e + LoRA_t(Attn(MoE(h_t)))]

Only the LoRA adapter changes per iteration; the attention, MoE, LTI, and ACT
parameters are shared across all iterations (weight-tied).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .act import AdaptiveComputationTime
from .attention import Attention
from .config import MythosConfig
from .lti import LTIInjection
from .moe import MoELayer
from .utils import RMSNorm


# ═══════════════════════════════════════════════════════════
# Sinusoidal Depth Signal
# ═══════════════════════════════════════════════════════════

class DepthSignal(nn.Module):
    """Sinusoidal embedding of the loop iteration index.

    Produces a vector of size ``rank`` that encodes which iteration we are
    on, similar to positional encodings but over the depth axis.  This is fed
    into the LoRA adapter to modulate each iteration's behavior.

    Parameters
    ----------
    max_iters : int — maximum number of loop iterations.
    rank : int — dimensionality of the depth embedding (matches LoRA rank).
    """

    def __init__(self, max_iters: int, rank: int):
        super().__init__()
        self.max_iters = max_iters
        self.rank = rank
        # Pre-compute embeddings: (max_iters, rank)
        pe = torch.zeros(max_iters, rank)
        position = torch.arange(0, max_iters, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, rank, 2, dtype=torch.float32) * -(math.log(10000.0) / rank)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, iteration: int) -> torch.Tensor:
        """Get depth signal for a specific iteration.

        Returns
        -------
        Tensor ``(rank,)``
        """
        idx = min(iteration, self.max_iters - 1)
        return self.pe[idx]


# ═══════════════════════════════════════════════════════════
# Per-Iteration LoRA Adapter
# ═══════════════════════════════════════════════════════════

class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation modulated by a sinusoidal depth signal.

    Instead of a fixed LoRA, the B matrix is computed as:

        B_t = B_base + depth_signal(t) @ W_depth

    This gives each loop iteration a slightly different adaptation while sharing
    most parameters.

    Parameters
    ----------
    dim : int
    rank : int
    """

    def __init__(self, dim: int, rank: int, max_iters: int):
        super().__init__()
        self.dim = dim
        self.rank = rank

        self.down = nn.Linear(dim, rank, bias=False)
        self.up_base = nn.Linear(rank, dim, bias=False)

        # Depth modulation: depth_signal (rank) → contribution to B (rank, dim)
        self.depth_proj = nn.Linear(rank, rank * dim, bias=False)
        nn.init.zeros_(self.depth_proj.weight)

        self.depth_signal = DepthSignal(max_iters, rank)

    def forward(self, x: torch.Tensor, iteration: int) -> torch.Tensor:
        """Apply LoRA with depth-dependent modulation.

        Parameters
        ----------
        x : Tensor ``(batch, seq_len, dim)``
        iteration : int — current loop iteration index.

        Returns
        -------
        Tensor ``(batch, seq_len, dim)`` — LoRA residual.
        """
        h = self.down(x)  # (B, S, rank)
        depth = self.depth_signal(iteration)  # (rank,)

        # Base LoRA
        base_out = self.up_base(h)  # (B, S, dim)

        # Depth-modulated contribution
        depth_weight = self.depth_proj(depth)  # (rank * dim,)
        depth_weight = depth_weight.view(self.rank, self.dim)  # (rank, dim)
        depth_out = h @ depth_weight  # (B, S, dim)

        return base_out + depth_out


# ═══════════════════════════════════════════════════════════
# Recurrent Block (single iteration)
# ═══════════════════════════════════════════════════════════

class RecurrentBlock(nn.Module):
    """One iteration of the recurrent block.

    Contains attention, MoE, LoRA, and ACT halting.  The LTI state injection
    is handled by the parent model to keep the recurrence explicit.

    Parameters
    ----------
    config : MythosConfig
    """

    def __init__(self, config: MythosConfig):
        super().__init__()
        self.config = config

        # Attention (GQA or MLA)
        self.attn = Attention(config)

        # MoE FFN
        self.moe = MoELayer(config)

        # LoRA per-iteration adapter
        if config.lora_rank > 0:
            self.lora = LoRAAdapter(config.dim, config.lora_rank, config.max_loop_iters)
        else:
            self.lora = None

        # ACT halting
        self.act = AdaptiveComputationTime(config)

        # Pre-norm for the block input
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)

    def forward(
        self,
        h: torch.Tensor,
        encoding: torch.Tensor,
        iteration: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one recurrent block iteration.

        Parameters
        ----------
        h : Tensor ``(batch, seq_len, dim)`` — current hidden state.
        encoding : Tensor ``(batch, seq_len, dim)`` — residual input encoding.
        iteration : int — current loop iteration (0-indexed).

        Returns
        -------
        h_new : Tensor ``(batch, seq_len, dim)`` — updated hidden state (after LTI + ACT).
        block_out : Tensor ``(batch, seq_len, dim)`` — transformer block output (before LTI).
        halt_mask : Tensor ``(batch, seq_len, 1)`` — 1.0 for still-running positions.
        """
        # ACT: compute which positions should keep running
        halt_mask, halt_prob = self.act(h)

        # Attention + MoE
        block_out = self.attn(h)
        block_out = self.moe(block_out)

        # LoRA adaptation (depth-modulated)
        if self.lora is not None:
            block_out = block_out + self.lora(h, iteration)

        # Apply ACT mask: halted positions stop accumulating
        block_out = block_out * halt_mask

        return block_out, block_out, halt_mask
