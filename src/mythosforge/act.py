"""
mythosforge.act — Adaptive Computation Time (ACT) halting mechanism.

ACT (Graves, 2016) allows the model to learn *when to stop* processing each
position in a sequence independently.  A small halting network produces a
sigmoid probability ``p_t`` at each iteration.  Positions accumulate halting
probability until they cross a threshold (default 0.99), at which point they
stop receiving updates.

This is crucial for the recurrent block: not all tokens need the same number
of loop iterations.  Easy tokens (punctuation, common words) can halt early,
saving computation for harder tokens that need more reasoning.

The halting state is reset at the beginning of each new sequence.

References
----------
* Adaptive Computation Time for Recurrent Neural Networks (Graves, 2016)
  https://arxiv.org/abs/1603.08983
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MythosConfig


class AdaptiveComputationTime(nn.Module):
    """Per-position adaptive halting for the recurrent block.

    Parameters
    ----------
    config : MythosConfig

    Forward behaviour
    -----------------
    At each iteration the module produces a halting probability per position.
    Positions that have accumulated enough halting mass (>= ``threshold``)
    receive a mask of 0 (halted) so the recurrent block can skip them.
    """

    def __init__(self, config: MythosConfig):
        super().__init__()
        self.dim = config.dim
        self.eps = config.act_eps
        self.threshold = 0.99

        # Small halting network
        self.halt_proj = nn.Sequential(
            nn.Linear(config.dim, config.dim // 4),
            nn.SiLU(),
            nn.Linear(config.dim // 4, 1),
        )

        self.reset_state()

    def reset_state(self) -> None:
        """Reset cumulative halting probabilities.

        Call this at the beginning of each new sequence.
        """
        self._halt_sum: torch.Tensor | None = None
        self._still_running: torch.Tensor | None = None

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute halting probabilities and update running state.

        Parameters
        ----------
        h : Tensor ``(batch, seq_len, dim)`` — current hidden state.

        Returns
        -------
        halt_mask : Tensor ``(batch, seq_len, 1)`` — 1.0 for positions still
            running, 0.0 for halted positions.
        halt_prob : Tensor ``(batch, seq_len, 1)`` — raw halting probability
            for this iteration (before accumulation).
        """
        # Raw halting sigmoid
        p = torch.sigmoid(self.halt_proj(h))  # (B, S, 1)

        # Clamp minimum to avoid division by zero later
        p = p + self.eps

        # Initialise halting state on first call
        if self._halt_sum is None:
            self._halt_sum = torch.zeros_like(p)
            self._still_running = torch.ones_like(p)

        # Clamp accumulated sum so it doesn't exceed 1.0
        new_sum = torch.clamp(self._halt_sum + p, max=1.0)
        self._halt_sum = new_sum

        # Positions that haven't reached threshold yet
        still_running = (new_sum < self.threshold).float()
        # Once halted, always halted
        if self._still_running is not None:
            still_running = still_running * self._still_running
        self._still_running = still_running

        # Mask for use in the recurrent block: 1 = still running, 0 = halted
        halt_mask = still_running

        return halt_mask, p

    def get_remainder(self) -> torch.Tensor | None:
        """Get the remaining halting mass for positions that were force-halted.

        After the last iteration, positions that haven't naturally reached the
        threshold are given a final update weighted by ``1 - halt_sum`` to
        ensure the total halting mass sums to exactly 1.0 for every position.

        Returns
        -------
        Tensor ``(batch, seq_len, 1)`` or ``None`` if not yet initialised.
        """
        if self._halt_sum is None:
            return None
        return torch.clamp(1.0 - self._halt_sum, min=0.0)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, threshold={self.threshold}"
