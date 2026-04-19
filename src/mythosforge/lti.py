"""
mythosforge.lti — Stable Linear Time-Invariant (LTI) recurrent state injection.

The LTI mechanism provides a stable recurrence for the hidden state across
loop iterations inside the recurrent block:

    h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)

where:
* ``A`` is a scalar or diagonal matrix in **(0, 1)** by parametric construction,
  guaranteeing spectral stability (eigenvalues < 1).
* ``B`` is a learnable projection that injects the residual input ``e``.
* The Transformer part is the main recurrent block output.

The ``A`` matrix is parameterised in log-space so that it can never leave (0, 1):

    A = exp(-exp(clamp(log_dt + log_A, -20, 20)))

and further clamped to be strictly below 1.0 to avoid float32 saturation.

References
----------
* Parcae: Scaling Laws for Looped Language Models  https://arxiv.org/abs/2604.12946
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import MythosConfig


class LTIInjection(nn.Module):
    """Stable Linear Time-Invariant state injection.

    Maintains a learnable recurrent state ``h_t`` that evolves across loop
    iterations.  The decay factor ``A`` is guaranteed to be in ``(0, 1)`` by
    construction, ensuring the recurrence is always contractive.

    Parameters
    ----------
    config : MythosConfig

    State shape
    -----------
    ``h`` is ``(*batch, dim)`` — matching the model's hidden dimension.
    """

    def __init__(self, config: MythosConfig):
        super().__init__()
        self.dim = config.dim

        # Learnable log-space parameters
        self.log_A = nn.Parameter(torch.zeros(config.dim))
        self.log_dt = nn.Parameter(torch.zeros(1))

        # Projection for injecting the input encoding e into the state
        self.B = nn.Linear(config.dim, config.dim, bias=False)

        self.reset_state()

    def reset_state(self) -> None:
        """Reset the recurrent state to zeros.

        Call this at the beginning of each new sequence.
        """
        self._state: torch.Tensor | None = None

    def get_A(self) -> torch.Tensor:
        """Compute the stable decay matrix A ∈ (0, 1).

        Returns
        -------
        Tensor ``(*, dim)`` — strictly positive, strictly less than 1.
        """
        A = torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))
        # Clamp below 1.0 to avoid float32 saturation
        A = torch.minimum(
            A,
            torch.nextafter(torch.ones_like(A), torch.zeros_like(A)),
        )
        return A

    def init_state(self, batch_shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialise or re-initialise the recurrent state.

        Parameters
        ----------
        batch_shape : Size — e.g. ``(batch_size,)`` or ``(batch_size, seq_len)``.
        device : torch.device
        dtype : torch.dtype

        Returns
        -------
        Tensor ``(*batch_shape, dim)`` — zero-initialised state.
        """
        state = torch.zeros(*batch_shape, self.dim, device=device, dtype=dtype)
        self._state = state
        return state

    def forward(
        self,
        h_prev: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        """One step of the LTI recurrence.

        Parameters
        ----------
        h_prev : Tensor ``(*batch, dim)`` — previous recurrent state.
        encoding : Tensor ``(*batch, dim)`` — input encoding to inject.

        Returns
        -------
        Tensor ``(*batch, dim)`` — updated recurrent state.
        """
        A = self.get_A()
        B_enc = self.B(encoding)
        h_new = A * h_prev + B_enc
        self._state = h_new
        return h_new

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
