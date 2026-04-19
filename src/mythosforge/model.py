"""
mythosforge.model — Full OpenMythos recurrent-depth transformer.

Architecture:
    Input Tokens
         │
         ▼
    ┌─────────────┐
    │   PRELUDE   │  Dense Transformer blocks × prelude_layers
    └──────┬──────┘
           │ h₀
           ▼
    ┌──────────────────────────────────────────┐
    │        RECURRENT BLOCK (×N loops)        │
    │  Switchable Attention (MLA / GQA)        │
    │  MoE FFN (Routed + Shared Experts)       │
    │  LoRA Adapter (per-iteration)            │
    │  LTI Injection (stable A ∈ (0,1))        │
    │  ACT Halting (per-position stopping)     │
    └────────────────────┬─────────────────────┘
                         │ h_T
                         ▼
    ┌─────────────┐
    │    CODA     │  Dense Transformer blocks × coda_layers
    └──────┬──────┘
           │
           ▼
      RMSNorm → LM Head → Output Logits
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .block import RecurrentBlock
from .config import MythosConfig
from .lti import LTIInjection
from .utils import RMSNorm, SwiGLU


# ═══════════════════════════════════════════════════════════
# Dense Transformer Block (Prelude / Coda)
# ═══════════════════════════════════════════════════════════

class DenseBlock(nn.Module):
    """Standard dense transformer block with attention + SwiGLU FFN.

    Used in the Prelude and Coda stages (executed once, not looped).

    Parameters
    ----------
    config : MythosConfig
    """

    def __init__(self, config: MythosConfig):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = SwiGLU(config.dim, config.expert_dim)
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor ``(batch, seq_len, dim)``
        mask : Optional causal mask.

        Returns
        -------
        Tensor ``(batch, seq_len, dim)``
        """
        x = self.attn(x, mask=mask)
        residual = x
        x = self.norm(x)
        x = residual + self.ffn(x)
        return x


# ═══════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════

class OpenMythos(nn.Module):
    """OpenMythos — Recurrent-Depth Transformer.

    Parameters
    ----------
    config : MythosConfig
        Full configuration for the model.

    Example
    -------
    >>> from mythosforge import OpenMythos, MythosConfig
    >>> cfg = MythosConfig(vocab_size=256, dim=64, n_heads=4, n_kv_heads=4,
    ...                     max_seq_len=64, max_loop_iters=2, prelude_layers=1,
    ...                     coda_layers=1, n_experts=4, n_shared_experts=1,
    ...                     n_experts_per_tok=1, expert_dim=16, lora_rank=4,
    ...                     attn_type="gqa")
    >>> model = OpenMythos(cfg)
    >>> ids = torch.randint(0, 256, (1, 8))
    >>> logits = model(ids, n_loops=2)
    >>> logits.shape
    torch.Size([1, 8, 256])
    """

    def __init__(self, config: MythosConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.tok_embed = nn.Embedding(config.vocab_size, config.dim)

        # Prelude — dense transformer blocks
        self.prelude = nn.ModuleList([
            DenseBlock(config) for _ in range(config.prelude_layers)
        ])

        # Recurrent block (single block, executed N times)
        self.recurrent = RecurrentBlock(config)

        # LTI state injection
        self.injection = LTIInjection(config)

        # Coda — dense transformer blocks
        self.coda = nn.ModuleList([
            DenseBlock(config) for _ in range(config.coda_layers)
        ])

        # Final norm + LM head
        self.final_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying between embeddings and LM head
        self.lm_head.weight = self.tok_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialization for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a standard causal attention mask.

        Returns
        -------
        Tensor ``(1, 1, seq_len, seq_len)`` — 0 for allowed, -inf for masked.
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
    ) -> torch.Tensor:
        """Full forward pass: Prelude → Recurrent(×N) → Coda → logits.

        Parameters
        ----------
        input_ids : LongTensor ``(batch, seq_len)``
            Token IDs from the tokenizer.
        n_loops : int or None
            Number of recurrent loop iterations.  If ``None``, uses
            ``config.max_loop_iters``.

        Returns
        -------
        Tensor ``(batch, seq_len, vocab_size)`` — output logits.
        """
        if n_loops is None:
            n_loops = self.config.max_loop_iters
        n_loops = min(n_loops, self.config.max_loop_iters)

        B, S = input_ids.shape
        device = input_ids.device
        dtype = self.tok_embed.weight.dtype

        # Causal mask for attention
        causal_mask = self._make_causal_mask(S, device)

        # ── Token Embedding ────────────────────────────
        x = self.tok_embed(input_ids)  # (B, S, dim)

        # ── Prelude ────────────────────────────────────
        for block in self.prelude:
            x = block(x, mask=causal_mask)

        # Save encoding for LTI injection
        encoding = x.detach()  # (B, S, dim) — stop gradients for LTI input

        # ── Recurrent Block × N ────────────────────────
        # Reset LTI and ACT state for new sequence
        self.injection.reset_state()
        self.recurrent.act.reset_state()

        # Initialise LTI state
        h = self.injection.init_state((B, S), device, dtype)
        h = h + x  # h₀ = 0 + prelude_output

        for t in range(n_loops):
            # Transformer block
            block_out, _, halt_mask = self.recurrent(h, encoding, t)

            # LTI state update
            h = self.injection(h, encoding) + block_out

        # ACT remainder: ensure halted positions get a final weighted update
        remainder = self.recurrent.act.get_remainder()
        if remainder is not None and n_loops > 0:
            block_out, _, _ = self.recurrent(h, encoding, n_loops)
            h = h + remainder * block_out

        x = h

        # ── Coda ───────────────────────────────────────
        for block in self.coda:
            x = block(x, mask=causal_mask)

        # ── LM Head ────────────────────────────────────
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        n_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive token generation.

        Parameters
        ----------
        input_ids : LongTensor ``(batch, seq_len)`` — prompt tokens.
        max_new_tokens : int — maximum tokens to generate.
        n_loops : int or None — recurrent loop iterations.
        temperature : float — sampling temperature (1.0 = greedy argmax).
        top_k : int — top-k sampling filter (0 = disabled).

        Returns
        -------
        LongTensor ``(batch, seq_len + generated)`` — full sequence.
        """
        self.eval()
        tokens = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(tokens[:, -self.config.max_seq_len:], n_loops=n_loops)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component.

        Returns
        -------
        dict with keys: ``total``, ``embedding``, ``prelude``, ``recurrent``,
        ``injection``, ``coda``, ``lm_head``.
        """
        counts: dict[str, int] = {}
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        counts["embedding"] = sum(p.numel() for p in self.tok_embed.parameters())
        counts["prelude"] = sum(p.numel() for p in self.prelude.parameters())
        counts["recurrent"] = sum(p.numel() for p in self.recurrent.parameters())
        counts["injection"] = sum(p.numel() for p in self.injection.parameters())
        counts["coda"] = sum(p.numel() for p in self.coda.parameters())
        counts["lm_head"] = sum(p.numel() for p in self.lm_head.parameters())

        return counts

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.config.vocab_size}, dim={self.config.dim}, "
            f"n_heads={self.config.n_heads}, attn_type={self.config.attn_type!r}, "
            f"max_loop_iters={self.config.max_loop_iters}, "
            f"prelude_layers={self.config.prelude_layers}, "
            f"coda_layers={self.config.coda_layers}"
        )
