"""
Tests for mythosforge — Pure PyTorch OpenMythos implementation.

Run with:
    python -m pytest tests/ -v

Each test uses a tiny config to keep runtime fast (< 10s total on CPU).
"""

import sys
import os
import math

import pytest
import torch

# Add src to path so we can import mythosforge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mythosforge import OpenMythos, MythosConfig
from mythosforge.attention import GroupedQueryAttention, MultiLatentAttention
from mythosforge.moe import MoELayer
from mythosforge.lti import LTIInjection
from mythosforge.act import AdaptiveComputationTime
from mythosforge.utils import RMSNorm, SwiGLU, RotaryEmbedding, apply_rotary_emb, repeat_kv


# ── Shared tiny configs ──────────────────────────────────

def gqa_config() -> MythosConfig:
    return MythosConfig(
        vocab_size=256,
        dim=64,
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
        attn_type="gqa",
    )


def mla_config() -> MythosConfig:
    return MythosConfig(
        vocab_size=256,
        dim=64,
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
        attn_type="mla",
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )


# ═══════════════════════════════════════════════════════════
# Utils
# ═══════════════════════════════════════════════════════════

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_not_nan(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64) * 100
        out = norm(x)
        assert not torch.isnan(out).any()


class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLU(64, 128)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == x.shape


class TestRotaryEmbedding:
    def test_cos_sin_shape(self):
        rope = RotaryEmbedding(dim=16, max_seq_len=32)
        cos, sin = rope(10)
        assert cos.shape == (10, 8)
        assert sin.shape == (10, 8)

    def test_apply_rotary_shape(self):
        x = torch.randn(2, 4, 10, 16)
        rope = RotaryEmbedding(dim=16, max_seq_len=32)
        cos, sin = rope(10)
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape


class TestRepeatKV:
    def test_identity_when_n1(self):
        x = torch.randn(2, 4, 10, 16)
        out = repeat_kv(x, 1)
        assert torch.allclose(x, out)

    def test_expand(self):
        x = torch.randn(2, 2, 10, 16)
        out = repeat_kv(x, 3)
        assert out.shape == (2, 6, 10, 16)


# ═══════════════════════════════════════════════════════════
# Attention
# ═══════════════════════════════════════════════════════════

class TestGQA:
    def test_forward_shape(self):
        cfg = gqa_config()
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, cfg.dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_no_nan(self):
        cfg = gqa_config()
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, cfg.dim)
        out = attn(x)
        assert not torch.isnan(out).any()


class TestMLA:
    def test_forward_shape(self):
        cfg = mla_config()
        attn = MultiLatentAttention(cfg)
        x = torch.randn(2, 8, cfg.dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_no_nan(self):
        cfg = mla_config()
        attn = MultiLatentAttention(cfg)
        x = torch.randn(2, 8, cfg.dim)
        out = attn(x)
        assert not torch.isnan(out).any()


# ═══════════════════════════════════════════════════════════
# MoE
# ═══════════════════════════════════════════════════════════

class TestMoE:
    def test_forward_shape(self):
        cfg = gqa_config()
        moe = MoELayer(cfg)
        x = torch.randn(2, 8, cfg.dim)
        out = moe(x)
        assert out.shape == x.shape

    def test_no_nan(self):
        cfg = gqa_config()
        moe = MoELayer(cfg)
        x = torch.randn(2, 8, cfg.dim)
        out = moe(x)
        assert not torch.isnan(out).any()


# ═══════════════════════════════════════════════════════════
# LTI
# ═══════════════════════════════════════════════════════════

class TestLTI:
    def test_A_stable(self):
        cfg = gqa_config()
        lti = LTIInjection(cfg)
        A = lti.get_A()
        assert (A > 0).all(), "A must be > 0"
        assert (A < 1).all(), "A must be < 1"

    def test_init_state_shape(self):
        cfg = gqa_config()
        lti = LTIInjection(cfg)
        state = lti.init_state(torch.Size([2, 8]), torch.device("cpu"), torch.float32)
        assert state.shape == (2, 8, cfg.dim)
        assert (state == 0).all()

    def test_forward(self):
        cfg = gqa_config()
        lti = LTIInjection(cfg)
        h = torch.randn(2, 8, cfg.dim)
        enc = torch.randn(2, 8, cfg.dim)
        h_new = lti(h, enc)
        assert h_new.shape == h.shape
        assert not torch.isnan(h_new).any()


# ═══════════════════════════════════════════════════════════
# ACT
# ═══════════════════════════════════════════════════════════

class TestACT:
    def test_halt_mask_shape(self):
        cfg = gqa_config()
        act = AdaptiveComputationTime(cfg)
        h = torch.randn(2, 8, cfg.dim)
        mask, p = act(h)
        assert mask.shape == (2, 8, 1)
        assert p.shape == (2, 8, 1)

    def test_halt_prob_range(self):
        cfg = gqa_config()
        act = AdaptiveComputationTime(cfg)
        h = torch.randn(2, 8, cfg.dim)
        _, p = act(h)
        assert (p >= cfg.act_eps).all()
        assert (p <= 1.0 + cfg.act_eps).all()

    def test_accumulation(self):
        cfg = gqa_config()
        act = AdaptiveComputationTime(cfg)
        h = torch.randn(2, 8, cfg.dim)
        for _ in range(5):
            mask, p = act(h)
        # After 5 iterations, most positions should have halted
        assert mask.sum() < 2 * 8  # fewer than all positions still running


# ═══════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════

class TestOpenMythosGQA:
    def setup_method(self):
        self.cfg = gqa_config()
        self.model = OpenMythos(self.cfg)
        self.model.eval()

    def test_forward_shape(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits = self.model(ids, n_loops=2)
        assert logits.shape == (1, 8, self.cfg.vocab_size)

    def test_no_nan(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits = self.model(ids, n_loops=2)
        assert not torch.isnan(logits).any()

    def test_generate_shape(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=2, n_loops=2)
        assert out.shape[1] == 6  # 4 prompt + 2 generated

    def test_count_parameters(self):
        counts = self.model.count_parameters()
        assert "total" in counts
        assert "trainable" in counts
        assert counts["total"] > 0
        assert counts["trainable"] == counts["total"]

    def test_A_stable_in_model(self):
        A = self.model.injection.get_A()
        assert (A > 0).all()
        assert (A < 1).all()

    def test_batch_forward(self):
        ids = torch.randint(0, self.cfg.vocab_size, (3, 8))
        with torch.no_grad():
            logits = self.model(ids, n_loops=2)
        assert logits.shape == (3, 8, self.cfg.vocab_size)


class TestOpenMythosMLA:
    def setup_method(self):
        self.cfg = mla_config()
        self.model = OpenMythos(self.cfg)
        self.model.eval()

    def test_forward_shape(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits = self.model(ids, n_loops=2)
        assert logits.shape == (1, 8, self.cfg.vocab_size)

    def test_no_nan(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits = self.model(ids, n_loops=2)
        assert not torch.isnan(logits).any()

    def test_generate_shape(self):
        ids = torch.randint(0, self.cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=2, n_loops=2)
        assert out.shape[1] == 6


# ═══════════════════════════════════════════════════════════
# End-to-end smoke test
# ═══════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_gqa_vs_mla_same_shape(self):
        """GQA and MLA must produce same output shape."""
        ids = torch.randint(0, 256, (1, 8))
        gqa_model = OpenMythos(gqa_config())
        mla_model = OpenMythos(mla_config())
        gqa_model.eval()
        mla_model.eval()

        with torch.no_grad():
            logits_gqa = gqa_model(ids, n_loops=2)
            logits_mla = mla_model(ids, n_loops=2)

        assert logits_gqa.shape == logits_mla.shape

    def test_variable_loops(self):
        """Model should work with 1, 2, or max_loop_iters."""
        cfg = gqa_config()
        model = OpenMythos(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))

        for n_loops in [1, 2]:
            with torch.no_grad():
                logits = model(ids, n_loops=n_loops)
            assert logits.shape == (1, 4, cfg.vocab_size)
            assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
