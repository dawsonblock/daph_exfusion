"""Tests for the merge toolkit core logic."""
import math
import torch
import pytest

from daph_exfusion.merge_toolkit import (
    MemoryBankExFusionFFN,
    MemoryBankExFusionMamba,
    DAPHDecoderLayer,
    incorporate_kfac_scores,
    SwiGLUFFN,
)


def test_swiglu_ffn_forward():
    up = torch.nn.Linear(16, 32, bias=True)
    gate = torch.nn.Linear(16, 32, bias=True)
    down = torch.nn.Linear(32, 16, bias=True)
    ffn = SwiGLUFFN(up, gate, down)
    x = torch.randn(2, 5, 16)
    out = ffn(x)
    assert out.shape == x.shape


def test_memory_bank_ffn_forward_and_merge():
    ffn = MemoryBankExFusionFFN(
        hidden_size=16, intermediate_size=32, num_experts=3, activation="swiglu", bias=True
    )
    x = torch.randn(2, 5, 16)
    out = ffn(x)
    assert out.shape == x.shape

    fishers = []
    for e in ffn.experts:
        d = {}
        for name, p in e.named_parameters():
            d[name] = torch.rand_like(p.data) + 1e-3
        fishers.append(d)

    ffn.merge_to_dense(
        pipeline=["dare", "ties", "fisher"],
        fisher_diagonals=fishers,
        seed=0,
    )
    assert ffn.is_merged
    merged_out = ffn(x)
    assert merged_out.shape == x.shape
    assert ffn.merged_ffn.up.bias.shape == (32,)


def test_bias_preserved_in_merge():
    """Explicit regression test for the bias-discard bug."""
    ffn = MemoryBankExFusionFFN(
        hidden_size=8, intermediate_size=16, num_experts=2, activation="swiglu", bias=True
    )
    fishers = [
        {name: torch.rand_like(p) + 1e-3 for name, p in e.named_parameters()}
        for e in ffn.experts
    ]
    ffn.merge_to_dense(pipeline=["dare", "ties", "fisher"], fisher_diagonals=fishers, seed=0)
    assert ffn.merged_ffn.up.bias is not None
    assert ffn.merged_ffn.gate.bias is not None
    assert ffn.merged_ffn.down.bias is not None


def test_kfac_score_mismatch_raises():
    with pytest.raises(ValueError, match="K-FAC score count"):
        incorporate_kfac_scores(
            memory_bank=torch.ones(4),
            kfac_scores={"layer_0": 1.0, "layer_1": 2.0},
        )


def test_mamba_seed_determinism():
    """Two runs with the same seed must produce identical merged parameters."""
    hidden_size = 8
    num_experts = 2

    def make_block():
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = torch.nn.Linear(hidden_size, hidden_size * 2, bias=False)
                self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.x_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.dt_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.A_log = torch.nn.Parameter(torch.zeros(hidden_size))
                self.D = torch.nn.Parameter(torch.zeros(hidden_size))

            def forward(self, x):
                return self.out_proj(x)
        return Block()

    # Build dummy fisher diagonals
    def make_fishers(mamba):
        return [
            {name: torch.rand_like(p) + 1e-3 for name, p in e.named_parameters()}
            for e in mamba.experts
        ]

    mamba1 = MemoryBankExFusionMamba(make_block, num_experts, hidden_size)
    mamba1.merge_to_dense(fisher_diagonals=make_fishers(mamba1), seed=42)
    p1 = {name: param.clone() for name, param in mamba1.merged_mamba.named_parameters()}

    mamba2 = MemoryBankExFusionMamba(make_block, num_experts, hidden_size)
    mamba2.merge_to_dense(fisher_diagonals=make_fishers(mamba2), seed=42)
    p2 = {name: param.clone() for name, param in mamba2.merged_mamba.named_parameters()}

    for name in p1.keys():
        assert torch.allclose(p1[name], p2[name])
