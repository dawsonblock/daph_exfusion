"""Tests for AdaptiveTopPMacroRouter and DAPHDecoderLayerV2."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from daph_exfusion.adaptive_top_p_router import AdaptiveTopPMacroRouter, DAPHDecoderLayerV2
from daph_exfusion.merge_toolkit import MemoryBankExFusionFFN, MemoryBankExFusionMamba


def test_adaptive_router_basic():
    B, L, D = 2, 5, 16
    router = AdaptiveTopPMacroRouter(D, num_paths=3, base_threshold=0.85, difficulty_scale=0.3)
    x = torch.randn(B, L, D)

    mask, probs = router(x)
    assert mask.shape == (B, L, 3)
    assert probs.shape == (B, L, 3)
    # Each token must have at least one active path
    assert mask.sum(dim=-1).min() >= 1
    # Probabilities must sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B, L), atol=1e-5)


def test_adaptive_router_external_difficulty():
    B, L, D = 2, 3, 8
    router = AdaptiveTopPMacroRouter(D, num_paths=3, use_external_difficulty=True)
    x = torch.randn(B, L, D)
    diff = torch.rand(B, L)

    mask, probs = router(x, external_difficulty=diff)
    assert mask.shape == (B, L, 3)
    # High difficulty should lower threshold → more paths active
    easy_mask, _ = router(x, external_difficulty=torch.zeros(B, L))
    hard_mask, _ = router(x, external_difficulty=torch.ones(B, L))
    assert easy_mask.sum() <= hard_mask.sum()


def test_adaptive_router_threshold_monotonicity():
    """Higher difficulty should never decrease the number of active paths."""
    B, L, D = 4, 8, 16
    router = AdaptiveTopPMacroRouter(D, num_paths=3, base_threshold=0.85, difficulty_scale=0.3)
    x = torch.randn(B, L, D)

    mask_low, _ = router(x, external_difficulty=torch.full((B, L), 0.1))
    mask_high, _ = router(x, external_difficulty=torch.full((B, L), 0.9))

    # On average, high difficulty should activate more paths
    assert mask_high.sum(dim=-1).float().mean() >= mask_low.sum(dim=-1).float().mean()


def test_daph_decoder_layer_v2_forward():
    hidden_size, intermediate_size = 16, 32
    num_experts, num_heads = 2, 4
    B, L = 2, 5

    ffn = MemoryBankExFusionFFN(
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_experts=num_experts, activation="swiglu", bias=False,
    )

    def make_mamba():
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
                self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.x_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.dt_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.A_log = nn.Parameter(torch.zeros(hidden_size))
                self.D = nn.Parameter(torch.zeros(hidden_size))

            def forward(self, x):
                h = self.in_proj(x)
                a, b = h[..., :hidden_size], h[..., hidden_size:]
                u = F.silu(a) * b
                return self.out_proj(u + self.D * u)
        return Block()

    mamba = MemoryBankExFusionMamba(make_mamba, num_experts, hidden_size)

    class DummyAttn(nn.Module):
        def forward(self, x, mask=None):
            return x

    layer = DAPHDecoderLayerV2(
        hidden_size=hidden_size,
        ffn_exfusion_factory=lambda: ffn,
        mamba_exfusion_factory=lambda: mamba,
        attention_factory=lambda: DummyAttn(),
        use_cheap_path=True,
    )

    x = torch.randn(B, L, hidden_size)
    out = layer(x)
    assert out.shape == x.shape


def test_decoder_v2_merge():
    """DAPHDecoderLayerV2 can merge its ExFusion paths."""
    hidden_size, intermediate_size = 16, 32
    num_experts = 2

    ffn = MemoryBankExFusionFFN(
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_experts=num_experts, activation="swiglu", bias=False,
    )

    def make_mamba():
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
                self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.x_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.dt_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.A_log = nn.Parameter(torch.zeros(hidden_size))
                self.D = nn.Parameter(torch.zeros(hidden_size))

            def forward(self, x):
                return self.out_proj(x)
        return Block()

    mamba = MemoryBankExFusionMamba(make_mamba, num_experts, hidden_size)

    layer = DAPHDecoderLayerV2(
        hidden_size=hidden_size,
        ffn_exfusion_factory=lambda: ffn,
        mamba_exfusion_factory=lambda: mamba,
        attention_factory=None,
        use_cheap_path=False,
    )

    # Warm up memory banks
    x = torch.randn(2, 4, hidden_size)
    for _ in range(3):
        layer(x)

    # Build dummy Fisher diagonals
    ffn_fishers = [
        {name: torch.rand_like(p) + 1e-3 for name, p in e.named_parameters()}
        for e in ffn.experts
    ]
    mamba_fishers = [
        {name: torch.rand_like(p) + 1e-3 for name, p in e.named_parameters()}
        for e in mamba.experts
    ]

    layer.merge_exfusion_paths(
        path="both",
        pipeline=["dare", "ties", "fisher"],
        fisher_diagonals=ffn_fishers,
        mamba_fisher_diagonals=mamba_fishers,
        seed=0,
    )

    assert ffn.is_merged
    assert mamba.is_merged
    out = layer(x)
    assert out.shape == x.shape
