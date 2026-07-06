"""Tests for v4.5.0 router enhancements: multi-signal difficulty, cost-aware
routing, z-loss, exploration noise, learnable thresholds, diagnostics, and
MLX decode-mode specialisation."""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from daph_exfusion.adaptive_top_p_router import AdaptiveTopPMacroRouter, DAPHDecoderLayerV2


# ── Multi-signal difficulty ──────────────────────────────────────────────

def test_multi_signal_difficulty_shape():
    """Multi-signal combiner produces valid (B,L,3) mask and probs."""
    B, L, D = 2, 5, 16
    router = AdaptiveTopPMacroRouter(D, num_paths=3, multi_signal_difficulty=True)
    x = torch.randn(B, L, D)
    mask, probs = router(x)
    assert mask.shape == (B, L, 3)
    assert probs.shape == (B, L, 3)
    assert mask.sum(dim=-1).min() >= 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B, L), atol=1e-5)


def test_multi_signal_difficulty_has_combiner():
    """The difficulty_combiner module exists when multi_signal is enabled."""
    router = AdaptiveTopPMacroRouter(16, multi_signal_difficulty=True)
    assert hasattr(router, "difficulty_combiner")
    # Should not have combiner when disabled
    router2 = AdaptiveTopPMacroRouter(16, multi_signal_difficulty=False)
    assert not hasattr(router2, "difficulty_combiner")


def test_multi_signal_difficulty_monotonicity():
    """Multi-signal difficulty still respects threshold monotonicity."""
    B, L, D = 4, 8, 16
    router = AdaptiveTopPMacroRouter(D, num_paths=3, multi_signal_difficulty=True)
    x = torch.randn(B, L, D)
    mask_low, _ = router(x, external_difficulty=torch.full((B, L), 0.1))
    mask_high, _ = router(x, external_difficulty=torch.full((B, L), 0.9))
    assert mask_high.sum(dim=-1).float().mean() >= mask_low.sum(dim=-1).float().mean()


# ── Cost-aware routing ───────────────────────────────────────────────────

def test_cost_aware_routing_biases_cheap_path():
    """With cost penalty, the cheap path should be activated more often."""
    B, L, D = 4, 16, 32
    torch.manual_seed(42)
    x = torch.randn(B, L, D)

    # Without cost penalty
    router_no_cost = AdaptiveTopPMacroRouter(D, num_paths=3, path_costs=None)
    mask_no_cost, _ = router_no_cost(x)

    # With cost penalty — expensive paths (attn=3.0, eff=2.0) penalised
    router_cost = AdaptiveTopPMacroRouter(
        D, num_paths=3, path_costs=(3.0, 2.0, 0.5), cost_penalty=1.0
    )
    # Copy router weights so the comparison is fair
    router_cost.router.weight.data.copy_(router_no_cost.router.weight.data)
    mask_cost, _ = router_cost(x)

    # Cheap path (index 2) should be active more often with cost penalty
    cheap_no_cost = mask_no_cost[:, :, 2].sum().item()
    cheap_cost = mask_cost[:, :, 2].sum().item()
    assert cheap_cost >= cheap_no_cost


def test_cost_aware_routing_cost_log_bias_exists():
    """The cost_log_bias buffer is created when path_costs is provided."""
    router = AdaptiveTopPMacroRouter(16, path_costs=(3.0, 2.0, 0.5))
    assert router.cost_log_bias is not None
    assert router.cost_log_bias.shape == (3,)
    # Expensive paths should have more negative bias
    assert router.cost_log_bias[0] < router.cost_log_bias[2]

    router2 = AdaptiveTopPMacroRouter(16, path_costs=None)
    assert router2.cost_log_bias is None


# ── Router z-loss ────────────────────────────────────────────────────────

def test_z_loss_returns_scalar():
    """compute_z_loss returns a scalar tensor."""
    router = AdaptiveTopPMacroRouter(16, num_paths=3)
    x = torch.randn(2, 4, 16)
    z_loss = router.compute_z_loss(x)
    assert z_loss.dim() == 0  # scalar
    assert z_loss.item() >= 0  # squared values are non-negative


def test_z_loss_via_decoder_layer():
    """DAPHDecoderLayerV2 exposes compute_z_loss delegating to its router."""
    hidden_size, intermediate_size = 16, 32
    ffn = __import__("daph_exfusion.merge_toolkit", fromlist=["MemoryBankExFusionFFN"]).MemoryBankExFusionFFN(
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_experts=2, activation="swiglu", bias=False,
    )
    layer = DAPHDecoderLayerV2(
        hidden_size=hidden_size,
        ffn_exfusion_factory=lambda: ffn,
        attention_factory=None,
        use_cheap_path=True,
    )
    x = torch.randn(2, 4, hidden_size)
    z_loss = layer.compute_z_loss(x)
    assert z_loss.dim() == 0
    assert z_loss.item() >= 0


# ── Exploration noise ────────────────────────────────────────────────────

def test_exploration_noise_changes_output_in_training():
    """During training, exploration noise should produce different outputs
    for the same input across two calls."""
    B, L, D = 2, 4, 16
    torch.manual_seed(0)
    router = AdaptiveTopPMacroRouter(D, num_paths=3, exploration_noise=0.5)
    router.train()
    x = torch.randn(B, L, D)

    mask1, probs1 = router(x)
    mask2, probs2 = router(x)
    # With noise, the probabilities should differ
    assert not torch.allclose(probs1, probs2, atol=1e-6)


def test_exploration_noise_disabled_in_eval():
    """In eval mode, no noise is added — outputs are deterministic."""
    B, L, D = 2, 4, 16
    torch.manual_seed(0)
    router = AdaptiveTopPMacroRouter(D, num_paths=3, exploration_noise=0.5)
    router.eval()
    x = torch.randn(B, L, D)

    mask1, probs1 = router(x)
    mask2, probs2 = router(x)
    assert torch.allclose(probs1, probs2, atol=1e-6)


# ── Learnable thresholds ─────────────────────────────────────────────────

def test_learnable_thresholds_are_parameters():
    """With learnable_threshold=True, thresholds are nn.Parameters."""
    router = AdaptiveTopPMacroRouter(16, learnable_threshold=True)
    assert isinstance(router.base_threshold, nn.Parameter)
    assert isinstance(router.difficulty_scale, nn.Parameter)


def test_learnable_thresholds_fixed_by_default():
    """By default, thresholds are buffers (not learnable)."""
    router = AdaptiveTopPMacroRouter(16, learnable_threshold=False)
    assert not isinstance(router.base_threshold, nn.Parameter)
    assert not isinstance(router.difficulty_scale, nn.Parameter)


# ── Robustness: external difficulty clamping ─────────────────────────────

def test_external_difficulty_clamped():
    """External difficulty values outside [0, 1] are clamped."""
    B, L, D = 2, 3, 8
    router = AdaptiveTopPMacroRouter(D, num_paths=3, use_external_difficulty=True)
    x = torch.randn(B, L, D)

    # Values > 1 should be clamped to 1
    mask_clamped, _ = router(x, external_difficulty=torch.full((B, L), 5.0))
    mask_one, _ = router(x, external_difficulty=torch.ones(B, L))
    assert torch.equal(mask_clamped, mask_one)

    # Values < 0 should be clamped to 0
    mask_neg, _ = router(x, external_difficulty=torch.full((B, L), -5.0))
    mask_zero, _ = router(x, external_difficulty=torch.zeros(B, L))
    assert torch.equal(mask_neg, mask_zero)


# ── Diagnostics ──────────────────────────────────────────────────────────

def test_diagnostics_logging():
    """Diagnostics track average active paths per difficulty bin."""
    B, L, D = 4, 8, 16
    router = AdaptiveTopPMacroRouter(D, num_paths=3)
    router.enable_diagnostics(True)
    x = torch.randn(B, L, D)
    _ = router(x)

    diags = router.get_diagnostics()
    assert len(diags) == 1
    assert "bins" in diags[0]
    assert len(diags[0]["bins"]) == 4  # [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0)

    # Disable and verify clearing
    router.enable_diagnostics(False)
    assert len(router.get_diagnostics()) == 0


def test_diagnostics_via_decoder_layer():
    """DAPHDecoderLayerV2 exposes diagnostics through convenience methods."""
    from daph_exfusion.merge_toolkit import MemoryBankExFusionFFN
    hidden_size = 16
    ffn = MemoryBankExFusionFFN(
        hidden_size=hidden_size, intermediate_size=32,
        num_experts=2, activation="swiglu", bias=False,
    )
    layer = DAPHDecoderLayerV2(
        hidden_size=hidden_size,
        ffn_exfusion_factory=lambda: ffn,
        use_cheap_path=True,
    )
    layer.enable_diagnostics(True)
    x = torch.randn(2, 4, hidden_size)
    _ = layer(x)
    diags = layer.get_diagnostics()
    assert len(diags) == 1


# ── MLX enhancements ────────────────────────────────────────────────────

def test_mlx_cost_aware_routing():
    """MLX router with cost penalty activates cheap path more."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    from daph_exfusion.mlx_inference import MLXAdaptiveTopPMacroRouter

    B, L, D = 4, 8, 16
    mx.random.seed(42)
    x = mx.random.normal((B, L, D))

    router_no_cost = MLXAdaptiveTopPMacroRouter(D, path_costs=None)
    router_cost = MLXAdaptiveTopPMacroRouter(
        D, path_costs=(3.0, 2.0, 0.5), cost_penalty=1.0
    )
    # Copy weights for fair comparison
    router_cost.router.weight = router_no_cost.router.weight

    mask_no, _, _ = router_no_cost(x)
    mask_cost, _, _ = router_cost(x)

    cheap_no = int(np.array(mask_no[:, :, 2]).sum())
    cheap_cost = int(np.array(mask_cost[:, :, 2]).sum())
    assert cheap_cost >= cheap_no


def test_mlx_multi_signal_difficulty():
    """MLX multi-signal difficulty produces valid shapes."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    from daph_exfusion.mlx_inference import MLXAdaptiveTopPMacroRouter

    B, L, D = 2, 4, 16
    router = MLXAdaptiveTopPMacroRouter(D, multi_signal_difficulty=True)
    x = mx.random.normal((B, L, D))
    mask, probs, diff = router(x)
    assert mask.shape == (B, L, 3)
    assert probs.shape == (B, L, 3)
    assert diff.shape == (B, L, 1)
    # At least one path active
    counts = np.array(mask.sum(axis=-1))
    assert counts.min() >= 1


def test_mlx_decode_mode_reduces_paths():
    """decode_mode=True should activate fewer paths on average."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    from daph_exfusion.mlx_inference import MLXAdaptiveTopPMacroRouter

    B, L, D = 2, 1, 16  # Single token (decode scenario)
    mx.random.seed(123)
    x = mx.random.normal((B, L, D))

    router = MLXAdaptiveTopPMacroRouter(D)
    mask_prefill, _, _ = router(x, decode_mode=False)
    mask_decode, _, _ = router(x, decode_mode=True)

    avg_prefill = np.array(mask_prefill.sum(axis=-1)).mean()
    avg_decode = np.array(mask_decode.sum(axis=-1)).mean()
    # Decode mode should activate <= paths than prefill mode
    assert avg_decode <= avg_prefill


def test_mlx_learnable_threshold():
    """MLX router with learnable_threshold stores mx arrays."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import MLXAdaptiveTopPMacroRouter

    router = MLXAdaptiveTopPMacroRouter(16, learnable_threshold=True)
    assert hasattr(router, "base_threshold")
    assert hasattr(router, "difficulty_scale")
    # Should still produce valid output
    x = mx.random.normal((2, 4, 16))
    mask, probs, diff = router(x)
    assert mask.shape == (2, 4, 3)
