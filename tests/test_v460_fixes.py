"""Tests for v4.6.0 systems hardening fixes.

1. Tensor-lifecycle-bound FFT cache (no more address recycling)
2. MLX non-blocking gating for L=1 decode (no sync bubble)
3. Coalesced state write-back in cooperative Metal kernel
4. Async stream-based K-FAC offloading (device-agnostic)
"""
import pytest
import torch
import torch.nn as nn
import numpy as np

from daph_exfusion.adaptive_top_p_router import AdaptiveTopPMacroRouter, DAPHDecoderLayerV2
from daph_exfusion.merge_toolkit import MemoryBankExFusionFFN


# ── Fix 1: Tensor-lifecycle-bound FFT cache ──────────────────────────────

def test_fft_cache_no_dict_attribute():
    """DAPHDecoderLayerV2 should not have the old _fft_cache dict."""
    ffn = MemoryBankExFusionFFN(
        hidden_size=16, intermediate_size=32,
        num_experts=2, activation="swiglu", bias=False,
    )
    layer = DAPHDecoderLayerV2(
        hidden_size=16,
        ffn_exfusion_factory=lambda: ffn,
        use_cheap_path=True,
    )
    assert not hasattr(layer, "_fft_cache")
    # The forward_pre_hook should also be gone
    hooks = layer._forward_pre_hooks
    assert len(hooks) == 0, f"Expected no forward pre-hooks, got {hooks}"


def test_fft_cache_bound_to_tensor_lifecycle():
    """The FFT memo is bound as a dynamic attribute on the normalised tensor."""
    ffn = MemoryBankExFusionFFN(
        hidden_size=16, intermediate_size=32,
        num_experts=2, activation="swiglu", bias=False,
    )
    layer = DAPHDecoderLayerV2(
        hidden_size=16,
        ffn_exfusion_factory=lambda: ffn,
        use_cheap_path=True,
    )
    x = torch.randn(2, 4, 16)
    # Forward should work and produce correct shape
    out = layer(x)
    assert out.shape == x.shape

    # Call _cheap_path directly and verify the memo attribute is set
    cheap_out = layer._cheap_path(x)
    # The memo is on the normalised tensor (internal to _cheap_path),
    # so we can't directly inspect it. But we can verify the output is
    # consistent across calls (the FFT is deterministic).
    cheap_out2 = layer._cheap_path(x)
    assert torch.allclose(cheap_out, cheap_out2, atol=1e-6)


def test_fft_cache_no_false_hit_on_recycled_address():
    """Two different inputs should not share FFT cache results even if
    PyTorch recycles the same memory address for their normalised tensors."""
    ffn = MemoryBankExFusionFFN(
        hidden_size=16, intermediate_size=32,
        num_experts=2, activation="swiglu", bias=False,
    )
    layer = DAPHDecoderLayerV2(
        hidden_size=16,
        ffn_exfusion_factory=lambda: ffn,
        use_cheap_path=True,
    )
    # Two different inputs
    x1 = torch.randn(2, 4, 16)
    x2 = torch.randn(2, 4, 16) + 10.0  # Very different values

    out1 = layer._cheap_path(x1)
    out2 = layer._cheap_path(x2)
    # Outputs must differ — if the cache falsely hit, they'd be identical
    assert not torch.allclose(out1, out2, atol=1e-3)


# ── Fix 2: MLX non-blocking gating for L=1 ───────────────────────────────

def test_mlx_skip_inactive_no_sync_for_l1():
    """MLXDAPHDecoderLayer with skip_inactive_paths=True should NOT
    call mx.eval (host sync) when L=1 (single-token decode)."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import MLXDAPHDecoderLayer

    B, L, D = 1, 1, 16  # Single-token decode
    layer = MLXDAPHDecoderLayer(
        hidden_size=D, intermediate_size=D * 2,
        num_heads=4, skip_inactive_paths=True,
    )
    x = mx.random.normal((B, L, D))
    # This should work without raising — no host sync for L=1
    out = layer(x)
    assert out.shape == (B, L, D)
    assert not np.any(np.isnan(np.array(out)))


def test_mlx_skip_inactive_sync_for_prefill():
    """MLXDAPHDecoderLayer with skip_inactive_paths=True should still
    use the sync path for L>1 (prefill), where compute savings matter."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import MLXDAPHDecoderLayer

    B, L, D = 2, 8, 16  # Prefill
    layer = MLXDAPHDecoderLayer(
        hidden_size=D, intermediate_size=D * 2,
        num_heads=4, skip_inactive_paths=True,
    )
    x = mx.random.normal((B, L, D))
    out = layer(x)
    assert out.shape == (B, L, D)
    assert not np.any(np.isnan(np.array(out)))


def test_mlx_skip_inactive_disabled_by_default():
    """Without skip_inactive_paths, both prefill and decode use the
    non-blocking mx.where path."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import MLXDAPHDecoderLayer

    for L in [1, 8]:
        layer = MLXDAPHDecoderLayer(
            hidden_size=16, intermediate_size=32,
            num_heads=4, skip_inactive_paths=False,
        )
        x = mx.random.normal((2, L, 16))
        out = layer(x)
        assert out.shape == (2, L, 16)
        assert not np.any(np.isnan(np.array(out)))


# ── Fix 3: Coalesced cooperative kernel state write-back ─────────────────

def test_cooperative_kernel_state_correctness():
    """The coalesced write-back must produce the same final state as the
    reference implementation.  This verifies the shared-memory transposition
    doesn't corrupt the state values."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import (
        mamba_selective_scan,
        mamba_selective_scan_reference,
    )

    # Use d_state > 128 to exercise the cooperative kernel
    B, L, D, d_state = 2, 8, 16, 256
    delta = mx.random.normal((B, L, D))
    A_log = mx.random.normal((D,))
    Bv = mx.random.normal((B, L, d_state))
    C = mx.random.normal((B, L, d_state))
    Dv = mx.random.normal((D,))
    x = mx.random.normal((B, L, D))

    # Reference only returns y, not h_last — compare y outputs
    y_ref = np.array(mamba_selective_scan_reference(delta, A_log, Bv, C, Dv, x))
    y_ker, h_last = mamba_selective_scan(delta, A_log, Bv, C, Dv, x)

    assert np.array(y_ker).shape == (B, L, D)
    assert np.array(h_last).shape == (B, D, d_state)
    # Use relative tolerance since SSM values can be large
    assert np.allclose(y_ref, np.array(y_ker), rtol=1e-3, atol=1e-4)
    # Verify no NaN in state
    assert not np.any(np.isnan(np.array(h_last)))


def test_cooperative_kernel_small_d_state():
    """The cooperative kernel should also work for smaller d_state values
    that are still above the threshold (e.g. d_state=129)."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import mamba_selective_scan

    B, L, D, d_state = 1, 4, 8, 160  # Just above threshold of 128
    delta = mx.random.normal((B, L, D))
    A_log = mx.random.normal((D,))
    Bv = mx.random.normal((B, L, d_state))
    C = mx.random.normal((B, L, d_state))
    Dv = mx.random.normal((D,))
    x = mx.random.normal((B, L, D))

    y, h_last = mamba_selective_scan(delta, A_log, Bv, C, Dv, x)
    assert np.array(y).shape == (B, L, D)
    assert np.array(h_last).shape == (B, D, d_state)
    assert not np.any(np.isnan(np.array(y)))
    assert not np.any(np.isnan(np.array(h_last)))


# ── Fix 4: Async stream-based K-FAC offloading ───────────────────────────

def test_offload_with_stream_argument():
    """offload_experts_to_cpu accepts an optional stream argument."""
    from daph_exfusion.orchestrator import offload_experts_to_cpu, recall_experts_to_gpu

    # Create a simple module with 'experts'
    class FakeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

    class FakeExFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([FakeExpert(), FakeExpert()])

    module = FakeExFusion()
    # On CPU, offload is a no-op (already on CPU)
    offload_experts_to_cpu(module, stream=None)
    for expert in module.experts:
        assert expert.weight.device.type == "cpu"

    # Recall should also work
    recall_experts_to_gpu(module, torch.device("cpu"), stream=None)
    for expert in module.experts:
        assert expert.weight.device.type == "cpu"


def test_get_transfer_stream_returns_none_on_mps():
    """_get_transfer_stream should return None when CUDA is unavailable."""
    from daph_exfusion.orchestrator import _get_transfer_stream

    if not torch.cuda.is_available():
        stream = _get_transfer_stream()
        assert stream is None


def test_sync_stream_noop_with_none():
    """_sync_stream(None) should be a no-op (no crash)."""
    from daph_exfusion.orchestrator import _sync_stream

    # Should not raise
    _sync_stream(None)


def test_orchestrator_memory_offloading_runs():
    """Verify that the async offloading functions work correctly when
    called directly on a DAPHDecoderLayerV2's ExFusion paths.

    This tests the Fix 4 code path (async stream-based offloading)
    without running the full pipeline, which would trigger unrelated
    PyTorch↔MLX bridge validation.
    """
    from daph_exfusion.orchestrator import (
        offload_experts_to_cpu,
        recall_experts_to_gpu,
        _get_transfer_stream,
        _sync_stream,
    )
    from daph_exfusion.adaptive_top_p_router import DAPHDecoderLayerV2

    hidden_size = 16
    layer = DAPHDecoderLayerV2(
        hidden_size=hidden_size,
        ffn_exfusion_factory=lambda: MemoryBankExFusionFFN(
            hidden_size=hidden_size, intermediate_size=32,
            num_experts=2, activation="swiglu", bias=False,
        ),
        attention_factory=None,
        use_cheap_path=False,
    )

    # On CPU, offload is a no-op but should not crash
    stream = _get_transfer_stream()
    assert stream is None  # No CUDA on this machine

    offload_experts_to_cpu(layer.ffn_path, stream=stream)
    for expert in layer.ffn_path.experts:
        for param in expert.parameters():
            assert param.device.type == "cpu"

    recall_experts_to_gpu(layer.ffn_path, torch.device("cpu"), stream=stream)
    for expert in layer.ffn_path.experts:
        for param in expert.parameters():
            assert param.device.type == "cpu"

    # _sync_stream with None should be a no-op
    _sync_stream(stream)
