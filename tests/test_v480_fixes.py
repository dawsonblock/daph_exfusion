"""Tests for v4.8.0 systems constraints resolution.

1. PersistentHostMemoryBank + GIL-free copy_ offloading
2. Runtime SIMD-width probe for cooperative kernel portability
"""
import pytest
import torch
import torch.nn as nn
import numpy as np


# ── Fix 1: PersistentHostMemoryBank + GIL-free copy_ ─────────────────────

def test_persistent_host_memory_bank_creation():
    """PersistentHostMemoryBank pre-allocates CPU buffers for all expert params."""
    from daph_exfusion.orchestrator import PersistentHostMemoryBank

    class FakeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))
            self.bias = nn.Parameter(torch.randn(4))

    class FakeExFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([FakeExpert(), FakeExpert()])

    module = FakeExFusion()
    bank = PersistentHostMemoryBank(module)

    # Should have buffers for all params of all experts
    assert len(bank.buffers) == 4  # 2 experts * 2 params each
    assert "expert_0.weight" in bank.buffers
    assert "expert_0.bias" in bank.buffers
    assert "expert_1.weight" in bank.buffers
    assert "expert_1.bias" in bank.buffers

    # Buffers should be on CPU
    for buf in bank.buffers.values():
        assert buf.device.type == "cpu"
        assert buf.shape in [(4, 4), (4,)]


def test_persistent_host_memory_bank_get_buffer():
    """get_buffer returns the correct pre-allocated tensor."""
    from daph_exfusion.orchestrator import PersistentHostMemoryBank

    class FakeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

    class FakeExFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([FakeExpert(), FakeExpert()])

    module = FakeExFusion()
    bank = PersistentHostMemoryBank(module)

    buf0 = bank.get_buffer(0, "weight")
    buf1 = bank.get_buffer(1, "weight")
    assert buf0.shape == (4, 4)
    assert buf1.shape == (4, 4)
    # Buffers should be distinct
    assert buf0.data_ptr() != buf1.data_ptr()


def test_persistent_host_memory_bank_no_experts():
    """Bank creation on a module without experts produces empty buffers."""
    from daph_exfusion.orchestrator import PersistentHostMemoryBank

    class NoExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

    module = NoExperts()
    bank = PersistentHostMemoryBank(module)
    assert len(bank.buffers) == 0


def test_async_offloader_with_host_bank_offload():
    """Offloader with persistent bank uses copy_ (GIL-free) for offloading."""
    from daph_exfusion.orchestrator import AsyncMemoryOffloader

    class FakeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

    class FakeExFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([FakeExpert(), FakeExpert()])

    module = FakeExFusion()
    # Save original param data for comparison
    original_data = [p.data.clone() for e in module.experts for p in e.parameters()]

    offloader = AsyncMemoryOffloader(
        torch.device("cpu"), module=module, use_async=True
    )
    assert offloader.host_bank is not None

    offloader.offload_async(module)
    offloader.synchronize()

    # Params should now point to the pre-allocated CPU buffers
    for i, expert in enumerate(module.experts):
        for name, param in expert.named_parameters():
            buf = offloader.host_bank.get_buffer(i, name)
            # param.data should be the same tensor object as the buffer
            assert param.data.data_ptr() == buf.data_ptr()
            # Data should match original
            orig = original_data[i * len(list(expert.parameters())) +
                                list(dict(expert.named_parameters()).keys()).index(name)]
            assert torch.allclose(param.data, orig)


def test_async_offloader_with_host_bank_recall():
    """Offloader with persistent bank uses copy_ for recall."""
    from daph_exfusion.orchestrator import AsyncMemoryOffloader

    class FakeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

    class FakeExFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([FakeExpert(), FakeExpert()])

    module = FakeExFusion()
    original_data = [p.data.clone() for e in module.experts for p in e.parameters()]

    offloader = AsyncMemoryOffloader(
        torch.device("cpu"), module=module, use_async=True
    )

    # Offload then recall
    offloader.offload_async(module)
    offloader.synchronize()
    offloader.recall_async(module)
    offloader.synchronize()

    # After recall, data should match original (copy_ preserves values)
    for i, expert in enumerate(module.experts):
        for name, param in expert.named_parameters():
            orig = original_data[i * len(list(expert.parameters())) +
                                list(dict(expert.named_parameters()).keys()).index(name)]
            assert torch.allclose(param.data, orig)


def test_async_offloader_without_module_falls_back():
    """Offloader without module parameter falls back to .to() (backward compat)."""
    from daph_exfusion.orchestrator import AsyncMemoryOffloader

    class FakeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

    class FakeExFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([FakeExpert(), FakeExpert()])

    module = FakeExFusion()
    # No module parameter — should fall back to synchronous .to()
    offloader = AsyncMemoryOffloader(torch.device("cpu"), use_async=True)
    assert offloader.host_bank is None

    # Should still work (synchronous fallback)
    offloader.offload_async(module)
    offloader.synchronize()
    for expert in module.experts:
        for param in expert.parameters():
            assert param.device.type == "cpu"


def test_async_offloader_copy_preserves_values():
    """copy_ based offload/recall preserves parameter values exactly."""
    from daph_exfusion.orchestrator import AsyncMemoryOffloader

    class FakeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(8, 8) * 10)
            self.bias = nn.Parameter(torch.randn(8) * 5)

    class FakeExFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([FakeExpert(), FakeExpert(), FakeExpert()])

    module = FakeExFusion()
    # Deep copy original state
    import copy
    original = copy.deepcopy(module.state_dict())

    offloader = AsyncMemoryOffloader(
        torch.device("cpu"), module=module, use_async=True
    )

    # Offload all experts
    offloader.offload_async(module)
    offloader.synchronize()

    # Recall all experts
    offloader.recall_async(module)
    offloader.synchronize()

    # Verify values are preserved
    for key, val in module.state_dict().items():
        assert torch.allclose(val, original[key], atol=1e-6), f"Mismatch in {key}"


# ── Fix 2: Runtime SIMD-width probe ──────────────────────────────────────

def test_probe_hardware_simd_width():
    """probe_hardware_simd_width returns a positive integer."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import probe_hardware_simd_width

    width = probe_hardware_simd_width()
    assert isinstance(width, int)
    assert width > 0
    # On Apple Silicon, this should be 32
    assert width == 32  # Apple GPU standard


def test_probe_hardware_simd_width_cached():
    """The probe result is cached — second call returns same value."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import probe_hardware_simd_width, _PROBED_SIMD_WIDTH

    w1 = probe_hardware_simd_width()
    w2 = probe_hardware_simd_width()
    assert w1 == w2
    assert _PROBED_SIMD_WIDTH is not None


def test_compute_cooperative_tiling_uses_probed_width():
    """_compute_cooperative_tiling uses the probed SIMD width."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import (
        _compute_cooperative_tiling,
        probe_hardware_simd_width,
    )

    W_probed = probe_hardware_simd_width()
    W, S, channels_per_tg, tg_size = _compute_cooperative_tiling(256)
    assert W == W_probed
    assert S == (256 + W_probed - 1) // W_probed


def test_cooperative_kernel_correctness_after_probe():
    """The cooperative kernel still produces correct output after the
    SIMD-width probe is integrated."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import (
        mamba_selective_scan,
        mamba_selective_scan_reference,
        probe_hardware_simd_width,
    )

    # Ensure probe has run
    W = probe_hardware_simd_width()
    assert W == 32

    B, L, D, d_state = 2, 8, 16, 256
    delta = mx.random.normal((B, L, D))
    A_log = mx.random.normal((D,))
    Bv = mx.random.normal((B, L, d_state))
    C = mx.random.normal((B, L, d_state))
    Dv = mx.random.normal((D,))
    x = mx.random.normal((B, L, D))

    y_ref = np.array(mamba_selective_scan_reference(delta, A_log, Bv, C, Dv, x))
    y_ker, h_last = mamba_selective_scan(delta, A_log, Bv, C, Dv, x)

    assert np.array(y_ker).shape == (B, L, D)
    assert np.array(h_last).shape == (B, D, d_state)
    assert np.allclose(y_ref, np.array(y_ker), rtol=1e-3, atol=1e-4)
    assert not np.any(np.isnan(np.array(h_last)))
