"""Tests for v4.7.0 systems constraints resolution.

1. AsyncMemoryOffloader: thread-pool pipelined offloading for MPS/CPU
2. Dynamic shared memory tiling in cooperative Metal kernel
"""
import pytest
import torch
import torch.nn as nn
import numpy as np


# ── Issue 1: AsyncMemoryOffloader ────────────────────────────────────────

def test_async_offloader_creation_cpu():
    """AsyncMemoryOffloader can be created for CPU device."""
    from daph_exfusion.orchestrator import AsyncMemoryOffloader
    offloader = AsyncMemoryOffloader(torch.device("cpu"), use_async=True)
    assert offloader.device.type == "cpu"
    assert offloader.use_async is True
    # On CPU, no CUDA stream should be created
    assert offloader.cuda_stream is None
    assert offloader.thread_future is None


def test_async_offloader_sync_mode():
    """In sync mode (use_async=False), transfers are synchronous."""
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
    offloader = AsyncMemoryOffloader(torch.device("cpu"), use_async=False)

    # Synchronous offload — should be a no-op on CPU
    offloader.offload_async(module)
    for expert in module.experts:
        for param in expert.parameters():
            assert param.device.type == "cpu"

    # Synchronous recall
    offloader.recall_async(module)
    for expert in module.experts:
        for param in expert.parameters():
            assert param.device.type == "cpu"


def test_async_offloader_thread_pool_offload():
    """On MPS/CPU, async offload uses the thread pool."""
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
    offloader = AsyncMemoryOffloader(torch.device("cpu"), use_async=True)

    # Async offload — on CPU this is a no-op but should use thread pool
    offloader.offload_async(module)
    assert offloader.thread_future is not None

    # Synchronize — should block until thread completes
    offloader.synchronize()
    assert offloader.thread_future is None

    # Verify params are on CPU
    for expert in module.experts:
        for param in expert.parameters():
            assert param.device.type == "cpu"


def test_async_offloader_thread_pool_recall():
    """On MPS/CPU, async recall uses the thread pool."""
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
    offloader = AsyncMemoryOffloader(torch.device("cpu"), use_async=True)

    offloader.recall_async(module)
    assert offloader.thread_future is not None
    offloader.synchronize()
    assert offloader.thread_future is None


def test_async_offloader_synchronize_noop():
    """synchronize() with no pending transfer is a no-op."""
    from daph_exfusion.orchestrator import AsyncMemoryOffloader
    offloader = AsyncMemoryOffloader(torch.device("cpu"), use_async=True)
    # Should not raise
    offloader.synchronize()


def test_async_offloader_no_experts_attribute():
    """offload_async/recall_async on a module without 'experts' is a no-op."""
    from daph_exfusion.orchestrator import AsyncMemoryOffloader

    class NoExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

    module = NoExperts()
    offloader = AsyncMemoryOffloader(torch.device("cpu"), use_async=True)
    offloader.offload_async(module)
    offloader.recall_async(module)
    offloader.synchronize()
    # Should not have spawned a thread
    assert offloader.thread_future is None


def test_async_offloader_chained_operations():
    """Multiple offload/recall operations in sequence work correctly."""
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
    offloader = AsyncMemoryOffloader(torch.device("cpu"), use_async=True)

    # Chain: offload → sync → recall → sync
    offloader.offload_async(module)
    offloader.synchronize()
    offloader.recall_async(module)
    offloader.synchronize()

    for expert in module.experts:
        for param in expert.parameters():
            assert param.device.type == "cpu"


def test_transfer_pool_singleton():
    """The global transfer pool is lazily created and reused."""
    from daph_exfusion.orchestrator import _get_transfer_pool
    pool1 = _get_transfer_pool()
    pool2 = _get_transfer_pool()
    assert pool1 is pool2


# ── Issue 2: Dynamic shared memory tiling ────────────────────────────────

def test_compute_cooperative_tiling_small_d_state():
    """Small d_state (128) should use max 8 channels per TG."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import _compute_cooperative_tiling
    W, S, channels_per_tg, tg_size = _compute_cooperative_tiling(128)
    assert W == 32
    assert S == 4  # ceil(128/32) = 4
    assert channels_per_tg == 8  # 4096//128=32, capped at 8
    assert tg_size == 256  # 8 * 32


def test_compute_cooperative_tiling_medium_d_state():
    """Medium d_state (512) should use 8 channels (16KB shared mem)."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import _compute_cooperative_tiling
    W, S, channels_per_tg, tg_size = _compute_cooperative_tiling(512)
    assert W == 32
    assert S == 16  # ceil(512/32) = 16
    assert channels_per_tg == 8  # 8192//512=16, capped at 8
    assert tg_size == 256  # 8 * 32
    # Verify shared memory is exactly 16KB
    shared_bytes = channels_per_tg * 512 * 4
    assert shared_bytes == 16384


def test_compute_cooperative_tiling_large_d_state():
    """Large d_state (1024) should use 8 channels (32KB shared mem)."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import _compute_cooperative_tiling
    W, S, channels_per_tg, tg_size = _compute_cooperative_tiling(1024)
    assert W == 32
    assert S == 32  # ceil(1024/32) = 32
    assert channels_per_tg == 8  # 8192//1024=8
    assert tg_size == 256  # 8 * 32
    shared_bytes = channels_per_tg * 1024 * 4
    assert shared_bytes == 32768


def test_compute_cooperative_tiling_very_large_d_state():
    """Very large d_state (2048) should reduce to 4 channels."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import _compute_cooperative_tiling
    W, S, channels_per_tg, tg_size = _compute_cooperative_tiling(2048)
    assert W == 32
    assert S == 64  # ceil(2048/32) = 64
    assert channels_per_tg == 4  # 8192//2048=4
    assert tg_size == 128  # 4 * 32
    shared_bytes = channels_per_tg * 2048 * 4
    assert shared_bytes == 32768


def test_compute_cooperative_tiling_max_d_state():
    """Maximum supported d_state (8192) should use 1 channel."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import _compute_cooperative_tiling
    W, S, channels_per_tg, tg_size = _compute_cooperative_tiling(8192)
    assert W == 32
    assert S == 256  # ceil(8192/32) = 256
    assert channels_per_tg == 1  # 8192//8192=1
    assert tg_size == 32  # 1 * 32
    shared_bytes = channels_per_tg * 8192 * 4
    assert shared_bytes == 32768


def test_compute_cooperative_tiling_always_under_32kb():
    """Shared memory must never exceed 32KB for any valid d_state."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import _compute_cooperative_tiling
    # Test all d_state values that trigger the cooperative kernel (> 128)
    for d_state in range(129, 8193, 32):
        W, S, channels_per_tg, tg_size = _compute_cooperative_tiling(d_state)
        shared_bytes = channels_per_tg * d_state * 4
        assert shared_bytes <= 32768, (
            f"d_state={d_state}: shared memory {shared_bytes} > 32768"
        )
        # tg_size must be a multiple of W
        assert tg_size % W == 0
        assert tg_size > 0


def test_compute_cooperative_tiling_exceeds_max():
    """d_state > 8192 should raise ValueError."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import _compute_cooperative_tiling
    with pytest.raises(ValueError, match="d_state"):
        _compute_cooperative_tiling(8193)


def test_cooperative_kernel_correctness_d256():
    """The cooperative kernel with dynamic tiling produces correct output
    for d_state=256 (8 channels/TG, 8KB shared mem)."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import (
        mamba_selective_scan,
        mamba_selective_scan_reference,
    )

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


def test_cooperative_kernel_correctness_d512():
    """The cooperative kernel with dynamic tiling produces correct output
    for d_state=512 (8 channels/TG, 16KB shared mem)."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import (
        mamba_selective_scan,
        mamba_selective_scan_reference,
    )

    B, L, D, d_state = 1, 4, 8, 512
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


def test_cooperative_kernel_correctness_d1024():
    """The cooperative kernel with dynamic tiling produces correct output
    for d_state=1024 (4 channels/TG, 16KB shared mem).

    This exercises the dynamic channel reduction — without it, the
    shared memory would be 32KB, potentially exceeding L1 capacity.
    """
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import (
        mamba_selective_scan,
        mamba_selective_scan_reference,
    )

    B, L, D, d_state = 1, 4, 8, 1024
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
    # Use looser tolerance for large d_state due to accumulation
    assert np.allclose(y_ref, np.array(y_ker), rtol=5e-3, atol=1e-3)
    assert not np.any(np.isnan(np.array(h_last)))
