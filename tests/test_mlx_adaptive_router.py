"""Tests for MLX adaptive top-p router and stateful decoder."""
import pytest
import numpy as np

pytest.importorskip("mlx.core")

import mlx.core as mx
from daph_exfusion.mlx_inference import (
    MLXAdaptiveTopPMacroRouter,
    KVCache,
    SSMState,
    MLXStatefulDAPHDecoderLayer,
)


def test_adaptive_router_shape():
    B, L, D = 2, 5, 16
    router = MLXAdaptiveTopPMacroRouter(D, num_paths=3)
    x = mx.random.normal((B, L, D))

    mask, probs, diff = router(x)
    assert mask.shape == (B, L, 3)
    assert probs.shape == (B, L, 3)
    assert diff.shape == (B, L, 1)


def test_adaptive_router_top1_minimum():
    """Every token must have at least one active path."""
    B, L, D = 4, 8, 16
    router = MLXAdaptiveTopPMacroRouter(D, num_paths=3)
    x = mx.random.normal((B, L, D))

    mask, _, _ = router(x)
    # Sum over paths axis
    path_counts = mask.sum(axis=-1)  # (B, L)
    assert path_counts.min() >= 1


def test_adaptive_router_probability_sum():
    """Path probabilities must sum to 1."""
    B, L, D = 2, 3, 8
    router = MLXAdaptiveTopPMacroRouter(D, num_paths=3)
    x = mx.random.normal((B, L, D))

    _, probs, _ = router(x)
    sums = probs.sum(axis=-1)
    assert np.allclose(np.array(sums), 1.0, atol=1e-5)


def test_kv_cache_update():
    cache = KVCache()
    bsz, heads, seq, dim = 2, 4, 1, 8
    k = mx.random.normal((bsz, heads, seq, dim))
    v = mx.random.normal((bsz, heads, seq, dim))

    k_out, v_out = cache.update(k, v)
    assert k_out.shape == (bsz, heads, seq, dim)
    assert v_out.shape == (bsz, heads, seq, dim)

    # Second update should concatenate
    k2 = mx.random.normal((bsz, heads, 2, dim))
    v2 = mx.random.normal((bsz, heads, 2, dim))
    k_out2, v_out2 = cache.update(k2, v2)
    assert k_out2.shape == (bsz, heads, seq + 2, dim)


def test_ssm_state_update():
    bsz, d_model, d_state = 2, 16, 16
    state = SSMState(bsz, d_model, d_state)
    assert np.array(state.h).shape == (bsz, d_model, d_state)

    new_h = mx.random.normal((bsz, d_model, d_state))
    state.update(new_h)
    assert np.allclose(np.array(state.h), np.array(new_h))


def test_stateful_decoder_forward():
    B, L, D = 2, 4, 16
    layer = MLXStatefulDAPHDecoderLayer(
        hidden_size=D,
        intermediate_size=D * 2,
        num_heads=4,
    )
    x = mx.random.normal((B, L, D))
    out = layer(x)
    assert out.shape == (B, L, D)


def test_stateful_decoder_with_cache():
    B, L, D = 1, 2, 8
    layer = MLXStatefulDAPHDecoderLayer(
        hidden_size=D,
        intermediate_size=D * 2,
        num_heads=2,
    )
    x = mx.random.normal((B, L, D))
    cache = KVCache()
    ssm = SSMState(B, D)

    out = layer(x, kv_cache=cache, ssm_state=ssm)
    assert out.shape == (B, L, D)
    assert cache.keys is not None
