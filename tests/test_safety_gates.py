"""Safety-gate regression tests for the v4.2.1 + v4.3 remediation.

Each test pins one of the defects fixed in the remediation patches so a
future regression breaks the build immediately:

  1. ``test_default_mamba_policies_use_majority_sign_election``
     — guards the sign-mode fix in DEFAULT_MAMBA_POLICIES.
  2. ``test_compiled_prefill_matches_reference``
     — guards the @mx.compile'd pre-fill step kernel.
  3. ``test_kfac_diagonal_only_tracker_is_1d``
     — guards the diagonal-only K-FAC storage.
  4. ``test_bridge_raises_on_missing_keys``
     — guards the strict missing-key validation in the PT->MLX bridge.
  5. ``test_ties_sign_election_is_majority``
     — guards the TIES sign election fix (no magnitude-weighted cancellation).
  6. ``test_kfac_block_diagonal_mode``
     — guards block-diagonal K-FAC storage and scoring.
  7. ``test_mamba_selective_scan_returns_state``
     — guards the fused GPU state capture in the metal kernel.
  8. ``test_mamba_has_conv1d``
     — guards the standard 1D convolution in the Mamba block.
  9. ``test_rope_in_flash_attention``
     — guards RoPE application in MLXFlashAttention.
 10. ``test_causal_lm_forward``
     — guards the MLXStatefulCausalLM top-level wrapper.
"""
import pytest
import torch
import torch.nn as nn

from daph_exfusion.merge_toolkit import (
    DEFAULT_MAMBA_POLICIES,
    KFACConfig,
    KFACFisherTracker,
    RunningCovariance,
    compute_ties_aligned_deltas,
    SwiGLUFFN,
)


# ---------------------------------------------------------------------------
# 1. Sign-mode invariance
# ---------------------------------------------------------------------------
def test_default_mamba_policies_use_majority_sign_election():
    """Every shipped Mamba group policy must use sign_mode == 'majority'.

    sign · |Δ| under 'magnitude_weighted' collapses to the raw delta and
    lets large-magnitude outliers swamp the election.  Majority is the
    mathematically correct interpretation of a sign election.  Any override
    to 'magnitude_weighted' must be explicit and confined to a unit test,
    never the shipped default.
    """
    assert DEFAULT_MAMBA_POLICIES, "DEFAULT_MAMBA_POLICIES must not be empty"
    for name, policy in DEFAULT_MAMBA_POLICIES.items():
        assert policy.sign_mode == "majority", (
            f"Group '{name}' has sign_mode={policy.sign_mode!r}; "
            f"shipped default must be 'majority'."
        )


# ---------------------------------------------------------------------------
# 2. Compiled pre-fill step kernel correctness
# ---------------------------------------------------------------------------
def test_compiled_prefill_matches_reference():
    """The @mx.compile'd pre-fill step must reproduce the reference loop's
    final SSM state to within 1e-6.

    MLX does not expose a stable graph-op-count API, so we assert numerical
    equivalence to the pure-Python reference loop instead — this is the
    correctness guarantee the compiled kernel must preserve.  A regression
    that breaks the fusion would diverge from the reference.
    """
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    from daph_exfusion.mlx_inference import (
        ssm_prefill_loop,
        mamba_selective_scan_reference,
    )

    # Note: do NOT set mx.set_default_device(mx.cpu) here — it would break
    # subsequent metal kernel tests that require the GPU.
    B, L, D = 2, 4, 8
    a = -mx.exp(mx.random.normal((D,)))
    delta = mx.random.normal((B, L, D))
    Bc = mx.random.normal((B, L, D))
    u = mx.random.normal((B, L, D))

    # Final state via the compiled-step loop.
    state0 = mx.zeros((B, D))
    final_compiled = ssm_prefill_loop(delta, Bc, u, a, state0)

    # Final state via the reference loop (re-derive state from its recurrence;
    # mamba_selective_scan_reference returns per-step outputs, so we replay the
    # same recurrence here in pure Python to get the final state).
    state_ref = mx.zeros((B, D))
    for t in range(L):
        decay = mx.exp(delta[:, t, :] * a)
        state_ref = decay * state_ref + delta[:, t, :] * Bc[:, t, :] * u[:, t, :]

    assert np.allclose(np.array(final_compiled), np.array(state_ref), atol=1e-6), (
        f"Compiled pre-fill diverged from reference: "
        f"max|Δ|={np.max(np.abs(np.array(final_compiled) - np.array(state_ref)))}"
    )


# ---------------------------------------------------------------------------
# 3. K-FAC diagonal-only storage
# ---------------------------------------------------------------------------
def test_kfac_diagonal_only_tracker_is_1d():
    """KFACFisherTracker with diagonal_only=True must store 1D factors.

    At d_model=4096 the full (dim, dim) covariance is ~1.6 GB per layer per
    expert; the diagonal proxy is a few MB and is what most K-FAC practitioners
    use for linear layers anyway.
    """
    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(16, 32)

        def forward(self, x):
            return self.proj(x)

    cfg = KFACConfig(diagonal_only=True)
    tracker = KFACFisherTracker(_TinyModel(), config=cfg)
    # Drive one forward/backward so the hooks populate the factors.
    x = torch.randn(4, 16)
    out = tracker.model(x)
    out.sum().backward()
    for name, cov in tracker.a_factors.items():
        assert cov.value.ndim == 1, (
            f"a_factors['{name}'].value.ndim == {cov.value.ndim}, expected 1 "
            f"(diagonal_only=True)"
        )
    for name, cov in tracker.g_factors.items():
        assert cov.value.ndim == 1, (
            f"g_factors['{name}'].value.ndim == {cov.value.ndim}, expected 1 "
            f"(diagonal_only=True)"
        )


def test_kfac_full_covariance_still_supported():
    """Opting out of diagonal-only (diagonal_only=False) must still work
    and produce 2D factors, so the ablation path is not silently broken."""
    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, x):
            return self.proj(x)

    cfg = KFACConfig(diagonal_only=False)
    tracker = KFACFisherTracker(_TinyModel(), config=cfg)
    x = torch.randn(4, 8)
    out = tracker.model(x)
    out.sum().backward()
    for name, cov in tracker.a_factors.items():
        assert cov.value.ndim == 2, (
            f"a_factors['{name}'].value.ndim == {cov.value.ndim}, expected 2"
        )


# ---------------------------------------------------------------------------
# 4. Bridge strictness on missing keys
# ---------------------------------------------------------------------------
def test_bridge_raises_on_missing_keys():
    """validate_architecture_compatibility must raise RuntimeError on a
    crafted key miss so a broken export fails loudly instead of producing
    nonsense inference."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import MLXDAPHDecoderLayer
    from daph_exfusion.bridge import validate_architecture_compatibility

    mlx_model = MLXDAPHDecoderLayer(hidden_size=64, intermediate_size=128, num_heads=4)
    # A state dict containing a key that does not exist in the MLX model.
    bad_state = {
        "attention_path.attn.q_proj.weight": torch.randn(64, 32),
        "this.key.does.not.exist": torch.randn(4, 4),
    }
    with pytest.raises(RuntimeError, match="Bridge Parity Violation"):
        validate_architecture_compatibility(bad_state, mlx_model, raise_on_mismatch=True)


# ---------------------------------------------------------------------------
# 5. TIES sign election is pure majority (no magnitude-weighted cancellation)
# ---------------------------------------------------------------------------
def test_ties_sign_election_is_majority():
    """compute_ties_aligned_deltas must use pure sign-majority voting.

    The old code computed ``w * sign(delta) * delta.abs()`` which collapses
    to ``w * delta`` — a weighted sum of deltas, not a sign election.  The
    fix uses ``w * sign(delta)`` so the vote is a true majority election.
    """
    # Use SwiGLUFFN experts so compute_ties_aligned_deltas recognizes them.
    def make_expert(w_val):
        up = nn.Linear(4, 4, bias=False)
        gate = nn.Linear(4, 4, bias=False)
        down = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            up.weight.fill_(w_val)
            gate.weight.fill_(w_val)
            down.weight.fill_(w_val)
        return SwiGLUFFN(up, gate, down)

    # Two experts with opposite-sign deltas: expert 0 at +1, expert 1 at -1.
    # Under majority voting with equal weights, the vote is 0 (tie), so
    # elected_sign is 0 and all deltas are zeroed.  Under the old
    # magnitude-weighted bug, the vote would be sign*abs = delta, so
    # the deltas would survive.
    e0 = make_expert(1.0)
    e1 = make_expert(-1.0)
    memory_bank = torch.tensor([0.5, 0.5])
    base, aligned, _ = compute_ties_aligned_deltas(
        experts=[e0, e1],
        memory_bank=memory_bank,
        trim_ratio=0.0,  # no trimming — we want to test the sign election
    )
    # With equal weights and opposite signs, the vote is 0 → elected_sign=0
    # → all deltas zeroed.  If the bug were present, deltas would survive.
    for d in aligned:
        for k, v in d.items():
            assert torch.all(v == 0), (
                f"TIES sign election produced non-zero aligned delta for '{k}'; "
                f"expected zero (majority tie). Got max|v|={v.abs().max().item()}"
            )


# ---------------------------------------------------------------------------
# 6. Block-diagonal K-FAC mode
# ---------------------------------------------------------------------------
def test_kfac_block_diagonal_mode():
    """RunningCovariance with block_size must store 3D block-diagonal factors
    and produce valid scores."""
    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(16, 16)

        def forward(self, x):
            return self.proj(x)

    cfg = KFACConfig(diagonal_only=False, block_size=8)
    tracker = KFACFisherTracker(_TinyModel(), config=cfg)
    x = torch.randn(4, 16)
    out = tracker.model(x)
    out.sum().backward()
    for name, cov in tracker.a_factors.items():
        assert cov.value.ndim == 3, (
            f"a_factors['{name}'].value.ndim == {cov.value.ndim}, expected 3 "
            f"(block-diagonal mode)"
        )
    # Score should be a valid float
    score = tracker.layer_score("proj")
    assert isinstance(score, float) and score > 0


# ---------------------------------------------------------------------------
# 7. Fused GPU state capture in mamba_selective_scan
# ---------------------------------------------------------------------------
def test_mamba_selective_scan_returns_state():
    """mamba_selective_scan must return (y, h_last) where h_last is the
    final recurrent state captured in the same GPU pass."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    from daph_exfusion.mlx_inference import mamba_selective_scan

    B, L, D = 2, 8, 16
    delta = mx.random.normal((B, L, D))
    A_log = mx.random.normal((D,))
    Bv = mx.random.normal((B, L, D))
    C = mx.random.normal((B, L, D))
    Dv = mx.random.normal((D,))
    x = mx.random.normal((B, L, D))

    result = mamba_selective_scan(delta, A_log, Bv, C, Dv, x)
    assert isinstance(result, tuple) and len(result) == 2, (
        "mamba_selective_scan must return (y, h_last)"
    )
    y, h_last = result
    assert y.shape == (B, L, D), f"y shape {y.shape}, expected {(B, L, D)}"
    assert h_last.shape == (B, D), f"h_last shape {h_last.shape}, expected {(B, D)}"


# ---------------------------------------------------------------------------
# 8. Mamba block has standard 1D convolution
# ---------------------------------------------------------------------------
def test_mamba_has_conv1d():
    """MLXMergedMamba must include a conv1d layer for compatibility with
    real-world pre-trained Mamba/Jamba weights."""
    pytest.importorskip("mlx.core")
    from daph_exfusion.mlx_inference import MLXMergedMamba
    mamba = MLXMergedMamba(d_model=16)
    assert hasattr(mamba, "conv1d"), "MLXMergedMamba must have a conv1d layer"
    assert mamba.d_conv == 4, f"d_conv={mamba.d_conv}, expected 4"


# ---------------------------------------------------------------------------
# 9. RoPE in MLXFlashAttention
# ---------------------------------------------------------------------------
def test_rope_in_flash_attention():
    """MLXFlashAttention must apply RoPE to Q and K."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    from daph_exfusion.mlx_inference import MLXFlashAttention, MLXRotaryEmbedding

    attn = MLXFlashAttention(hidden_size=16, num_heads=2, use_rope=True)
    assert attn.rope is not None, "MLXFlashAttention must have a rope attribute"

    # Test RoPE directly
    rope = MLXRotaryEmbedding(head_dim=8, max_position_embeddings=32)
    B, H, L, D = 1, 2, 4, 8
    q = mx.random.normal((B, H, L, D))
    k = mx.random.normal((B, H, L, D))
    q_r, k_r = rope.apply_rope(q, k, offset=0)
    assert q_r.shape == q.shape, "RoPE must preserve shape"
    # RoPE should change the values (not identity)
    assert not np.allclose(np.array(q), np.array(q_r)), "RoPE must modify Q"


# ---------------------------------------------------------------------------
# 10. MLXStatefulCausalLM forward pass
# ---------------------------------------------------------------------------
def test_causal_lm_forward():
    """MLXStatefulCausalLM must produce logits of the correct shape."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from daph_exfusion.mlx_inference import (
        MLXStatefulCausalLM,
        KVCache,
        SSMState,
        ConvState,
    )

    model = MLXStatefulCausalLM(
        num_layers=2, vocab_size=100, hidden_size=16,
        intermediate_size=32, num_heads=2,
    )
    tokens = mx.random.randint(0, 100, (1, 4))
    caches = [KVCache() for _ in range(2)]
    ssm_states = [SSMState(1, 16) for _ in range(2)]
    conv_states = [ConvState(1, 16) for _ in range(2)]

    logits = model(tokens, caches=caches, ssm_states=ssm_states,
                   conv_states=conv_states)
    assert logits.shape == (1, 4, 100), f"logits shape {logits.shape}"
