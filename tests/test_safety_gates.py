"""Safety-gate regression tests for the v4.2.1 remediation.

Each test pins one of the four defects fixed in the v4.2.1 patch so a
future regression breaks the build immediately:

  1. ``test_default_mamba_policies_use_majority_sign_election``
     — guards the sign-mode fix in DEFAULT_MAMBA_POLICIES.
  2. ``test_compiled_prefill_matches_reference``
     — guards the @mx.compile'd pre-fill step kernel.
  3. ``test_kfac_diagonal_only_tracker_is_1d``
     — guards the diagonal-only K-FAC storage.
  4. ``test_bridge_raises_on_missing_keys``
     — guards the strict missing-key validation in the PT->MLX bridge.
"""
import pytest
import torch
import torch.nn as nn

from daph_exfusion.merge_toolkit import (
    DEFAULT_MAMBA_POLICIES,
    KFACConfig,
    KFACFisherTracker,
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

    mx.set_default_device(mx.cpu)
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
    with pytest.raises(RuntimeError, match="Parameter mismatch"):
        validate_architecture_compatibility(bad_state, mlx_model, raise_on_mismatch=True)
