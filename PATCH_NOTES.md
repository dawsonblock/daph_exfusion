# Patch Notes

## v4.2.1 — Remediation Patch (2026-07-05)

Four defects called out in the v4.2 review are fixed in this patch. Each fix
is pinned by a regression test in `tests/test_safety_gates.py` so a future
regression breaks the build immediately.

### 1. Mamba sign-election now uses `majority` everywhere

`DEFAULT_MAMBA_POLICIES` previously set `sign_mode="magnitude_weighted"` for
`in_proj`, `out_proj`, `x_proj`, and `dt_proj`. Under `magnitude_weighted`,
`sign · |Δ|` collapses to the raw delta, so large-magnitude outliers swamp
the election and the "vote" is meaningless. All six groups now default to
`sign_mode="majority"` (the mathematically correct interpretation of a sign
election). Researchers may still override to `"magnitude_weighted"` for
ablations, but the shipped default is `majority`.

- Regression: `test_default_mamba_policies_use_majority_sign_election`

### 2. MLX pre-fill loop uses a compiled per-step kernel

The stateful decoder's pre-fill path built L eager graph nodes per timestep
(`exp`, `mul`, `mul`, `mul`, `add`), so CPU→GPU graph build dominated latency
for L≥1k. The per-step update is now wrapped in `@mx.compile` as
`_ssm_prefill_step`, fusing the elementwise chain into a single compiled
kernel call per timestep. The Python loop remains (the sequential SSM scan
has a data dependency across timesteps) but each iteration is one compiled
node instead of ~5 eager ops. The pure-Python reference loop
(`mamba_selective_scan_reference`) is preserved for correctness checks.

- New exports: `_ssm_prefill_step`, `ssm_prefill_loop`
- Regression: `test_compiled_prefill_matches_reference` (compiled vs.
  reference final state within 1e-6)

### 3. K-FAC tracker defaults to diagonal-only factors

`RunningCovariance` previously allocated a full `(dim, dim)` covariance per
factor — ~1.6 GB per layer per expert at `d_model=4096`, unusable on any
consumer GPU. `KFACConfig.diagonal_only` (default `True`) now makes each
factor store only the diagonal: a 1D tensor of length `dim` (a few MB). The
forward/backward hooks compute `(x * x).sum(0)` directly instead of
`x.t() @ x`, so the large transient covariance is also avoided. All
consumers (`get_factors`, `layer_score`, `weight_diag_proxy`) handle both
1D (diagonal) and 2D (full) factors, so setting `diagonal_only=False` still
works for ablations.

- Regression: `test_kfac_diagonal_only_tracker_is_1d`,
  `test_kfac_full_covariance_still_supported`

### 4. PT→MLX bridge fails loudly on missing keys

`validate_architecture_compatibility` previously `continue`d past any PT
key with no MLX counterpart, so a single missing projection weight produced
nonsense inference but went unnoticed. It now detects missing keys in both
directions (`missing_in_mlx`, `missing_in_pt`) and raises `RuntimeError`
under `raise_on_mismatch=True` (the default in `load_mlx_model` with
`strict=True`), or emits a warning and returns `False` when lenient. Also
fixed: the bridge used `Module.iter_flat_views()`, which does not exist in
MLX 0.31; it now uses `mlx.utils.tree_flatten` to enumerate parameters.

- Regression: `test_bridge_raises_on_missing_keys`,
  `test_validate_raises_on_missing_keys`,
  `test_validate_warns_on_missing_keys_when_lenient`

### Other

- Fixed a pre-existing `SyntaxError` in `tests/test_import.py` (unterminated
  multi-line f-string) that blocked the whole suite from collecting.
- Bumped version to `2026.07.4.2.1` in `__init__.py` and `pyproject.toml`.

### Known pre-existing failures (not addressed in this patch)

The following test failures exist on MLX 0.31 / the current environment and
are unrelated to the four fixes above; they are tracked separately:

- `test_adaptive_router.py`, `test_mlx_adaptive_router.py`: `mx.clip` was
  called with `min_val`/`max_val` kwargs; MLX 0.31 requires positional
  `a_min`/`a_max`.
- `test_benchmark.py`: shape mismatch in benchmark fixtures.
- `test_bridge.py::test_map_key_*`: `clean_pytorch_keys` key-mapping does
  not match the test expectations.
- `test_merge_toolkit.py::test_mamba_seed_determinism`: the test does not
  seed expert weight initialization, so two `make_block()` calls produce
  different experts.

---

## 1. `merge_toolkit.py`: K-FAC Expert Aggregation

Add or import:

```python
from daph_exfusion.upgrade_utils import aggregate_kfac_scores_to_experts
```

Use it after K-FAC tracking:

```python
expert_scores = aggregate_kfac_scores_to_experts(
    layer_scores=tracker.all_layer_scores(),
    num_experts=num_experts,
    path_prefix="ffn_path",
)
```

The utility parses names such as:

```text
ffn_path.experts.0.up.weight
ffn_path.experts[1].down.weight
mamba_path.experts.2.dt_proj.weight
```

and emits:

```python
{"expert_0": 0.82, "expert_1": 0.43, "expert_2": 0.91}
```

## 2. `merge_toolkit.py`: Robust Mamba Policy Lookup

Replace the body of `_lookup_group_policy` with:

```python
from daph_exfusion.upgrade_utils import lookup_group_policy_robust

def _lookup_group_policy(policies, param_name):
    return lookup_group_policy_robust(
        policies=policies,
        param_name=param_name,
        default_factory=GroupMergePolicy,
    )
```

This keeps exact matching, adds parent-name matching, then longest substring
matching, and warns before falling back to the default policy.

## 3. `mlx_inference.py`: Stateful Decoding Fixes

The current skeleton should be upgraded in three parts:

1. `KVCache.update(k, v)` must concatenate keys and values on the sequence axis.
2. `SSMState` must own the recurrent hidden state and expose `update(h_new)`.
3. `MLXStatefulDAPHDecoderLayer.__call__` must:
   - update KV cache before attention,
   - use single-token recurrence for Mamba when `L == 1`,
   - blend `attn_out`, `eff_out`, and `cheap_out` with router masks and router
     probabilities.

The exact shape assumptions used by the supplied implementation are:

```text
hidden: (B, L, D)
k/v cache: (B, num_heads, L_cache, head_dim)
SSM hidden state: (B, D)
```

If the real Mamba implementation uses expanded inner dimensions, adjust
`SSMState` to match the post-`in_proj` recurrent channel count instead of
blindly using `hidden_size`.

## 4. Real Validation Gates

Run at minimum:

```bash
python -m compileall -q daph_exfusion
python -m pytest tests/test_import.py tests/test_upgrade_utils.py
```

Then run a real calibration smoke:

```bash
python - <<'PY'
import torch
from daph_exfusion.orchestrator import AutomatedMergePipeline
print("orchestrator import ok")
PY
```

For MLX, validate on Apple Silicon with `mlx` installed:

```bash
python - <<'PY'
import mlx.core as mx
from daph_exfusion.mlx_inference import KVCache, SSMState
print("mlx state imports ok")
PY
```
