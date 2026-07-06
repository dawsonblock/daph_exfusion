# Patch Notes

## v4.3.2 — Performance & K-FAC Low-Rank Mode (2026-07-05)

Addresses the four priority next steps identified in the v4.3.1 deep
analysis, moving the codebase from research-ready toward performance-grade
territory suitable for 7- to 13-B local models on Apple Silicon.

### 1. Packed Token Dispatch (Gather-Run-Scatter) for FFN

`DAPHDecoderLayerV2` (PyTorch) now uses packed dispatch for the FFN path
when in merged/inference mode. Tokens that don't select the efficient path
are gathered into a contiguous batch, the FFN runs only on that subset,
and results are scattered back — realizing real FLOP savings instead of
computing all paths on all tokens and masking.

Attention and Mamba paths still run on the full sequence because they
require full-sequence context (K/V cache, recurrent state).

MLX's lazy evaluation model prevents dynamic tensor sizing from data
(required for gather), so the MLX path continues to compute on all tokens
and mask. The FLOP savings are realized in the PyTorch inference path.

- Regression: `test_packed_ffn_dispatch`

### 2. Full-Sequence SSM Scan Verification

Verified that the pre-fill path uses `mamba_selective_scan` (the Metal
kernel) as a single GPU dispatch — no Python-level sequence loop remains
in the production code path. The only `for t in range(L)` loops are in:
- `mamba_selective_scan_reference` — pure-Python reference for correctness
- `ssm_prefill_loop` — retained for ablation, not used in main path
- `test_scan_correctness` — test verification code

MLX does not expose a public `mx.scan` primitive; the Metal kernel already
achieves O(1) dispatch scaling.

### 3. Low-Rank K-FAC Mode for SSM A_log/D

Added a fourth storage mode to `RunningCovariance`: `low_rank=True` stores
`(U, s)` where `U` is `(dim, rank)` with orthonormal columns and `s` is
`(rank,)` eigenvalues, approximating the covariance as `U @ diag(s) @ U^T`.

Memory: O(dim * rank) — at D=4096, rank=32: ~512 KB vs 1.6 GB full, a
99.97% reduction, while capturing the dominant global curvature directions.

The update uses incremental SVD: projects new samples onto the complement
of the current subspace, QR-orthonormalizes the residual, and SVDs the
combined factor to extract the new top-k eigenvectors.

`KFACFisherTracker.get_factors()` returns the factored `(U, s)` form for
low-rank mode, and `layer_score()` / `weight_diag_proxy()` handle the
factored form without materializing the full matrix.

- Config: `KFACConfig(low_rank=True, rank=32)`
- Regression: `test_low_rank_kfac_mode`

### 4. Block-Diagonal K-FAC Edge-Dim Tests

Parametrized test covering dims that are not divisible by `block_size`,
including the edge case where `dim < block_size`:

- (4096, 128) — standard, exactly divisible
- (8192, 128) — standard, exactly divisible
- (4096, 64) — different block size
- (4097, 128) — non-divisible: dim % block_size = 1
- (4100, 128) — non-divisible: dim % block_size = 4
- (100, 128) — dim < block_size (edge case)
- (256, 128) — exactly 2 blocks

Verifies covariance factor shapes, diagonal extraction, and score
computation for all configurations.

- Regression: `test_block_diag_kfac_edge_dims` (7 parametrized cases)

### 5. Cheap-Path FFT Memoisation

`DAPHDecoderLayerV2._cheap_path` now memoises the `fft2` result based on
the normalised tensor's data pointer, shape, and device. Repeated calls
with the same input (e.g. when the residual stream hasn't changed between
paths in the same layer) avoid recomputing the FFT, reducing micro-latency
on long sequences with heavy cheap-path traffic.

### Test results

- **53 passed, 8 failed** (all 8 failures are pre-existing and unrelated).
- All 23 safety-gates tests pass (9 new tests added in v4.3.1 + v4.3.2).

---

## v4.3.1 — Boundary Condition & GQA Patch (2026-07-05)

Three boundary-condition fixes identified in the v4.3 audit, ensuring
complete production stability for short prompts, modern GQA/MQA weight
layouts, and long autoregressive generation runs.

### 1. Short Pre-fill ConvState Padding

When a prompt's sequence length `L` is shorter than `d_conv - 1` (3 for a
standard `d_conv=4` Mamba block), the slice `u[:, -(k-1):, :]` returned a
smaller sequence dimension, collapsing `ConvState.history` from `(B, 3, D)`
to `(B, L, D)`. This caused a shape mismatch crash on the first decoding
step.

Fixed by left-zero-padding the history when `L < d_conv - 1`, preserving
the `(B, d_conv - 1, D)` shape.

Also fixed `ConvState.update()` to return the full `(B, kernel_size, D)`
window (history + new input) instead of `(B, kernel_size - 1, D)`, which
was causing conv1d to receive an input shorter than the kernel size.

- Regression: `test_short_prefill_conv_state_padding`

### 2. Grouped-Query Attention (GQA/MQA) Support

`MLXFlashAttention` previously assumed `num_kv_heads == num_heads`, making
it incompatible with GQA/MQA configurations used by Llama-3, Mistral, Qwen,
etc. K/V projections now produce `num_kv_heads * head_dim` instead of
`hidden_size`, and MLX's `scaled_dot_product_attention` natively broadcasts
K/V across query groups.

`num_kv_heads` parameter threaded through `MLXAttentionPath`,
`MLXDAPHDecoderLayer`, `MLXStatefulDAPHDecoderLayer`, and
`MLXStatefulCausalLM`. RoPE is applied before KV-cache insertion with the
correct position offset.

- Regression: `test_grouped_query_attention_projections`

### 3. Memory-Stable Autoregressive Generation

`MLXStatefulCausalLM.generate()` provides a complete pre-fill + decode
loop with explicit `mx.eval()` calls after each step to prune the lazy
evaluation graph, preventing VRAM accumulation during long generation
runs. Supports greedy (`temperature=0.0`) and stochastic sampling.

- Regression: `test_causal_lm_generation`

### Test results

- **44 passed, 8 failed** (all 8 failures are pre-existing and unrelated).
- All 14 safety-gates tests pass (3 new tests added).

---

## v4.3 — Structural Parity & Hardware Acceleration (2026-07-05)

A three-phase remediation addressing mathematical gaps, hardware execution
bottlenecks, and model fidelity issues identified in the v4.2.1 audit.

### Phase 1: Mathematical Parity & Scaling Fixes

#### 1.1 TIES sign election fix (FFN path)

`compute_ties_aligned_deltas` had the same magnitude-weighted cancellation
bug as the Mamba groups: `w * sign(delta) * delta.abs()` collapses to
`w * delta`, making the "sign election" a weighted sum of deltas. Fixed to
use pure `w * sign(delta)` majority voting.

- Regression: `test_ties_sign_election_is_majority`

#### 1.2 Block-diagonal K-FAC

`RunningCovariance` now supports three storage modes:
- **diagonal** (`diagonal_only=True`): 1D `(dim,)` — O(dim) memory
- **block-diagonal** (`diagonal_only=False, block_size=B`): 3D
  `(num_blocks, B, B)` — O(dim*B) memory. At D=4096, B=128: ~2 MB vs
  ~1.6 GB full. Preserves localized bilinear curvature within each block.
- **full** (`diagonal_only=False, block_size=None`): 2D `(dim, dim)`

`RunningCovariance.update()` now accepts raw activations `(samples, dim)`
and computes the appropriate covariance internally. All consumers
(`get_factors`, `layer_score`, `weight_diag_proxy`) handle all three modes.

- Regression: `test_kfac_block_diagonal_mode`

#### 1.3 Macro-router probability re-normalization

`DAPHDecoderLayerV2` (PyTorch) and `MLXStatefulDAPHDecoderLayer` (MLX)
previously multiplied path outputs by raw softmax probabilities AND the
binary mask, causing tokens that activate only a subset of paths to have
their output scale squashed (e.g., by 0.6 for an easy token using only the
cheap path). This caused exponential scale decay across decoder layers.

Both routers now re-normalize probabilities over selected paths:
`norm_probs = (probs * mask) / (probs * mask).sum()` so active paths always
sum to 1.0.

### Phase 2: Hardware-Accelerated State Capture (MLX)

#### 2.1 Fused GPU state capture in metal kernel

The `_mamba_scan_kernel` Metal kernel now writes the final recurrent state
`h_last` directly from the GPU register in the same pass as the output
sequence. `mamba_selective_scan()` returns `(y, h_last)` instead of just
`y`. This eliminates the sequential O(L) Python pre-fill loop entirely —
pre-fill and state capture complete in a single GPU dispatch.

All callers updated (`MLXMergedMamba.__call__`, stateful decoder prefill
path). The `ssm_prefill_loop` and `_ssm_prefill_step` compiled functions
are retained for reference/ablation.

- Regression: `test_mamba_selective_scan_returns_state`

#### 2.2 Pre-fill path uses GPU scan

`MLXStatefulDAPHDecoderLayer`'s pre-fill path (L > 1 with `ssm_state`) now
calls `mamba_selective_scan` directly to get both outputs and final state
in one GPU pass, instead of running `self.mamba_path(hidden)` for outputs
and then `ssm_prefill_loop` for the state.

### Phase 3: Model Parity & Stateful Wiring

#### 3.1 Standard 1D convolution in Mamba

`MLXMergedMamba` now includes a depthwise `nn.Conv1d` (kernel=4, causal
left-padding) between the input projection gating and the selective scan,
matching the architecture of Mamba-1/Mamba-2/Jamba. This is required for
loading real-world pre-trained Mamba weights.

The PyTorch demo factory (`make_mamba_factory`) also includes conv1d. A
`conv1d` merge policy (drop_rate=0.15, sign_mode="majority") is added to
`DEFAULT_MAMBA_POLICIES`.

`ConvState` class provides a rolling history buffer for single-token
decoding steps. The stateful decoder seeds `ConvState` from the last
`d_conv - 1` projected inputs during pre-fill.

- Regression: `test_mamba_has_conv1d`

#### 3.2 Rotary Position Embeddings (RoPE)

`MLXRotaryEmbedding` applies rotary position embeddings to Q and K before
scaled dot-product attention, matching the architecture of modern
autoregressive Transformers (Llama, Mistral, Qwen). `MLXFlashAttention`
accepts `use_rope=True` (default) and an `offset` parameter for KV-cache
position tracking during decoding.

- Regression: `test_rope_in_flash_attention`

#### 3.3 Strict bridge lenient-loading fix

`load_mlx_model(strict=False)` now **aborts** weight transfer when
validation fails, instead of silently proceeding to load a partial state
dict that would produce nonsense inference. Error messages improved with
structured "Bridge Parity Violation" format listing missing keys in both
directions.

#### 3.4 MLXStatefulCausalLM top-level wrapper

New `MLXStatefulCausalLM` class provides a complete multi-layer causal LM
container with embedding, per-layer state management (KV-cache, SSM state,
conv state), causal masking for pre-fill, and LM head. Handles the
transition from pre-fill (L > 1) to single-token decoding (L = 1).

- Regression: `test_causal_lm_forward`

### Other fixes

- Fixed `mx.clip(threshold, min_val=..., max_val=...)` → positional args
  for MLX 0.31 compatibility. This unblocked 5 previously-failing MLX
  adaptive router tests.
- Fixed `ArrayAt.set()` → inverse-permutation `take_along_axis` for MLX
  0.31 compatibility (ArrayAt doesn't support `.set()` in 0.31).
- Added missing `test_scan_correctness` and `test_scan_with_real_weights`
  functions to `mlx_inference.py` — their absence was silently breaking
  the entire MLX import stack (`_mlx_available = False`).

### Test results

- **41 passed, 8 failed** (all 8 failures are pre-existing and unrelated
  to this patch).
- All 11 safety-gates tests pass.
- All 7 MLX adaptive router tests pass (5 were previously failing).
- New exports: `MLXRotaryEmbedding`, `ConvState`, `MLXStatefulCausalLM`,
  `test_scan_correctness`, `test_scan_with_real_weights`.

---

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
