# Patch Notes

## v4.3.9 — Architectural Hardening (2026-07-05)

Six fixes addressing cache safety, structural parity, compute efficiency,
robust topology discovery, orchestrator completeness, and Metal shader
scalability.

### 1. FFT Cache Invalidation Hazard

`_cheap_path` memoised FFT results using `data_ptr()` as a cache key.
PyTorch's caching allocator aggressively recycles deallocated memory
addresses, so a freed tensor's `data_ptr()` can be reassigned to a
completely different tensor, causing silent false cache hits on stale
data.

Fix: added `x._version` to the cache key. The version counter increments
on any in-place modification, so stale tensors with recycled addresses
won't match.

### 2. Cheap Path Residual Structure Alignment

`MLXFNetBlock` (MLX) added an internal residual connection
(`residual + self.ff(x_fft)`), but `DAPHDecoderLayerV2._cheap_path`
(PyTorch) returned only `self.cheap_proj(x_fft)` without residual. Both
outer decoder layers add their own residual, so the MLX path
double-residualled, causing numerical divergence across the bridge.

Fix: removed the internal residual from `MLXFNetBlock`. Both paths now
return only the projection output, letting the outer decoder layer
handle residual blending uniformly.

### 3. MLX Routing Compute Savings

`MLXDAPHDecoderLayer` computed all three paths (attention, FFN+Mamba,
cheap) unconditionally even though the macro-router selects only one
dominant path per forward. This negated the efficiency benefit of
routing.

Fix: added `skip_inactive_paths` flag (default `False`). When enabled,
the layer syncs to read the dominant path index and computes only that
path, trading a small device-to-host sync for real FLOP savings.
Disabled by default to preserve JIT graph caching.

### 4. Robust FFN Topology Discovery

`_get_weights` extracted weights from non-SwiGLU experts using hardcoded
indices (`expert[0]`, `expert[3]`). If a user passed an expert with
auxiliary layers (LayerNorm, Dropout, custom activations), the indices
targeted wrong submodules, causing attribute errors or silent corruption.

Fix: added `_extract_linear_weights` helper that dynamically discovers
`nn.Linear` layers by recursively traversing child modules. The first
`nn.Linear` is "up", the last is "down". Replaced both hardcoded
instances.

### 5. Orchestrator Mamba Support

`AutomatedMergePipeline` only automated FFN expert tracking and Fisher
diagonal. Mamba experts required manual Fisher dictionary construction,
making the "automated" pipeline incomplete.

Fix: `execute()` now automatically tracks K-FAC scores and builds Fisher
diagonals for Mamba experts when the layer has a `mamba_path` with
experts. Mamba is included in the coordinate-descent calibration loop
and results are returned in the diagnostic dict.

### 6. Dynamic Metal Shader Register Allocation

The Metal kernel used a static `float h_states[16]` and rejected
`d_state > 16`. Modern Mamba-2 and Jamba models frequently use
`d_state = 32, 64, or higher`.

Fix: replaced the static kernel with a dynamic kernel factory
`_get_mamba_scan_kernel(d_state)` that generates shader source with the
correct array size. Compiled kernels are cached per `d_state`. Supports
`d_state` up to 128 (configurable via `_MAMBA_MAX_D_STATE`). The
hardcoded `d_state > 16` guard is replaced by the dynamic limit.

### Test results

- **63 passed, 0 failed** — full green suite maintained.

---

## v4.3.8 — Metal Shader OOB Guard (2026-07-05)

### 1. Fused Metal Shader Out-Of-Bounds Thread Guard (Critical)

The `_mamba_scan_kernel` Metal shader parallelizes over `B * D` threads
with a threadgroup size of 256. The GPU driver rounds the execution grid
up to a multiple of 256, so when `B * D` is not a multiple of 256 (e.g.
`B=1, D=16` → grid of 16 rounded to 256), padded threads with
`elem >= B * D` would execute without any bounds check.

These padded threads compute `b_idx = elem / d >= 1` and `c_idx = elem % d`,
then perform memory reads/writes at `idx_x = (b_idx * L + t) * d + c_idx`,
which points beyond the allocated tensor bounds. This could cause silent
memory corruption (overwriting adjacent MLX arrays in unified memory) or
GPU hardware exceptions if the out-of-bounds address crosses an
unallocated page boundary.

Fix: added a defensive boundary guard at the top of the Metal shader:

```metal
if (elem >= bsz * d) return;
```

Padded threads now exit immediately before any memory access, preventing
out-of-bounds reads/writes.

### Test results

- **63 passed, 0 failed** — full green suite maintained.

---

## v4.3.7 — SSM Broadcasting Fix & Bridge/Release Hygiene (2026-07-05)

Five fixes addressing the deep architectural audit of v4.3.6.

### 1. MLX SSM Broadcasting Bug (Critical, B>1)

The d_state-aware SSM code used `delta * a[:, None]` to compute the decay
factor for the `(B, D, d_state)` state. This broadcasting pattern fails
for `B > 1` when `B != D`: `(B, D) * (D, 1)` cannot be right-aligned.

Fix: replaced with `(delta * a)[:, :, None]` which correctly broadcasts
`(B, D) * (D,)` → `(B, D)`, then expands to `(B, D, 1)` for the d_state
dimension. Applied in three locations:
- `mamba_selective_scan_reference` (reference scan)
- `test_scan_correctness` (h_last verification loop)
- `MLXStatefulDAPHDecoderLayer.__call__` (single-token decode path)

Also updated `_ssm_prefill_step` and `ssm_prefill_loop` (dead code, used
only in tests) to use the new `(B, D, d_state)` state shape with correct
broadcasting.

### 2. Bridge Validation Returns Cleaned State

`validate_architecture_compatibility` transposed tensors in its local
`cleaned_state` copy, but `load_mlx_model` called `clean_pytorch_keys`
again and re-discovered transposes independently. The validated transpose
fix was lost.

Fix: `validate_architecture_compatibility` now returns the cleaned (and
transposed) state dict on success instead of `True`. `load_mlx_model`
reuses this validated state directly, eliminating the divergent second
clean/transpose pass. Backward compatible: a non-empty dict is truthy,
and `False` is still returned on failure.

### 3. Batch-Size-2 MLX Regression Tests

Added two new tests to guard against B>1 SSM broadcasting regressions:
- `test_stateful_mamba_decode_batch2`: pre-fills with B=2, then
  single-token decodes with B=2, verifying no NaNs from broadcasting
  failures.
- `test_mamba_selective_scan_batch2`: verifies the fused Metal kernel
  matches the Python reference for B=2 with d_state=16.

Also fixed `test_stateful_decoder_with_cache` to pass `d_state=16` to
`SSMState` (was using the default parameter).

### 4. README Version & Documentation Fix

- Updated version badge from `2026.07.4.2.1` to `2026.07.4.3.7`.
- Fixed backwards router documentation: "higher difficulty → lower
  threshold" corrected to "higher difficulty → higher threshold → more
  paths active" (matching the v4.3.4 formula fix).

### 5. Pycache Cleanup

Removed all `__pycache__/` directories and `*.pyc` files from the
repository tree. The `.gitignore` already excludes them.

### Test results

- **63 passed, 0 failed** — 2 new B>1 regression tests added.

---

## v4.3.6 — Runtime Safety Guards (2026-07-05)

Two defensive guards to ensure absolute runtime safety in production
environments.

### 1. Metal Shader Array Boundary Guard (d_state > 16)

The `_mamba_scan_kernel` Metal shader allocates a static register array
of size 16 (`float h_states[16]`) for peak GPU register execution speed.
If a user loads a custom Mamba configuration with `d_state > 16` (e.g.
32 or 64 for increased memory capacity), the inner loop would write
out of bounds, causing undefined GPU memory writes, inference corruption,
GPU execution hangs, or kernel panics.

Fix: `mamba_selective_scan` now validates `d_state` before dispatching
to the Metal kernel and raises a clean `ValueError` if `d_state > 16`,
directing the user to fall back to the Python reference path.

### 2. K-FAC SVD Convergence Fallback

In `RunningCovariance.update()` under the `low_rank=True` path,
`torch.linalg.svd` can fail to converge on heavily ill-conditioned
matrices (e.g. null calibration batches where most activations are
exact zeros), raising a `LinAlgError` that crashes the tracking pipeline.

Fix: the SVD call is now wrapped in a try/except. On `LinAlgError`, a
tiny numerical perturbation (`1e-6 * randn`) is added to break
degeneracy before retrying, guaranteeing tracking continuity.

### Test results

- **61 passed, 0 failed** — full green suite maintained.

---

## v4.3.5 — Real-World Model Compatibility (2026-07-05)

Five structural fixes enabling compatibility with real-world large-scale
causal models (LLaMA-3, Mistral, Mamba-130M, Jamba).

### 1. Mamba State Dimension (d_state) Decoupling (Critical)

`MLXMergedMamba` hardcoded `x_proj_B` and `x_proj_C` to output `d_model`
(e.g. 2048), but standard Mamba-1/2 models use `d_state=16`. This caused
weight loading failures and would result in out-of-bounds reads in the
Metal kernel if validation was bypassed.

Fix: `MLXMergedMamba` now accepts `d_state` (default 16). B/C projections
output `d_state`. The Metal kernel indexes B/C using `d_state` from
`B_shape[2]`. The SSM state now has shape `(B, D, d_state)` instead of
`(B, D)`, matching the standard Mamba architecture where each channel
maintains a `d_state`-dimensional state vector.

Updated: `MLXMergedMamba`, `_mamba_scan_kernel`, `mamba_selective_scan`,
`mamba_selective_scan_reference`, `SSMState`, `MLXStatefulDAPHDecoderLayer`
(single-step decode path), `MLXStatefulCausalLM`, `test_scan_correctness`,
`test_scan_with_real_weights`.

PyTorch `make_mamba_factory` in `demo.py` also updated to use `d_state`.

### 2. GQA Support in PyTorch Attention Path

`SimpleAttention` and `PyTAttentionPath` in `demo.py` were hardcoded to
MHA (`num_kv_heads == num_heads`), preventing K-FAC tracking and Fisher
profiling on GQA models (LLaMA-3, Mistral, Qwen-2) before merging.

Fix: added `num_kv_heads` parameter. K/V projections produce
`num_kv_heads * head_dim`. When `num_kv_heads < num_heads`, K/V are
repeat-interleaved to match the query head count.

Also fixed `torch.silu` → `F.silu` in the demo Mamba block.

### 3. Half-Precision FFT Safety

`torch.fft.fft2` does not support `float16` or `bfloat16` on many
platforms, causing runtime crashes during half-precision inference.

Fix: `_cheap_path` now upcasts to `float32` before FFT and casts back to
the original dtype afterward.

### 4. Low-Rank Frobenius Norm Damping Correction

`layer_score()` with `score_mode="fro"` computed `||s_damped||` which
misses the damping energy of the inactive `(D - k)` dimensions:
`d^2 * (D - k)`.

Fix: `||A||_F^2 = sum(s_damped^2) + d^2 * (D - k)`, correctly accounting
for isotropic damping across the full dimension.

### 5. torch.compile Compatibility Switch

The packed token dispatch (`_packed_dispatch`) uses dynamic boolean
slicing (`nonzero()`) which causes graph breaks under `torch.compile`,
triggering JIT recompilation on every forward step.

Fix: added `use_packed_dispatch` flag (default `True`) to
`DAPHDecoderLayerV2`. Set to `False` when using `torch.compile` to
disable packed dispatch and use the static full-sequence path.

### Test results

- **61 passed, 0 failed** — full green suite maintained.
- Updated `test_ssm_state_update` and `test_mamba_selective_scan_returns_state`
  to expect the new `(B, D, d_state)` state shape.

---

## v4.3.4 — Full Test Suite Green (2026-07-05)

Resolves all 8 remaining pre-existing test failures, achieving a 100%
pass rate across the entire test suite (61/61). The failures were caused
by out-of-date test mocks and a mathematical inversion in the difficulty
modulation, not by functional defects in the core algorithms.

### 1. Mamba Seed Determinism (test_merge_toolkit.py)

`test_mamba_seed_determinism` created two `MemoryBankExFusionMamba` blocks
without re-seeding PyTorch's RNG between instantiations, so the expert
networks started with different random weights. Seeding the merge step
alone was insufficient since the inputs to the merge were already different.

Fix: re-seed `torch.manual_seed(0)` before each block instantiation.

### 2. Bridge Key-Mapping (mlx_inference.py, 3 tests)

`clean_pytorch_keys` used leading-dot patterns (e.g. `.ffn_path.merged_ffn.`)
that only matched nested keys (e.g. `layer.0.ffn_path.merged_ffn.up.weight`)
but not root-level keys (e.g. `ffn_path.merged_ffn.up.weight`). The ignore
patterns for runtime state (`.experts.`, `.ffn_path.router.`) had the same
issue.

Fix: refactored to check both `key.startswith(pattern)` for root-level keys
and `".pattern" in key` for nested keys. Removed leading dots from ignore
patterns so substring matching works at any nesting depth.

### 3. Benchmark DummyModel (test_benchmark.py, 2 tests)

`DummyModel` contained only a `nn.Linear` layer. When `evaluate_lra_copy`
fed integer token IDs, the linear projection received dimensionally
incompatible inputs (`Long` vs `Float` dtype mismatch).

Fix: added an `nn.Embedding` layer that projects integer token IDs to
continuous vectors before the linear layer. The model auto-detects integer
inputs and routes them through the embedding.

### 4. Adaptive Router Difficulty Modulation (adaptive_top_p_router.py,
       mlx_inference.py, 2 tests)

The difficulty-threshold formula was mathematically inverted:
`threshold = base - scale * (diff - 0.5)`. In top-p (nucleus) selection,
a *higher* threshold means *more* paths are selected (need to accumulate
more probability mass). The formula made higher difficulty → lower
threshold → fewer paths, which is backwards from the design intent
("higher difficulty → more paths active").

Fix: inverted to `threshold = base + scale * (diff - 0.5)` in both the
PyTorch (`AdaptiveTopPMacroRouter`) and MLX (`MLXAdaptiveTopPMacroRouter`)
routers. Now: easy (diff=0) → lower threshold → fewer paths; hard (diff=1)
→ higher threshold → more paths.

Also fixed `test_daph_decoder_layer_v2_forward`: used `torch.silu` which
doesn't exist; replaced with `torch.nn.functional.silu`.

### Test results

- **61 passed, 0 failed** — full green test suite for the first time.
- All 23 safety-gates tests pass.
- All 8 previously-failing tests now pass.

---

## v4.3.3 — Low-Rank K-FAC Crash Fix & MLX Dispatch Optimization (2026-07-05)

Three fixes from a strict boundary-condition audit of the v4.3.2 low-rank
K-FAC SVD update and MLX stateful generation pipeline.

### 1. Low-Rank K-FAC SVD Update Dimensional Crash (Critical)

`RunningCovariance.update()` in `merge_toolkit.py` used elementwise
multiplication (`*`) instead of matrix multiplication (`@`) when combining
the residual basis with the new sample scaling:

```
# BUG: Q_resid (D, k) * R_resid (k, S) → shape mismatch crash
Q_resid * (R_resid * scale_new)

# FIX: Q_resid (D, k) @ R_resid (k, S) → (D, S) correct
Q_resid @ (R_resid * scale_new)
```

This crashed on the second batch (the incremental SVD update branch). The
original test only ran one forward/backward pass, so it never triggered the
`else:` branch where the bug lived.

Also fixed `scale_new` being a Python float (no `.sqrt()` method) — now
uses `** 0.5` operator.

- Regression: `test_low_rank_kfac_mode` now runs two batches

### 2. Low-Rank Trace Damping Deficit

`layer_score()` computed `trace(A) = sum(s + d) = sum(s) + k * d`, but the
mathematically exact trace of `U @ diag(s) @ U^T + d * I` is
`sum(s) + d * dim`. For rank=32, dim=4096, this missed `(dim - k) * d` of
damping mass, skewing layer-importance scores.

Fixed to: `tr_a = sum(s_damped) + d * (dim - k)`, which correctly accounts
for the isotropic damping across the full dimension.

### 3. MLX Generate Dispatch Consolidation

`MLXStatefulCausalLM.generate()` previously called `mx.eval()` in separate
loops for `next_token`, KV-caches, SSM states, and conv states — resulting
in `1 + 3 * num_layers` sequential GPU dispatches per token (97 dispatches
for a 32-layer model).

Consolidated into a single `mx.eval(*eval_targets)` call per step, so MLX
compiles and executes all arrays in one GPU pass — reducing to 1 dispatch
per token.

### Test results

- **53 passed, 8 failed** (all 8 failures are pre-existing and unrelated).
- All 23 safety-gates tests pass.
- `test_low_rank_kfac_mode` now runs two batches, covering the incremental
  SVD update branch.

---

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
