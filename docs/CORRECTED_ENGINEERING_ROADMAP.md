# Corrected Engineering Roadmap

This overlay addresses the issues identified in DAPH ExFusion v2026.07.4.2
without pretending the full source tree is available here.

## 1. Fisher Denominator Dilution

Problem:

Pruned expert deltas can still add Fisher mass to the denominator. That dilutes
surviving deltas.

Fix:

Only add Fisher contribution to the denominator where the expert delta is
non-zero:

```python
active_mask = (delta != 0).to(delta.dtype)
denom = denom + contrib * active_mask
```

Then clamp the denominator with `eps` before division.

Implementation:

Use `difficulty_weighted_fisher_merge_from_deltas_corrected` from
`daph_exfusion.merge_corrections`.

## 2. TIES Sign Election

Problem:

`sign(delta) * abs(delta)` is algebraically equal to `delta`. That means the
supposed magnitude-weighted sign election collapses into a raw weighted sum.

Fix:

Use pure sign majority as the default election mode:

```python
vote += expert_weight * sign(delta)
```

Keep magnitude-weighted voting only as an explicit optional mode.

Implementation:

Use `elect_sign_mask_corrected` or `elect_sign_mask_with_consensus` from
`daph_exfusion.merge_corrections`.

## 3. FNet Cheap-Path Parity

Problem:

The PyTorch cheap path and MLX cheap path do not match structurally.

Fix:

Use the same block layout:

- LayerNorm
- 2D FFT over sequence and hidden axes
- linear projection
- residual add

Implementation:

Use `PyTFNetBlock` from `daph_exfusion.routing_blocks` and replace direct calls
to the old `_cheap_path`.

## 4. Routing Mask Correctness

Problem:

`.any()` causes whole-path execution if any token needs a path. This does not
provide real batched FLOP savings.

Fix:

Short term:

Multiply each path output by the token-level mask before blending.

Long term:

Implement token packing/scatter-gather dispatch if actual FLOP savings matter.

## 5. MLX SSM Pre-Fill State

Problem:

Python loops over sequence length during pre-fill create CPU-to-GPU dispatch
overhead.

Fix:

Short term:

Compile the state accumulation helper with `mx.compile`.

Long term:

Modify the custom selective-scan Metal kernel to emit both the output sequence
and final recurrent state.

## 6. K-FAC Memory Pressure

Problem:

Dense covariance matrices scale as `O(D^2)` and will exhaust consumer hardware.

Fix:

Add a diagonal-only tracking mode for memory-constrained runs. Treat it as an
approximation for ranking, not a perfect K-FAC substitute.
