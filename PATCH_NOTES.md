# Patch Notes

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
