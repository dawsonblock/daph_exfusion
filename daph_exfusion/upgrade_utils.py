"""Upgrade utilities for DAPH ExFusion v2026.07.4.2.

These helpers are intentionally standalone so they can be copied into the real
package and wired into existing modules without forcing a broad refactor.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Mapping
from typing import Any


_EXPERT_PATTERNS = (
    re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)experts\[(\d+)\](?:\.|$)"),
)


def _extract_expert_index(layer_name: str) -> int | None:
    """Extract the expert index from a layer name if present.

    Matches patterns like ``experts.0`` or ``experts[1]`` and returns the
    integer index. If no pattern is found, returns ``None``.
    """
    for pattern in _EXPERT_PATTERNS:
        match = pattern.search(layer_name)
        if match:
            return int(match.group(1))
    return None


def aggregate_kfac_scores_to_experts(
    layer_scores: Mapping[str, float],
    num_experts: int,
    path_prefix: str | None = "ffn_path",
    *,
    normalize: bool = True,
    fill_value: float = 0.0,
) -> dict[str, float]:
    """Aggregate layer-level K-FAC scores into per-expert scores.

    Parameters are matched by expert index in names like
    ``ffn_path.experts.0.up.weight`` or ``ffn_path.experts[0].up.weight``.
    Scores are averaged per expert by default so experts with more tracked
    sublayers do not automatically dominate.

    Args:
        layer_scores: Mapping of full parameter names to scalar K-FAC scores.
        num_experts: Total number of experts in the FFN/Mamba module.
        path_prefix: Optional string to filter which layer names to consider.
        normalize: Whether to divide accumulated scores by the number of
            matched layers per expert. Default ``True``.
        fill_value: Value to assign when no scores are found for an expert.

    Returns:
        A dict mapping ``"expert_i"`` keys to aggregated scores.
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    # Initialize per-expert accumulators
    expert_sums: dict[str, float] = {f"expert_{idx}": 0.0 for idx in range(num_experts)}
    expert_counts: dict[str, int] = {f"expert_{idx}": 0 for idx in range(num_experts)}

    for layer_name, raw_score in layer_scores.items():
        # Skip layers that do not contain the path prefix
        if path_prefix and path_prefix not in layer_name:
            continue

        expert_idx = _extract_expert_index(layer_name)
        if expert_idx is None or expert_idx >= num_experts:
            continue

        key = f"expert_{expert_idx}"
        expert_sums[key] += float(raw_score)
        expert_counts[key] += 1

    # Normalize or assign fill values as configured
    aggregated: dict[str, float] = {}
    for idx in range(num_experts):
        key = f"expert_{idx}"
        count = expert_counts[key]
        if count == 0:
            aggregated[key] = float(fill_value)
        elif normalize:
            aggregated[key] = expert_sums[key] / count
        else:
            aggregated[key] = expert_sums[key]

    return aggregated


def lookup_group_policy_robust(
    policies: Mapping[str, Any],
    param_name: str,
    default_factory: Callable[[], Any],
) -> Any:
    """Resolve a parameter merge policy without silently missing near matches.

    Resolution order:
    1. exact parameter name
    2. parent module name, e.g. ``dt_proj`` from ``foo.dt_proj.weight``
    3. longest policy key contained in the parameter path
    4. warning plus default policy

    Args:
        policies: Mapping from parameter or module names to policy instances.
        param_name: Name of the parameter to resolve.
        default_factory: Callable used to create a default policy when no match
            is found.

    Returns:
        The resolved policy instance or a default policy if none matches.
    """
    # Exact match: highest priority
    if param_name in policies:
        return policies[param_name]

    # Try immediate parent of the leaf (e.g. dt_proj for dt_proj.weight)
    parts = [part for part in param_name.split(".") if part]
    parent_name = parts[-2] if len(parts) > 1 else ""
    if parent_name in policies:
        return policies[parent_name]

    # Longest substring match across policy keys
    candidates = [key for key in policies if key and key in param_name]
    if candidates:
        candidates.sort(key=len, reverse=True)
        return policies[candidates[0]]

    # Fallback: warn and instantiate a default policy
    warnings.warn(
        f"Parameter '{param_name}' could not be matched to any policy group. "
        f"Available policy keys: {list(policies.keys())}. Falling back to "
        "default GroupMergePolicy.",
        stacklevel=2,
    )
    return default_factory()