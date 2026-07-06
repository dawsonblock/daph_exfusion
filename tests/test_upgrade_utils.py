"""Tests for upgrade_utils aggregation and robust policy lookup."""

from daph_exfusion.upgrade_utils import (
    aggregate_kfac_scores_to_experts,
    lookup_group_policy_robust,
)


def test_aggregate_kfac_scores_to_experts_dot_and_bracket_names() -> None:
    scores = {
        "ffn_path.experts.0.up.weight": 2.0,
        "ffn_path.experts.0.down.weight": 4.0,
        "ffn_path.experts[1].up.weight": 10.0,
        "other.experts.1.up.weight": 100.0,
    }
    result = aggregate_kfac_scores_to_experts(scores, 3, "ffn_path")
    assert result == {
        "expert_0": 3.0,
        "expert_1": 10.0,
        "expert_2": 0.0,
    }


def test_lookup_group_policy_robust_prefers_exact_then_parent_then_substring() -> None:
    policies = {
        "dt_proj": "parent",
        "A_log": "core",
        "module.dt_proj.weight": "exact",
    }
    assert lookup_group_policy_robust(policies, "module.dt_proj.weight", dict) == "exact"
    assert lookup_group_policy_robust(policies, "foo.dt_proj.bias", dict) == "parent"
    assert lookup_group_policy_robust(policies, "foo.block.A_log", dict) == "core"


def test_lookup_group_policy_robust_fallback_warns() -> None:
    fallback = lookup_group_policy_robust({}, "foo.unknown.weight", dict)
    assert fallback == {}