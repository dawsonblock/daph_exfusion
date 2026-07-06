"""Tests for the thermodynamic router ablation module."""

import torch

from experiments.thermodynamic_router_ablation import (
    CheapPreAttentionDifficultyHead,
    FullSpecificHeatProbe,
    cv_vs_loss_correlation,
)


def test_full_specific_heat_probe_shape_and_cost() -> None:
    probe = FullSpecificHeatProbe(d_model=16, num_heads=4)
    x = torch.randn(2, 8, 16)
    cv = probe(x)
    cost = probe.estimate_cost(batch=2, seq_len=8)
    # The Cv tensor should have shape (B, L)
    assert cv.shape == (2, 8)
    # Cost estimates should reflect the number of score elements and multiply-adds
    assert cost.score_elements == 2 * 4 * 8 * 8
    assert cost.qk_multiply_adds == cost.score_elements * 4


def test_cheap_pre_attention_head_shape() -> None:
    head = CheapPreAttentionDifficultyHead(d_model=16)
    x = torch.randn(2, 8, 16)
    score = head(x)
    # Cheap head outputs a difficulty score per token between 0 and 1
    assert score.shape == (2, 8)
    assert torch.all(score >= 0)
    assert torch.all(score <= 1)


def test_cv_vs_loss_correlation_returns_scalar() -> None:
    cv = torch.randn(2, 8)
    loss = torch.randn(2, 8)
    corr = cv_vs_loss_correlation(cv, loss)
    # Correlation should be a 0‑dimensional tensor
    assert corr.ndim == 0