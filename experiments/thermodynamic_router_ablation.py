"""Experimental thermodynamic-router ablation.

This module is intentionally not part of the production ExFusion router. The
full Cv statistic requires materializing the same ``O(L^2)`` score tensor as
attention, so it should be used only to test whether Cv predicts useful routing
targets such as token loss, attention entropy, or merge residual error.

The classes defined here mirror the overlay used in the ExFusion v5 research
branch. They are provided to support ablation tests and should not be wired
into the primary decoder or routing logic by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class CvCostEstimate:
    batch: int
    heads: int
    seq_len: int
    head_dim: int

    @property
    def score_elements(self) -> int:
        return self.batch * self.heads * self.seq_len * self.seq_len

    @property
    def qk_multiply_adds(self) -> int:
        return self.score_elements * self.head_dim


class FullSpecificHeatProbe(nn.Module):
    """Full-attention Cv probe for offline diagnostics only."""

    def __init__(self, d_model: int, num_heads: int = 4, beta: float = 1.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.beta = beta
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def estimate_cost(self, batch: int, seq_len: int) -> CvCostEstimate:
        return CvCostEstimate(
            batch=batch,
            heads=self.num_heads,
            seq_len=seq_len,
            head_dim=self.head_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return token-level Cv with explicit O(L^2) score materialization."""
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        energy = -torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probs = torch.softmax(-self.beta * energy, dim=-1)
        mean_energy = torch.sum(probs * energy, dim=-1)
        mean_energy_sq = torch.sum(probs * energy.square(), dim=-1)
        cv_heads = self.beta ** 2 * (mean_energy_sq - mean_energy.square())
        return cv_heads.mean(dim=1)


class CheapPreAttentionDifficultyHead(nn.Module):
    """Cheap baseline router candidate with O(BLD) complexity."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = self.norm(x)
        variance = centered.var(dim=-1, unbiased=False)
        residual_norm = centered.norm(dim=-1) / (centered.shape[-1] ** 0.5)
        novelty = (centered[:, 1:] - centered[:, :-1]).norm(dim=-1)
        novelty = torch.cat([novelty[:, :1], novelty], dim=1)
        features = torch.stack([variance, residual_norm, novelty], dim=-1)
        return torch.sigmoid(self.score(features)).squeeze(-1)


def cv_vs_loss_correlation(cv: torch.Tensor, token_loss: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between flattened Cv and token-level loss."""
    cv_flat = cv.flatten().float()
    loss_flat = token_loss.flatten().float()
    cv_centered = cv_flat - cv_flat.mean()
    loss_centered = loss_flat - loss_flat.mean()
    denom = cv_centered.norm() * loss_centered.norm()
    if float(denom) == 0.0:
        return torch.tensor(0.0, device=cv.device)
    return torch.dot(cv_centered, loss_centered) / denom