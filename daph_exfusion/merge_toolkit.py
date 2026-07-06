"""
DAPH ExFusion Unified Merge Toolkit — v4.2 (Corrected, Tested)
================================================================
Research module for expert merging in Mixture-of-Experts models.

Includes:
  - K-FAC Fisher tracker (Linear + attention projections)
  - Difficulty-weighted DARE (Drop And REscale)
  - TIES v2 (per-parameter majority/magnitude-weighted sign election)
  - Fisher-weighted merge with group boosting (SSM core + sensitive projections)
  - Mamba grouped merge with per-group conflict-resolution policies
  - Greedy per-group calibration search (Mamba + FFN unified loop)
  - MemoryBankExFusionFFN & MemoryBankExFusionMamba classes
  - DAPHDecoderLayer with macro-routing + ExFusion paths

Usage:
    from daph_exfusion import (
        MemoryBankExFusionFFN, MemoryBankExFusionMamba, DAPHDecoderLayer,
        KFACFisherTracker, KFACConfig, unified_calibration_loop,
    )

Author: DAPH Research Team
Version: 2026.07.4.2

Known limitations (honest):
  - K-FAC produces layer-level scores; per-expert K-FAC aggregation is not
    yet implemented. incorporate_kfac_scores() requires pre-aggregated scores.
  - The macro router is a simple difficulty predictor, not a learned gating
    network. It is sufficient for research prototyping but not SOTA routing.
  - Mamba block_factory must expose standard in_proj/out_proj/x_proj/dt_proj
    parameters for grouped merge policies to apply correctly.
"""


import copy
import hashlib
import math
import zlib
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict, Callable, Union, Tuple, Iterable, Literal, Any

# -----------------------------------------------------------------------------
# Upgrade imports
#
# On patched builds, prefer robust group policy resolver from upgrade_utils.
try:
    from .upgrade_utils import lookup_group_policy_robust  # type: ignore
except Exception:
    lookup_group_policy_robust = None  # type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 0. SHARED UTILITIES
# =============================================================================

class SwiGLUFFN(nn.Module):
    """Proper SwiGLU FFN module: down(silu(gate(x)) * up(x)).

    Fix history: earlier drafts built `nn.Sequential(up, gate,
    SwiGLUGate(...), down)`, but a `Sequential` only ever threads a single
    tensor between stages, so `gate` would receive `up`'s output instead of
    the original `x` — a shape/semantics bug that surfaces the first time the
    expert is actually forward-propagated. This module computes both
    projections directly against `x` and gates explicitly.
    """

    def __init__(self, up: nn.Linear, gate: Optional[nn.Linear], down: nn.Linear):
        super().__init__()
        self.up = up
        self.gate = gate
        self.down = down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate is not None:
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.down(F.silu(self.up(x)))

    def __repr__(self):
        return (f"SwiGLUFFN(up={self.up.weight.shape}, "
                f"gate={'yes' if self.gate is not None else 'no'}, down={self.down.weight.shape})")


def _is_ssm_core_param(name: str) -> bool:
    """Identify Mamba SSM core parameters that need special protection."""
    ssm_patterns = ("A_log", "A", "D", "dt_proj", "delta_proj", "x_proj", "B_proj", "C_proj", "conv1d")
    return any(p in name for p in ssm_patterns)


def _resolve_param(expert: nn.Module, name: str) -> torch.Tensor:
    """
    Safely get parameter data from a module by name.

    `get_parameter` is the fast path but only resolves dotted paths through
    real submodules (e.g. "up.weight" on a module with a `.up` submodule);
    it raises for names it can't route (e.g. a bare custom attribute, or on
    some module container types), in which case we fall back to a linear
    scan over `named_parameters()`.
    """
    if hasattr(expert, "get_parameter"):
        try:
            return expert.get_parameter(name).data
        except (AttributeError, RuntimeError, KeyError):
            pass
    for n, p in expert.named_parameters():
        if n == name:
            return p.data
    raise ValueError(f"Parameter {name} not found in expert")


def _set_param(expert: nn.Module, name: str, value: torch.Tensor) -> None:
    """Safely set parameter data on a module by name. See `_resolve_param`."""
    if hasattr(expert, "get_parameter"):
        try:
            expert.get_parameter(name).data.copy_(value)
            return
        except (AttributeError, RuntimeError, KeyError):
            pass
    for n, p in expert.named_parameters():
        if n == name:
            p.data.copy_(value)
            return
    raise ValueError(f"Parameter {name} not found in expert")


# =============================================================================
# 1. K-FAC FISHER TRACKER
# =============================================================================

@dataclass
class KFACConfig:
    ema_decay: float = 0.95
    damping: float = 1e-4
    score_mode: str = "log_trace"  # {"trace", "log_trace", "fro", "spectral_proxy"}
    normalize_by_params: bool = True
    diag_eps: float = 1e-8
    max_samples_per_batch: Optional[int] = 4096
    track_bias: bool = False
    attention_only_projections: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # Kronecker factor storage mode:
    #   diagonal_only=True  — store E[a_i^2] as a 1D vector (O(dim) memory).
    #   diagonal_only=False, block_size=None — full E[a a^T] (O(dim^2) memory).
    #   diagonal_only=False, block_size=B — block-diagonal E[a a^T] with B-sized
    #     blocks (O(dim * B) memory).  At D=4096, B=128: ~2 MB vs ~1.6 GB full.
    #     Preserves localized bilinear curvature within each block.
    diagonal_only: bool = True
    block_size: Optional[int] = 128


class RunningCovariance:
    """Running EMA of a covariance estimate, supporting three storage modes.

    Accepts raw activations ``x`` of shape ``(samples, dim)`` and computes the
    appropriate covariance representation internally:

    - **diagonal** (``diagonal_only=True``): stores ``(dim,)`` — E[a_i^2].
    - **block-diagonal** (``diagonal_only=False, block_size=B``): stores
      ``(num_blocks, B, B)`` — block-wise E[a a^T].  The last block is padded
      if ``dim`` is not divisible by ``B``.
    - **full** (``diagonal_only=False, block_size=None``): stores ``(dim, dim)``.
    """

    def __init__(self, dim: int, decay: float = 0.95, *,
                 diagonal_only: bool = True, block_size: Optional[int] = 128,
                 device=None, dtype=torch.float32):
        self.decay = decay
        self.diagonal_only = diagonal_only
        self.block_size = block_size
        self.dim = dim

        if diagonal_only:
            self.mode = "diagonal"
            self.value = torch.zeros(dim, device=device, dtype=dtype)
        elif block_size is not None and block_size < dim:
            self.mode = "block"
            self.num_blocks = (dim + block_size - 1) // block_size
            self.last_block_size = dim - (self.num_blocks - 1) * block_size
            # Store blocks in a single padded tensor for uniform EMA updates.
            # The last block may be smaller; we pad it to block_size and track
            # the real size so scoring/proxy methods can ignore padding.
            self.value = torch.zeros(self.num_blocks, block_size, block_size,
                                     device=device, dtype=dtype)
        else:
            self.mode = "full"
            self.value = torch.zeros(dim, dim, device=device, dtype=dtype)
        self.initialized = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Accept raw activations ``(samples, dim)`` and update the EMA."""
        x = x.detach().float()
        samples = max(x.shape[0], 1)

        if self.mode == "diagonal":
            cov = (x * x).sum(0) / samples
        elif self.mode == "block":
            bs = self.block_size
            nb = self.num_blocks
            # Reshape into (samples, num_blocks, block_size), zero-padding the
            # last block's unused columns so all blocks have uniform width.
            padded = torch.zeros(x.shape[0], nb * bs, device=x.device, dtype=x.dtype)
            padded[:, :self.dim] = x
            x_blocks = padded.view(samples, nb, bs).permute(1, 0, 2)  # (nb, S, B)
            # cov per block: (nb, B, B) = x_blocks^T @ x_blocks
            cov = torch.bmm(x_blocks.transpose(1, 2), x_blocks) / samples
        else:  # full
            cov = (x.t() @ x) / samples

        if not self.initialized:
            self.value.copy_(cov)
            self.initialized = True
        else:
            self.value.mul_(self.decay).add_(cov, alpha=(1.0 - self.decay))

    def get(self) -> torch.Tensor:
        return self.value

    def diagonal(self) -> torch.Tensor:
        """Return the full diagonal as a 1D ``(dim,)`` tensor in any mode."""
        if self.mode == "diagonal":
            return self.value
        elif self.mode == "block":
            bs = self.block_size
            diags = torch.diagonal(self.value, dim1=1, dim2=2)  # (num_blocks, bs)
            # Flatten and trim padding from the last block.
            full = diags.reshape(-1)[:self.dim]
            return full
        else:
            return torch.diagonal(self.value, dim1=0, dim2=1)


class KFACFisherTracker:
    """
    Tracks K-FAC Fisher factors for Linear layers (including attention
    projections when implemented as separate nn.Linear modules):
        A = E[a a^T]   (input activation covariance)
        G = E[g g^T]   (output-gradient covariance)
    Exposes scalar layer scores and a diagonal Kronecker proxy for weights.
    """

    def __init__(self, model: nn.Module, config: Optional[KFACConfig] = None,
                 attention_projections_only: bool = False):
        self.model = model
        self.config = config or KFACConfig()
        self.attention_projections_only = attention_projections_only
        self.handles = []
        self.a_factors: Dict[str, RunningCovariance] = {}
        self.g_factors: Dict[str, RunningCovariance] = {}
        self.layer_modules: Dict[str, nn.Module] = {}
        self.layer_param_counts: Dict[str, int] = {}
        self._register_hooks()

    def _iter_supported_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if self.attention_projections_only:
                    if any(tag in name for tag in self.config.attention_only_projections):
                        yield name, module
                else:
                    yield name, module

    def _register_hooks(self):
        for name, module in self._iter_supported_layers():
            self.layer_modules[name] = module
            self.layer_param_counts[name] = module.weight.numel()

            in_dim = module.in_features + (1 if self.config.track_bias and module.bias is not None else 0)
            out_dim = module.out_features

            self.a_factors[name] = RunningCovariance(
                in_dim, decay=self.config.ema_decay,
                diagonal_only=self.config.diagonal_only,
                block_size=self.config.block_size,
                device=module.weight.device, dtype=torch.float32,
            )
            self.g_factors[name] = RunningCovariance(
                out_dim, decay=self.config.ema_decay,
                diagonal_only=self.config.diagonal_only,
                block_size=self.config.block_size,
                device=module.weight.device, dtype=torch.float32,
            )

            self.handles.append(module.register_forward_pre_hook(self._make_forward_hook(name)))
            self.handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))

    def _flatten_samples(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            flat = x
        else:
            flat = x.reshape(-1, x.shape[-1])
        if self.config.max_samples_per_batch is not None and flat.shape[0] > self.config.max_samples_per_batch:
            idx = torch.randperm(flat.shape[0], device=flat.device)[: self.config.max_samples_per_batch]
            flat = flat[idx]
        return flat

    def _make_forward_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            x = inputs[0].detach()
            x = self._flatten_samples(x).float()
            if self.config.track_bias and module.bias is not None:
                ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
                x = torch.cat([x, ones], dim=-1)
            # Pass raw activations; RunningCovariance.update computes the
            # appropriate covariance (diagonal / block-diagonal / full) internally.
            self.a_factors[name].update(x)
        return hook

    def _make_backward_hook(self, name: str):
        def hook(module: nn.Module, grad_input, grad_output):
            g = grad_output[0].detach()
            g = self._flatten_samples(g).float()
            self.g_factors[name].update(g)
        return hook

    @torch.no_grad()
    def get_factors(self, layer_name: str):
        A = self.a_factors[layer_name].get()
        G = self.g_factors[layer_name].get()
        d = self.config.damping
        if A.dim() == 1:
            # Diagonal-only mode: damping is added elementwise.
            A = A + d
            G = G + d
        elif A.dim() == 3:
            # Block-diagonal mode: add damping to each block's diagonal.
            nb, bs, _ = A.shape
            eye = torch.eye(bs, device=A.device, dtype=A.dtype).unsqueeze(0)  # (1, B, B)
            A = A + d * eye
            G = G + d * eye
        else:
            A = A + d * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            G = G + d * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        return A, G

    @torch.no_grad()
    def layer_score(self, layer_name: str) -> float:
        A, G = self.get_factors(layer_name)
        ndim = A.dim()

        def _trace(t):
            if ndim == 1:
                return t.sum()
            elif ndim == 3:
                # Sum of traces of all blocks (ignoring padding in last block).
                diags = torch.diagonal(t, dim1=1, dim2=2)  # (num_blocks, bs)
                return diags.sum()
            else:
                return torch.trace(t)

        if self.config.score_mode == "trace":
            score = _trace(A) * _trace(G)
        elif self.config.score_mode == "log_trace":
            score = torch.log1p(_trace(A)) * torch.log1p(_trace(G))
        elif self.config.score_mode == "fro":
            score = torch.norm(A, p="fro") * torch.norm(G, p="fro")
        elif self.config.score_mode == "spectral_proxy":
            if ndim == 1:
                ev_a = A.clamp_min(0)
                ev_g = G.clamp_min(0)
            elif ndim == 3:
                # Eigenvalues of each block; take the global max.
                ev_a = torch.linalg.eigvalsh(A).clamp_min(0)
                ev_g = torch.linalg.eigvalsh(G).clamp_min(0)
            else:
                ev_a = torch.linalg.eigvalsh(A)
                ev_g = torch.linalg.eigvalsh(G)
            score = ev_a.max().clamp_min(0) * ev_g.max().clamp_min(0)
        else:
            raise ValueError(f"Unknown score_mode: {self.config.score_mode}")

        if self.config.normalize_by_params:
            score = score / max(self.layer_param_counts[layer_name], 1)

        return float(score.item())

    @torch.no_grad()
    def all_layer_scores(self) -> Dict[str, float]:
        return {name: self.layer_score(name) for name in self.layer_modules.keys()}

    @torch.no_grad()
    def weight_diag_proxy(self, layer_name: str) -> torch.Tensor:
        """diag(G kron A) = outer(diag(G), diag(A)) reshaped to weight shape.

        Works in all three storage modes (diagonal, block-diagonal, full) by
        extracting the full diagonal via ``RunningCovariance.diagonal()``.
        """
        module = self.layer_modules[layer_name]
        A, G = self.get_factors(layer_name)

        # Extract the full diagonal regardless of storage mode.
        if A.dim() == 1:
            diag_a = A
            diag_g = G
        elif A.dim() == 3:
            diag_a = self.a_factors[layer_name].diagonal()
            diag_g = self.g_factors[layer_name].diagonal()
            # Re-apply damping to the extracted diagonals.
            diag_a = diag_a + self.config.damping
            diag_g = diag_g + self.config.damping
        else:
            diag_a = torch.diag(A)
            diag_g = torch.diag(G)

        if self.config.track_bias and module.bias is not None:
            diag_a_w = diag_a[:-1]
        else:
            diag_a_w = diag_a

        diag_f = torch.outer(diag_g, diag_a_w)
        return diag_f.view_as(module.weight).clamp_min(self.config.diag_eps)

    @torch.no_grad()
    def build_weight_importance(self, fisher_damping: str = "sqrt", normalize: str = "mean") -> Dict[str, torch.Tensor]:
        importance = {}
        for name in self.layer_modules.keys():
            imp = self.weight_diag_proxy(name)
            if fisher_damping == "sqrt":
                imp = torch.sqrt(imp)
            elif fisher_damping == "log1p":
                imp = torch.log1p(imp)

            if normalize == "mean":
                imp = imp / imp.mean().clamp_min(1e-8)
            elif normalize == "median":
                imp = imp / imp.median().clamp_min(1e-8)

            importance[name] = imp
        return importance

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# =============================================================================
# 2. DARE (Difficulty-Weighted Drop And REscale)
# =============================================================================

@torch.no_grad()
def apply_dare_to_delta(
    delta: torch.Tensor,
    drop_rate: float,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if drop_rate <= 0:
        return delta
    mask = (torch.rand(delta.shape, device=delta.device, generator=generator) > drop_rate).to(delta.dtype)
    return (delta * mask) / (1.0 - drop_rate + eps)


@torch.no_grad()
def difficulty_weighted_dare_deltas(
    deltas: List[Dict[str, torch.Tensor]],
    memory_bank: torch.Tensor,
    difficulty_importance: Optional[torch.Tensor] = None,
    drop_rate: float = 0.3,
    difficulty_drop_factor: float = 0.6,
    seed: Optional[int] = None,
    eps: float = 1e-8,
) -> List[Dict[str, torch.Tensor]]:
    if difficulty_importance is None:
        difficulty_importance = torch.ones(len(deltas), device=memory_bank.device)
    importance = difficulty_importance / difficulty_importance.sum().clamp_min(eps)

    g = torch.Generator(device=deltas[0][next(iter(deltas[0]))].device)
    if seed is not None:
        g.manual_seed(seed)

    processed = []
    for i, d in enumerate(deltas):
        imp = float(importance[i].item())
        effective_drop = drop_rate * (1.0 - imp * (1.0 - difficulty_drop_factor))
        item = {}
        for k, delta in d.items():
            item[k] = apply_dare_to_delta(delta, effective_drop, generator=g, eps=eps)
        processed.append(item)
    return processed


# =============================================================================
# 3. TIES v2 — Per-Parameter Majority/Magnitude-Weighted Sign Election
# =============================================================================

@torch.no_grad()
def elect_sign_mask(
    deltas: List[torch.Tensor],
    expert_weights: torch.Tensor,
    mode: str = "majority",
    eps: float = 1e-8,
):
    """
    Elect a per-parameter sign across experts and return alignment masks.

    By default, the election uses a pure majority vote (weighted by
    ``expert_weights``) rather than the prior magnitude-weighted scheme.  The
    ``magnitude_weighted`` option remains available for research purposes but
    should not be confused with a true sign election, as it couples the
    vote to the absolute value of the deltas.

    Args:
        deltas: A list of tensors containing per-expert parameter updates.
        expert_weights: A tensor of shape (num_experts,) giving the weight of
            each expert in the vote.
        mode: Either ``"majority"`` for a sign-only vote or
            ``"magnitude_weighted"`` to weight votes by |delta|.  Default is
            ``"majority"``.
        eps: Unused; kept for API symmetry.

    Returns:
        elected_sign: A tensor of the elected sign per parameter (+1, -1 or 0).
        aligned_masks: A list of boolean tensors indicating, for each expert,
            which parameters match the elected sign and are non-zero in both
            the expert delta and the elected sign.
    """
    if mode not in {"majority", "magnitude_weighted"}:
        raise ValueError(f"Unknown sign election mode: {mode}")

    vote = torch.zeros_like(deltas[0])
    for i, d in enumerate(deltas):
        s = torch.sign(d)
        if mode == "majority":
            vote = vote + expert_weights[i] * s
        else:
            vote = vote + expert_weights[i] * s * d.abs()

    elected_sign = torch.sign(vote)
    aligned_masks = [
        (torch.sign(d) == elected_sign) & (d != 0) & (elected_sign != 0)
        for d in deltas
    ]
    return elected_sign, aligned_masks


# -----------------------------------------------------------------------------
# Sign-Consensus Weighted Sign Election
#
# The standard sign election zeroes out parameter updates that disagree with the
# majority sign or magnitude-weighted majority. However, parameters with weak
# consensus (where few experts agree) may be overly promoted or suppressed.
# ``elect_sign_mask_with_consensus`` extends the election procedure by
# computing a per-parameter consensus ratio: the fraction of active experts
# agreeing with the elected sign. This ratio can be used as a multiplicative
# weight during the Fisher merge to penalize parameters with high disagreement.

@torch.no_grad()
def elect_sign_mask_with_consensus(
    deltas: List[torch.Tensor],
    expert_weights: torch.Tensor,
    mode: str = "majority",
    consensus_scaling: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Elect a consensus sign across experts and compute consensus strength.

    Args:
        deltas: List of parameter delta tensors from each expert.
        expert_weights: Tensor of shape (num_experts,) giving importance weights.
        mode: ``"majority"`` or ``"magnitude_weighted"``. In majority mode,
            sign votes are weighted by expert_weights; in magnitude mode, votes
            are additionally scaled by |delta|.
        consensus_scaling: Whether to return the consensus_ratio tensor. When
            False, ``consensus_ratio`` will be a tensor of ones.
        eps: Small constant for numerical stability (unused here but kept for
            API symmetry).

    Returns:
        elected_sign: Tensor of the elected sign (+1, -1 or 0) per parameter.
        aligned_masks: Boolean masks for each expert indicating where the
            parameter agrees with the elected sign.
        consensus_ratio: Tensor of floats in [0, 1] indicating the fraction of
            active experts that agreed with the elected sign. When
            ``consensus_scaling`` is False, this tensor is filled with ones.
    """
    if mode not in {"majority", "magnitude_weighted"}:
        raise ValueError(f"Unknown sign election mode: {mode}")

    vote = torch.zeros_like(deltas[0])
    # Precompute sign matrix for all experts
    sign_matrix = torch.stack([torch.sign(d) for d in deltas], dim=0)  # (E, ...)

    for i, d in enumerate(deltas):
        s = sign_matrix[i]
        if mode == "majority":
            vote = vote + expert_weights[i] * s
        else:
            vote = vote + expert_weights[i] * s * d.abs()

    elected_sign = torch.sign(vote)
    aligned_masks = [
        (torch.sign(d) == elected_sign) & (d != 0) & (elected_sign != 0)
        for d in deltas
    ]

    # Compute consensus ratio if requested
    if consensus_scaling:
        # Count how many experts have non-zero sign for each parameter
        active_experts = (sign_matrix != 0).sum(dim=0).clamp_min(1)
        # Count how many experts agree with the elected sign
        agreements = (sign_matrix == elected_sign.unsqueeze(0)).sum(dim=0)
        consensus_ratio = agreements.float() / active_experts.float()
    else:
        consensus_ratio = torch.ones_like(elected_sign, dtype=torch.float32)

    return elected_sign, aligned_masks, consensus_ratio


@torch.no_grad()
def compute_ties_aligned_deltas(
    experts: List[nn.Module],
    memory_bank: torch.Tensor,
    difficulty_importance: Optional[torch.Tensor] = None,
    trim_ratio: float = 0.2,
    difficulty_trim_factor: float = 0.7,
    precomputed_deltas: Optional[List[Dict[str, torch.Tensor]]] = None,
    eps: float = 1e-8,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]], torch.Tensor]:
    """
    Difficulty-weighted TIES trimming + sign election.

    Returns per-expert ALIGNED DELTAS (still one dict per expert — zeroed out
    wherever an expert failed trimming or sign alignment), NOT a pre-reduced
    single delta. Keeping per-expert granularity here is what lets a
    downstream Fisher-weighted merge (or any other reduction) see every
    expert's contribution; collapsing to a single delta at this stage was
    an earlier bug (the "TIES -> Fisher pipeline" bug: with only one delta
    left, a subsequent Fisher merge would silently use only
    `memory_bank[0]` and one Fisher diagonal, discarding every other
    expert).
    """
    if len(experts) == 0:
        raise ValueError("No experts provided")
    first = experts[0]
    is_swiglu = isinstance(first, SwiGLUFFN)

    if difficulty_importance is None:
        difficulty_importance = torch.ones(len(experts), device=memory_bank.device)
    importance = difficulty_importance / difficulty_importance.sum().clamp_min(eps)

    def get_weights(expert):
        if is_swiglu:
            d = {"up.weight": expert.up.weight.data}
            if expert.up.bias is not None:
                d["up.bias"] = expert.up.bias.data
            if expert.gate is not None:
                d["gate.weight"] = expert.gate.weight.data
                if expert.gate.bias is not None:
                    d["gate.bias"] = expert.gate.bias.data
            d["down.weight"] = expert.down.weight.data
            if expert.down.bias is not None:
                d["down.bias"] = expert.down.bias.data
            return d
        else:
            d = {"up.weight": expert[0].weight.data}
            if expert[0].bias is not None:
                d["up.bias"] = expert[0].bias.data
            down_idx = 3 if len(expert) > 3 else -1
            d["down.weight"] = expert[down_idx].weight.data
            if expert[down_idx].bias is not None:
                d["down.bias"] = expert[down_idx].bias.data
            return d

    base = {}
    for k in get_weights(first).keys():
        base[k] = sum(w * get_weights(e)[k] for w, e in zip(memory_bank, experts))

    aligned_deltas = []
    for i, expert in enumerate(experts):
        w = get_weights(expert)
        item = {}
        for param_name in base.keys():
            if precomputed_deltas is not None:
                delta = precomputed_deltas[i][param_name]
            else:
                delta = w[param_name] - base[param_name]
            imp = float(importance[i].item())
            effective_trim = trim_ratio * (1 - imp * (1 - difficulty_trim_factor))
            keep_count = int((1 - effective_trim) * delta.numel())
            if keep_count > 0:
                thresh = torch.topk(delta.abs().flatten(), keep_count).values.min()
                delta = delta * (delta.abs() >= thresh)
            item[param_name] = delta
        aligned_deltas.append(item)

    for param_name in base.keys():
        param_deltas = [d[param_name] for d in aligned_deltas]
        vote = torch.zeros_like(base[param_name])
        for i, delta in enumerate(param_deltas):
            w = memory_bank[i] * importance[i]
            # Pure sign-majority voting: decouple the consensus from raw delta
            # magnitudes so large outliers can't swamp the election.
            # (sign(delta) * delta.abs() == delta, which collapses the vote
            # into a weighted sum of deltas — the magnitude-weighted bug.)
            vote = vote + w * torch.sign(delta)
        elected_sign = torch.sign(vote)
        for i in range(len(aligned_deltas)):
            delta = aligned_deltas[i][param_name]
            aligned = (torch.sign(delta) == elected_sign) & (delta != 0) & (elected_sign != 0)
            aligned_deltas[i][param_name] = delta * aligned.to(delta.dtype)

    return base, aligned_deltas, importance


# =============================================================================
# 4. FISHER-WEIGHTED MERGE (with group boosting)
# =============================================================================

@torch.no_grad()
def difficulty_weighted_fisher_merge_from_deltas(
    deltas: List[Dict[str, torch.Tensor]],
    memory_bank: torch.Tensor,
    fisher_diagonals: List[Dict[str, torch.Tensor]],
    difficulty_importance: Optional[torch.Tensor] = None,
    fisher_power: float = 1.0,
    fisher_floor: float = 1e-8,
    sensitive_patterns: Optional[List[str]] = None,
    sensitive_fisher_boost: float = 1.0,
    ssm_fisher_boost: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Elementwise Fisher merge with optional boosting for sensitive / SSM
    parameter groups. Expects `len(deltas) == memory_bank.numel()` (i.e.
    per-expert deltas, not a pre-collapsed single delta) — see
    `compute_ties_aligned_deltas` docstring for why that invariant matters.
    """
    assert len(deltas) == memory_bank.numel(), (
        f"difficulty_weighted_fisher_merge_from_deltas expected one delta per "
        f"expert ({memory_bank.numel()}), got {len(deltas)}. If a prior stage "
        f"collapsed per-expert deltas into a single merged delta before "
        f"reaching Fisher, every expert but one is being silently discarded."
    )
    if difficulty_importance is None:
        difficulty_importance = torch.ones(len(deltas), device=memory_bank.device)
    difficulty = difficulty_importance / difficulty_importance.sum().clamp_min(eps)

    if sensitive_patterns is None:
        sensitive_patterns = ["up", "gate"]

    keys = list(deltas[0].keys())
    merged = {}

    for k in keys:
        is_sensitive = any(pat in k.lower() for pat in sensitive_patterns)
        is_ssm = _is_ssm_core_param(k)
        effective_power = fisher_power * (sensitive_fisher_boost if is_sensitive else 1.0) * (ssm_fisher_boost if is_ssm else 1.0)

        numer = torch.zeros_like(deltas[0][k])
        # initialise denom without eps; eps added at the end to avoid divide by zero
        denom = torch.zeros_like(deltas[0][k])

        for i in range(len(deltas)):
            f = fisher_diagonals[i][k].to(deltas[i][k].device).view_as(deltas[i][k])
            f = f.clamp_min(fisher_floor)

            if effective_power != 1.0:
                f = f.pow(effective_power)

            w = memory_bank[i] * difficulty[i]
            contrib = w * f
            delta = deltas[i][k]
            active_mask = (delta != 0).to(delta.dtype)
            numer = numer + contrib * delta
            denom = denom + contrib * active_mask

        merged[k] = numer / denom.clamp_min(eps)

    return merged


# =============================================================================
# 5. MAMBA GROUPED MERGE
# =============================================================================

@dataclass
class GroupMergePolicy:
    drop_rate: float = 0.3
    difficulty_drop_factor: float = 0.6
    # Default sign mode is pure majority.  Researchers can override to
    # "magnitude_weighted", but majority is mathematically the correct
    # interpretation of a sign election.
    sign_mode: str = "majority"
    use_sign_election: bool = True
    retain_unanimous_only: bool = False
    fisher_power: float = 1.0
    fisher_floor: float = 1e-8


DEFAULT_MAMBA_POLICIES: Dict[str, GroupMergePolicy] = {
    # sign_mode="majority" for every group: sign · |Δ| under magnitude_weighted
    # collapses to the raw delta and lets large-magnitude outliers swamp the
    # election.  Majority is the mathematically correct interpretation of a
    # sign election.  Researchers may still override to "magnitude_weighted"
    # for ablations, but the shipped default must be majority.
    "in_proj":  GroupMergePolicy(drop_rate=0.30, sign_mode="majority",
                                  retain_unanimous_only=False, fisher_power=1.0),
    "out_proj": GroupMergePolicy(drop_rate=0.30, sign_mode="majority",
                                  retain_unanimous_only=False, fisher_power=1.0),
    "x_proj":   GroupMergePolicy(drop_rate=0.30, sign_mode="majority",
                                  retain_unanimous_only=False, fisher_power=1.0),
    "dt_proj":  GroupMergePolicy(drop_rate=0.15, sign_mode="majority",
                                  retain_unanimous_only=False, fisher_power=1.5),
    "conv1d":   GroupMergePolicy(drop_rate=0.15, sign_mode="majority",
                                  retain_unanimous_only=False, fisher_power=1.5),
    "A_log":    GroupMergePolicy(drop_rate=0.05, sign_mode="majority",
                                  retain_unanimous_only=True, fisher_power=2.0),
    "D":        GroupMergePolicy(drop_rate=0.05, sign_mode="majority",
                                  retain_unanimous_only=True, fisher_power=2.0),
}


def _lookup_group_policy(policies: Dict[str, GroupMergePolicy], param_name: str) -> GroupMergePolicy:
    """
    Resolve the merge policy for a parameter name.

    When the upgrade utilities are available (``lookup_group_policy_robust``),
    this function delegates to that robust resolver. Otherwise, it falls back
    to a normalized substring matching strategy that prioritizes exact matches,
    parent module matches, and longest contained substrings. If no match is
    found, a warning is issued and a default ``GroupMergePolicy`` is returned.
    """
    # Use the robust resolver from upgrade_utils if it is available
    if lookup_group_policy_robust is not None:
        return lookup_group_policy_robust(
            policies=policies,
            param_name=param_name,
            default_factory=GroupMergePolicy,
        )

    # ----------------------------------------------------------------------
    # Fallback implementation (unchanged from v4.2)
    # ----------------------------------------------------------------------
    if param_name in policies:
        return policies[param_name]

    parts = param_name.split(".")
    parent_name = parts[-2] if len(parts) > 1 else ""
    norm_name = parts[-1]  # leaf component (e.g. weight, bias)

    if parent_name and parent_name in policies:
        return policies[parent_name]

    candidates = [key for key in policies if key in param_name]
    if candidates:
        candidates.sort(key=len, reverse=True)
        return policies[candidates[0]]

    warnings.warn(
        f"Parameter '{param_name}' could not be matched to any policy group. "
        f"Available policy keys: {list(policies.keys())}. Falling back to default GroupMergePolicy.",
        stacklevel=2,
    )
    return GroupMergePolicy()



@torch.no_grad()
def _validate_policies_against_keys(
    policies: Optional[Dict[str, GroupMergePolicy]],
    keys: Iterable[str],
    context: str = "",
) -> None:
    """
    Cross-checks a Mamba merge policies dict against the actual parameter
    keys it will be applied to, using the same substring-match resolution
    rule as `_lookup_group_policy` (so warnings reflect what will actually
    happen at merge time, not a stricter check that would cry wolf about
    e.g. "in_proj" vs "in_proj.weight" resolving just fine).

    This does NOT raise — an unresolved key falls back to
    `GroupMergePolicy()` defaults, which is valid — but it is almost always
    unintentional, so it's surfaced as a warning.
    """
    if policies is None:
        return
    keys_set = set(keys)
    policy_keys = set(policies.keys())

    unmatched_policy_keys = {pk for pk in policy_keys if not any(pk in k for k in keys_set)}
    if unmatched_policy_keys:
        warnings.warn(
            f"{context}Policy keys {sorted(unmatched_policy_keys)} do not match any "
            f"parameter name in this expert's state dict {sorted(keys_set)}. "
            f"These policies will never be applied (dead config entries).",
            stacklevel=2,
        )

    unresolved_param_keys = {k for k in keys_set if not any(pk in k for pk in policy_keys)}
    if unresolved_param_keys:
        warnings.warn(
            f"{context}Parameter(s) {sorted(unresolved_param_keys)} do not match any "
            f"policy key and will fall back to the default GroupMergePolicy(). "
            f"If this is unintentional (e.g. a typo'd key), fix the policies dict.",
            stacklevel=2,
        )


@torch.no_grad()
def mamba_grouped_aligned_deltas(
    expert_params: List[Dict[str, torch.Tensor]],
    memory_bank: torch.Tensor,
    difficulty_importance: Optional[torch.Tensor] = None,
    policies: Optional[Dict[str, GroupMergePolicy]] = None,
    seed: Optional[int] = None,
    eps: float = 1e-8,
    _validate: bool = True,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
    if difficulty_importance is None:
        difficulty_importance = torch.ones(len(expert_params), device=memory_bank.device)
    policies = policies or DEFAULT_MAMBA_POLICIES
    keys = list(expert_params[0].keys())

    if _validate:
        _validate_policies_against_keys(policies, keys, context="[mamba_grouped_aligned_deltas] ")

    base = {k: sum(w * p[k] for w, p in zip(memory_bank, expert_params)) for k in keys}
    deltas_by_group = {k: [p[k] - base[k] for p in expert_params] for k in keys}
    diff = difficulty_importance / difficulty_importance.sum().clamp_min(eps)
    expert_weights = memory_bank * diff

    aligned_deltas = [{} for _ in expert_params]
    for k in keys:
        policy = _lookup_group_policy(policies, k)
        deltas = deltas_by_group[k]
        g = torch.Generator(device=deltas[0].device)
        if seed is not None:
            g.manual_seed(seed + (zlib.crc32(k.encode()) & 0x7fffffff))
        dare_deltas = []
        for i, d in enumerate(deltas):
            imp = float(diff[i].item())
            effective_drop = policy.drop_rate * (1.0 - imp * (1.0 - policy.difficulty_drop_factor))
            dare_deltas.append(apply_dare_to_delta(d, effective_drop, generator=g, eps=eps))
        if policy.use_sign_election:
            _, aligned_masks = elect_sign_mask(dare_deltas, expert_weights, mode=policy.sign_mode, eps=eps)
        else:
            aligned_masks = [torch.ones_like(dare_deltas[0], dtype=torch.bool) for _ in dare_deltas]
        if policy.retain_unanimous_only:
            stacked = torch.stack(aligned_masks, dim=0)
            unanimous = stacked.all(dim=0)
            aligned_masks = [m & unanimous for m in aligned_masks]
        for i in range(len(expert_params)):
            aligned_deltas[i][k] = dare_deltas[i] * aligned_masks[i].to(dare_deltas[i].dtype)
    return base, aligned_deltas


@torch.no_grad()
def mamba_fisher_merge_from_aligned(
    base: Dict[str, torch.Tensor],
    aligned_deltas: List[Dict[str, torch.Tensor]],
    fishers: List[Dict[str, torch.Tensor]],
    memory_bank: torch.Tensor,
    difficulty_importance: Optional[torch.Tensor] = None,
    policies: Optional[Dict[str, GroupMergePolicy]] = None,
    ssm_fisher_boost: float = 1.0,
    eps: float = 1e-8,
    _validate: bool = True,
) -> Dict[str, torch.Tensor]:
    if difficulty_importance is None:
        difficulty_importance = torch.ones(len(aligned_deltas), device=memory_bank.device)
    policies = policies or DEFAULT_MAMBA_POLICIES

    if _validate:
        _validate_policies_against_keys(policies, base.keys(), context="[mamba_fisher_merge_from_aligned] ")

    diff = difficulty_importance / difficulty_importance.sum().clamp_min(eps)
    expert_weights = memory_bank * diff
    merged = {}
    for k in base.keys():
        policy = _lookup_group_policy(policies, k)
        is_ssm = _is_ssm_core_param(k)
        effective_power = policy.fisher_power * (ssm_fisher_boost if is_ssm else 1.0)
        numer = torch.zeros_like(base[k])
        # build denom without eps; eps added at clamp time
        denom = torch.zeros_like(base[k])
        for i in range(len(aligned_deltas)):
            delta = aligned_deltas[i][k]
            f = fishers[i][k].to(delta.device).clamp_min(policy.fisher_floor)
            if effective_power != 1.0:
                f = f.pow(effective_power)
            contrib = expert_weights[i] * f
            active_mask = (delta != 0).to(delta.dtype)
            numer = numer + contrib * delta
            denom = denom + contrib * active_mask
        merged[k] = base[k] + (numer / denom.clamp_min(eps))
    return merged


@torch.no_grad()
def mamba_grouped_merge(
    expert_params: List[Dict[str, torch.Tensor]],
    fishers: List[Dict[str, torch.Tensor]],
    memory_bank: torch.Tensor,
    difficulty_importance: Optional[torch.Tensor] = None,
    policies: Optional[Dict[str, GroupMergePolicy]] = None,
    ssm_fisher_boost: float = 1.0,
    seed: Optional[int] = None,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    `ssm_fisher_boost` is threaded through end-to-end here (fix: an earlier
    draft dropped it from this wrapper's own signature, so it silently fell
    back to the inner function's default of 1.0 no matter what the caller
    passed in). Policy validation happens once at this top level; the two
    inner calls skip their own validation (`_validate=False`) to avoid
    duplicate warnings for the same `policies` dict on one logical call.
    """
    resolved_policies = policies or DEFAULT_MAMBA_POLICIES
    _validate_policies_against_keys(
        resolved_policies, expert_params[0].keys(), context="[mamba_grouped_merge] "
    )
    base, aligned_deltas = mamba_grouped_aligned_deltas(
        expert_params, memory_bank, difficulty_importance, resolved_policies, seed, eps,
        _validate=False,
    )
    return mamba_fisher_merge_from_aligned(
        base, aligned_deltas, fishers, memory_bank, difficulty_importance,
        resolved_policies, ssm_fisher_boost=ssm_fisher_boost, eps=eps,
        _validate=False,
    )


@torch.no_grad()
def apply_merged_mamba_params(mamba_module: nn.Module, merged: Dict[str, torch.Tensor]):
    param_map = {}
    for name, param in mamba_module.named_parameters():
        if name in merged:
            param_map[name] = param
            continue
        for key in merged:
            if key in name or name in key:
                param_map[key] = param
                break
    for k, tensor_ in merged.items():
        if k in param_map:
            param_map[k].data.copy_(tensor_.to(param_map[k].device))
        else:
            warnings.warn(f"Parameter {k} not found in target Mamba module; skipping.")


# =============================================================================
# 6. K-FAC SCORE INCORPORATION
# =============================================================================

@torch.no_grad()
def incorporate_kfac_scores(
    memory_bank: torch.Tensor,
    kfac_scores: Dict[str, float],
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if len(kfac_scores) == 0:
        return memory_bank
    scores = torch.tensor(list(kfac_scores.values()), device=memory_bank.device, dtype=memory_bank.dtype)
    scores = scores / scores.mean().clamp_min(eps)
    scores = torch.softmax(scores / max(temperature, eps), dim=0)
    if scores.numel() != memory_bank.numel():
        raise ValueError(
            f"K-FAC score count ({scores.numel()}) must match expert count "
            f"({memory_bank.numel()}). Aggregate layer-level K-FAC scores into "
            f"per-expert scores before calling incorporate_kfac_scores()."
        )
    modulated = memory_bank * scores
    return modulated / modulated.sum().clamp_min(eps)


# =============================================================================
# 6a. AGGREGATE LAYER-LEVEL K-FAC SCORES TO EXPERT LEVEL
# =============================================================================

def aggregate_kfac_scores_to_experts(
    layer_scores: Dict[str, float],
    num_experts: int,
    path_prefix: str = "ffn_path",
) -> Dict[str, float]:
    """
    Aggregate layer-level K-FAC scores into expert-level scores.

    Many ExFusion models expose expert parameters under hierarchical names such
    as ``ffn_path.experts.0.up.weight``. ``KFACFisherTracker`` reports
    layer-level scores keyed by the full parameter path. However,
    ``incorporate_kfac_scores`` expects one score per expert. This utility
    parses the layer names to determine which expert index they belong to,
    accumulates their scores, and normalizes by the number of matched layers
    per expert.

    Args:
        layer_scores: A dict mapping ``str`` layer names to scalar K-FAC scores.
        num_experts:  Total number of experts in the FFN/Mamba path.
        path_prefix:  Prefix used to identify expert layers (default ``"ffn_path"``).

    Returns:
        A dict mapping expert identifiers (``"expert_0"`` ...) to aggregated
        average scores.
    """
    # Initialize accumulators
    expert_sums: Dict[str, float] = {f"expert_{i}": 0.0 for i in range(num_experts)}
    expert_counts: Dict[str, int] = {f"expert_{i}": 0 for i in range(num_experts)}

    for layer_name, score in layer_scores.items():
        # Only consider layers under the given path_prefix
        if path_prefix not in layer_name:
            continue
        # Match patterns like ".experts.0." or ".experts[0]."
        for i in range(num_experts):
            dot_pattern = f".experts.{i}."
            bracket_pattern = f".experts[{i}]."
            if dot_pattern in layer_name or bracket_pattern in layer_name:
                key = f"expert_{i}"
                expert_sums[key] += score
                expert_counts[key] += 1
                break

    # Normalize sums to average scores per expert
    final_scores: Dict[str, float] = {}
    for i in range(num_experts):
        key = f"expert_{i}"
        count = max(expert_counts[key], 1)
        final_scores[key] = expert_sums[key] / count
    return final_scores


# =============================================================================
# 7. FISHER DIAGONAL BUILDER
# =============================================================================

def build_fisher_diagonals(
    experts: List[nn.Module],
    dataloader: "torch.utils.data.DataLoader",
    model: nn.Module,
    loss_fn: Callable,
    num_batches: int = 8,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Compute squared-gradient Fisher diagonals for any list of experts.

    Fixes applied here:
      - `model_kwargs` lets callers pass extra forward arguments (e.g.
        `theta_t`, `training=True`) that the target model's forward may
        require; a prior draft always called `model(inputs)` with no way to
        supply these, causing a TypeError for any model whose forward needs
        more than one positional argument.
      - This function must NOT be wrapped in `@torch.no_grad()` at the top
        level: an earlier draft decorated the whole function with
        `@torch.no_grad()` while calling `loss.backward()` inside it. Under
        `no_grad`, the forward pass builds no autograd graph, so
        `.backward()` raises ("element 0 of tensors does not require grad
        and does not have a grad_fn"). Only the Fisher-accumulation
        bookkeeping (which touches `.grad.data`, already detached) runs
        under `no_grad`; the forward/backward pass itself runs under
        `torch.enable_grad()` so it works correctly even if this function is
        called from inside an outer `no_grad` context (e.g. a calibration
        loop that wraps everything in `no_grad` by habit).
    """
    if model_kwargs is None:
        model_kwargs = {}

    with torch.no_grad():
        fishers = []
        for expert in experts:
            expert_fisher = {}
            for name, param in expert.named_parameters():
                if param.requires_grad:
                    expert_fisher[name] = torch.zeros_like(param.data)
            fishers.append(expert_fisher)

    model.train()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        inputs, targets = None, None
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                inputs, targets = batch
            elif len(batch) == 1:
                inputs = batch[0]
                targets = batch[0]
            else:
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else batch[0]
        elif isinstance(batch, dict):
            inputs = batch.get("input_ids", batch.get("inputs", batch.get("hidden_states", batch)))
            targets = batch.get("labels", batch.get("targets", inputs))
        else:
            inputs = batch
            targets = batch

        model.zero_grad()
        with torch.enable_grad():
            outputs = model(inputs, **model_kwargs)
            loss = loss_fn(outputs, targets)
            loss.backward()

        with torch.no_grad():
            for idx, expert in enumerate(experts):
                for name, param in expert.named_parameters():
                    if param.grad is not None:
                        fishers[idx][name] += param.grad.data.pow(2)

    with torch.no_grad():
        for idx in range(len(experts)):
            for k in fishers[idx]:
                fishers[idx][k] = fishers[idx][k] / max(num_batches, 1)

    return fishers


# =============================================================================
# 8. MEMORY BANK EXFUSION FFN
# =============================================================================

class MemoryBankExFusionFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        router_hidden_size: Optional[int] = None,
        momentum: float = 0.95,
        momentum_scheduler: Optional[Any] = None,
        difficulty_method: Literal["entropy", "max_prob", "variance"] = "entropy",
        difficulty_aggregation: Literal["mean", "max", "percentile"] = "percentile",
        activation: Literal["silu", "swiglu"] = "silu",
        bias: bool = False,
        dropout: float = 0.0,
        sensitive_fisher_boost: float = 1.5,
        sensitive_patterns: Optional[List[str]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.momentum = momentum
        self.momentum_scheduler = momentum_scheduler
        self.difficulty_method = difficulty_method
        self.difficulty_aggregation = difficulty_aggregation
        self.activation = activation
        self.sensitive_fisher_boost = sensitive_fisher_boost
        self.sensitive_patterns = sensitive_patterns or ["up", "gate"]

        router_dim = router_hidden_size or hidden_size // 4
        self.router = nn.Linear(hidden_size, router_dim)
        self.router_out = nn.Linear(router_dim, num_experts)

        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            if activation == "swiglu":
                up = nn.Linear(hidden_size, intermediate_size, bias=bias)
                gate = nn.Linear(hidden_size, intermediate_size, bias=bias)
                down = nn.Linear(intermediate_size, hidden_size, bias=bias)
                expert = SwiGLUFFN(up, gate, down)
            else:
                expert = nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size, bias=bias),
                    nn.SiLU(),
                    nn.Linear(intermediate_size, hidden_size, bias=bias),
                )
                if dropout > 0:
                    expert = nn.Sequential(expert[0], expert[1], nn.Dropout(dropout), expert[2])
            self.experts.append(expert)

        self.register_buffer("memory_bank", torch.ones(num_experts) / num_experts)
        self.register_buffer("step_count", torch.tensor(0))

        self.is_merged = False
        self.merged_ffn = None
        self.last_token_difficulties = None
        self.last_batch_difficulty = 0.5

    def forward(self, x: torch.Tensor, macro_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.is_merged:
            return self.merged_ffn(x)

        router_hidden = F.relu(self.router(x))
        router_logits = self.router_out(router_hidden)

        token_diff = self._compute_token_difficulty(x, router_logits)
        self.last_token_difficulties = token_diff.detach()

        if self.difficulty_aggregation == "mean":
            batch_diff = token_diff.mean().detach()
        elif self.difficulty_aggregation == "max":
            batch_diff = token_diff.max().detach()
        else:
            batch_diff = torch.quantile(token_diff, 0.9).detach()
        self.last_batch_difficulty = batch_diff.item()  # only for logging

        router_weights = F.softmax(router_logits, dim=-1).mean(dim=(0, 1))

        if self.momentum_scheduler is not None:
            current_momentum = self.momentum_scheduler.get_momentum(
                self.step_count.item(), self.last_batch_difficulty
            )
        else:
            current_momentum = self.momentum

        if macro_context is not None:
            context_mod = torch.sigmoid(macro_context.mean()) * 0.1
            current_momentum = current_momentum * (1 - context_mod) + 0.85 * context_mod

        with torch.no_grad():
            self.memory_bank = (
                current_momentum * self.memory_bank
                + (1 - current_momentum) * router_weights
            )
            self.memory_bank = self.memory_bank / self.memory_bank.sum()
            self.step_count += 1

        fused = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            fused = fused + self.memory_bank[i] * expert_out

        return fused

    def _compute_token_difficulty(self, hidden, router_logits):
        if self.difficulty_method == "entropy":
            probs = F.softmax(router_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_ent = math.log(router_logits.size(-1))
            return (entropy / max_ent).clamp(0, 1)
        elif self.difficulty_method == "max_prob":
            probs = F.softmax(router_logits, dim=-1)
            return (1 - probs.max(dim=-1).values).clamp(0, 1)
        else:
            var = hidden.var(dim=-1)
            return torch.sigmoid(var / (var.mean() + 1e-8)).clamp(0, 1)

    def _get_weights(self, expert):
        is_swiglu = isinstance(expert, SwiGLUFFN)
        if is_swiglu:
            d = {"up.weight": expert.up.weight.data}
            if expert.up.bias is not None:
                d["up.bias"] = expert.up.bias.data
            if expert.gate is not None:
                d["gate.weight"] = expert.gate.weight.data
                if expert.gate.bias is not None:
                    d["gate.bias"] = expert.gate.bias.data
            d["down.weight"] = expert.down.weight.data
            if expert.down.bias is not None:
                d["down.bias"] = expert.down.bias.data
            return d
        d = {"up.weight": expert[0].weight.data}
        if expert[0].bias is not None:
            d["up.bias"] = expert[0].bias.data
        down_idx = 3 if len(expert) > 3 else -1
        d["down.weight"] = expert[down_idx].weight.data
        if expert[down_idx].bias is not None:
            d["down.bias"] = expert[down_idx].bias.data
        return d

    @torch.no_grad()
    def merge_to_dense(
        self,
        pipeline: List[Literal["dare", "ties", "fisher", "kfac"]] = ["dare", "ties", "fisher"],
        dare_drop_rate: float = 0.3,
        dare_difficulty_factor: float = 0.6,
        ties_trim_ratio: float = 0.2,
        ties_difficulty_trim_factor: float = 0.7,
        fisher_diagonals: Optional[List[Dict[str, torch.Tensor]]] = None,
        kfac_scores: Optional[Dict[str, float]] = None,
        kfac_temperature: float = 1.0,
        difficulty_importance: Optional[torch.Tensor] = None,
        fisher_power: float = 1.0,
        sensitive_fisher_boost: Optional[float] = None,
        fisher_floor: float = 1e-8,
        eps: float = 1e-8,
        seed: Optional[int] = None,
    ):
        """
        Fix history (this is the method that saw the most churn across the
        source debugging session):

          - K-FAC used to run as a pipeline "stage" interleaved with
            dare/ties/fisher. If it ran after "fisher" (or after any stage
            that had already collapsed per-expert deltas down to one merged
            delta), updating `self.memory_bank` at that point had nothing
            left to re-weight — the merge was already computed with the old
            memory bank. The fix: resolve K-FAC into a cloned
            `active_memory_bank` *before* base/deltas are even computed, so
            every downstream stage (including "fisher") already sees the
            K-FAC-modulated weights. "kfac" is no longer a stage in the
            dare/ties/fisher loop.
          - Pipeline is validated eagerly (unknown stage names raise
            immediately, duplicates warn) before any merge work runs.
          - The final combination, when the pipeline does not end on
            "fisher" (so `current_deltas` is still one dict per expert
            rather than a single collapsed delta), now weights by a
            normalized `active_memory_bank * difficulty_importance` instead
            of `active_memory_bank` alone — difficulty-awareness no longer
            silently disappears at the last step for pipelines like
            `["dare", "ties"]`.
        """
        if self.is_merged:
            return

        _VALID_STAGES = {"dare", "ties", "fisher", "kfac"}
        _unknown_stages = set(pipeline) - _VALID_STAGES
        if _unknown_stages:
            raise ValueError(
                f"Unknown merge stage(s) {sorted(_unknown_stages)} in pipeline. "
                f"Valid stages are: {sorted(_VALID_STAGES)}."
            )
        if len(pipeline) != len(set(pipeline)):
            warnings.warn(
                f"Duplicate stages detected in pipeline {pipeline}; "
                f"each stage will run once per occurrence, which may be unintended.",
                stacklevel=2,
            )

        device = self.memory_bank.device
        if difficulty_importance is None:
            difficulty_importance = torch.ones(self.num_experts, device=device)
        else:
            difficulty_importance = difficulty_importance.to(device)

        if sensitive_fisher_boost is None:
            sensitive_fisher_boost = self.sensitive_fisher_boost

        is_swiglu = isinstance(self.experts[0], SwiGLUFFN)
        has_gate = is_swiglu and self.experts[0].gate is not None

        # Resolve K-FAC modulation up front so every downstream weighted
        # operation (including "fisher") uses the correct expert weights
        # before any per-expert deltas collapse.
        active_memory_bank = self.memory_bank.detach().clone().to(device)
        if "kfac" in pipeline:
            if kfac_scores is None:
                raise ValueError("kfac_scores required when 'kfac' is included in pipeline.")
            active_memory_bank = incorporate_kfac_scores(
                active_memory_bank, kfac_scores, temperature=kfac_temperature, eps=eps
            )
        stage_pipeline = [stage for stage in pipeline if stage != "kfac"]

        base = {}
        first_weights = self._get_weights(self.experts[0])
        for k in first_weights.keys():
            base[k] = sum(w * self._get_weights(e)[k] for w, e in zip(active_memory_bank, self.experts))

        raw_deltas = []
        for e in self.experts:
            w = self._get_weights(e)
            raw_deltas.append({k: w[k] - base[k] for k in w.keys()})

        current_deltas = raw_deltas

        for stage in stage_pipeline:
            if stage == "dare":
                current_deltas = difficulty_weighted_dare_deltas(
                    deltas=current_deltas,
                    memory_bank=active_memory_bank,
                    difficulty_importance=difficulty_importance,
                    drop_rate=dare_drop_rate,
                    difficulty_drop_factor=dare_difficulty_factor,
                    seed=seed,
                    eps=eps,
                )

            elif stage == "ties":
                _, current_deltas, _ = compute_ties_aligned_deltas(
                    experts=self.experts,
                    memory_bank=active_memory_bank,
                    difficulty_importance=difficulty_importance,
                    trim_ratio=ties_trim_ratio,
                    difficulty_trim_factor=ties_difficulty_trim_factor,
                    precomputed_deltas=current_deltas,
                    eps=eps,
                )

            elif stage == "fisher":
                if fisher_diagonals is None:
                    raise ValueError("fisher_diagonals required for 'fisher' stage in FFN merge.")
                merged_delta = difficulty_weighted_fisher_merge_from_deltas(
                    deltas=current_deltas,
                    memory_bank=active_memory_bank,
                    fisher_diagonals=fisher_diagonals,
                    difficulty_importance=difficulty_importance,
                    fisher_power=fisher_power,
                    fisher_floor=fisher_floor,
                    sensitive_fisher_boost=sensitive_fisher_boost,
                    sensitive_patterns=self.sensitive_patterns,
                    eps=eps,
                )
                current_deltas = [merged_delta]
            # No else branch needed: pipeline validated eagerly above.

        if len(current_deltas) == 1:
            merged_delta = current_deltas[0]
        else:
            combine_weights = active_memory_bank * difficulty_importance
            combine_weights = combine_weights / combine_weights.sum().clamp_min(eps)
            merged_delta = {}
            for k in base.keys():
                merged_delta[k] = sum(
                    combine_weights[i] * current_deltas[i][k]
                    for i in range(len(current_deltas))
                )

        weight_keys = [k for k in merged_delta.keys() if k.endswith(".weight")]
        bias_map = {k.replace(".bias", ""): k for k in merged_delta.keys() if k.endswith(".bias")}
        merged_layers = {}
        for wkey in weight_keys:
            stem = wkey.replace(".weight", "")
            final = (base[wkey] + merged_delta[wkey]).to(device)
            has_bias = stem in bias_map
            layer = nn.Linear(final.shape[1], final.shape[0], bias=has_bias).to(device)
            layer.weight.data.copy_(final)
            if has_bias:
                bkey = bias_map[stem]
                layer.bias.data.copy_((base[bkey] + merged_delta[bkey]).to(device))
            merged_layers[stem] = layer

        if is_swiglu:
            self.merged_ffn = SwiGLUFFN(
                merged_layers["up"],
                merged_layers.get("gate", None),
                merged_layers["down"],
            ).to(device)
        else:
            self.merged_ffn = nn.Sequential(
                merged_layers["up"],
                nn.SiLU(),
                merged_layers["down"],
            ).to(device)

        self.memory_bank.copy_(active_memory_bank)
        self.is_merged = True


# =============================================================================
# 9. MEMORY BANK EXFUSION MAMBA  (new: not present in either source draft)
# =============================================================================

class MemoryBankExFusionMamba(nn.Module):
    """
    Mamba analogue of `MemoryBankExFusionFFN`.

    Design note: neither source draft actually implemented this class, even
    though it was listed in this module's own docstring/import example in
    every revision. The Mamba block architecture itself is external/
    user-defined (as in the toolkit's own example usage, which takes a
    `mamba_factory` callable), so this class is a thin difficulty-aware
    memory-bank wrapper around N instances produced by `block_factory`, and
    delegates the actual grouped merge math to `mamba_grouped_merge`
    (Section 5), which *is* battle-tested across the source debugging
    session.

    Unlike the FFN class, Mamba merging is not staged as
    dare -> ties -> fisher; `mamba_grouped_aligned_deltas` already applies
    per-group DARE + sign election in one pass, and
    `mamba_fisher_merge_from_aligned` always does the Fisher-weighted
    reduction. K-FAC (if used) is folded into `active_memory_bank` before
    that single call, for the same reason it must precede "fisher" in the
    FFN pipeline: once the grouped merge collapses to a dense parameter
    dict, there is no per-expert information left to re-weight.
    """

    def __init__(
        self,
        block_factory: Callable[[], nn.Module],
        num_experts: int = 4,
        router_hidden_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        momentum: float = 0.95,
        momentum_scheduler: Optional[Any] = None,
        difficulty_method: Literal["entropy", "max_prob", "variance"] = "entropy",
        difficulty_aggregation: Literal["mean", "max", "percentile"] = "percentile",
        ssm_fisher_boost: float = 1.5,
        policies: Optional[Dict[str, GroupMergePolicy]] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.momentum = momentum
        self.momentum_scheduler = momentum_scheduler
        self.difficulty_method = difficulty_method
        self.difficulty_aggregation = difficulty_aggregation
        self.ssm_fisher_boost = ssm_fisher_boost
        self.policies = policies or DEFAULT_MAMBA_POLICIES

        self.experts = nn.ModuleList([block_factory() for _ in range(num_experts)])

        # A router needs *some* feature dimension to look at; if the caller
        # didn't specify hidden_size explicitly, infer it from the first
        # Linear submodule of the first expert (works for standard Mamba
        # blocks whose in_proj/out_proj expose in/out features).
        if hidden_size is None:
            hidden_size = self._infer_hidden_size(self.experts[0])
        self.hidden_size = hidden_size
        router_dim = router_hidden_size or max(hidden_size // 4, 1)
        self.router = nn.Linear(hidden_size, router_dim)
        self.router_out = nn.Linear(router_dim, num_experts)

        self.register_buffer("memory_bank", torch.ones(num_experts) / num_experts)
        self.register_buffer("step_count", torch.tensor(0))

        self.is_merged = False
        self.merged_mamba = None
        self.last_token_difficulties = None
        self.last_batch_difficulty = 0.5

    @staticmethod
    def _infer_hidden_size(module: nn.Module) -> int:
        for _, sub in module.named_modules():
            if isinstance(sub, nn.Linear):
                return sub.in_features
        raise ValueError(
            "Could not infer hidden_size from the Mamba block factory; "
            "pass hidden_size explicitly to MemoryBankExFusionMamba."
        )

    def forward(self, x: torch.Tensor, macro_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.is_merged:
            return self.merged_mamba(x)

        router_hidden = F.relu(self.router(x))
        router_logits = self.router_out(router_hidden)

        token_diff = self._compute_token_difficulty(x, router_logits)
        self.last_token_difficulties = token_diff.detach()

        if self.difficulty_aggregation == "mean":
            batch_diff = token_diff.mean().detach()
        elif self.difficulty_aggregation == "max":
            batch_diff = token_diff.max().detach()
        else:
            batch_diff = torch.quantile(token_diff, 0.9).detach()
        self.last_batch_difficulty = batch_diff.item()

        router_weights = F.softmax(router_logits, dim=-1).mean(dim=(0, 1))

        if self.momentum_scheduler is not None:
            current_momentum = self.momentum_scheduler.get_momentum(
                self.step_count.item(), self.last_batch_difficulty
            )
        else:
            current_momentum = self.momentum

        if macro_context is not None:
            context_mod = torch.sigmoid(macro_context.mean()) * 0.1
            current_momentum = current_momentum * (1 - context_mod) + 0.85 * context_mod

        with torch.no_grad():
            self.memory_bank = (
                current_momentum * self.memory_bank
                + (1 - current_momentum) * router_weights
            )
            self.memory_bank = self.memory_bank / self.memory_bank.sum()
            self.step_count += 1

        fused = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            fused = fused + self.memory_bank[i] * expert(x)
        return fused

    def _compute_token_difficulty(self, hidden, router_logits):
        if self.difficulty_method == "entropy":
            probs = F.softmax(router_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_ent = math.log(router_logits.size(-1))
            return (entropy / max_ent).clamp(0, 1)
        elif self.difficulty_method == "max_prob":
            probs = F.softmax(router_logits, dim=-1)
            return (1 - probs.max(dim=-1).values).clamp(0, 1)
        else:
            var = hidden.var(dim=-1)
            return torch.sigmoid(var / (var.mean() + 1e-8)).clamp(0, 1)

    @torch.no_grad()
    def merge_to_dense(
        self,
        fisher_diagonals: List[Dict[str, torch.Tensor]],
        kfac_scores: Optional[Dict[str, float]] = None,
        kfac_temperature: float = 1.0,
        difficulty_importance: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        eps: float = 1e-8,
        merged_block: Optional[nn.Module] = None,
    ):
        """
        `fisher_diagonals` (one dict per expert) is required, mirroring the
        FFN class's requirement for its "fisher" stage — grouped Mamba
        merging always ends in a Fisher-weighted reduction.

        `merged_block`: an empty instance to copy merged parameters into
        (e.g. `block_factory()`). If omitted, one is created via
        `type(self.experts[0])(...)`-style copy — since Mamba block
        constructors are arbitrary and not introspectable in general, the
        safest default is to `copy.deepcopy` the first expert and overwrite
        its parameters with the merged values, so any non-parameter buffers
        / config carried on the instance survive the merge.
        """
        if self.is_merged:
            return

        device = self.memory_bank.device
        if difficulty_importance is None:
            difficulty_importance = torch.ones(self.num_experts, device=device)
        else:
            difficulty_importance = difficulty_importance.to(device)

        active_memory_bank = self.memory_bank.detach().clone().to(device)
        if kfac_scores is not None:
            active_memory_bank = incorporate_kfac_scores(
                active_memory_bank, kfac_scores, temperature=kfac_temperature, eps=eps
            )

        expert_params = [dict(e.named_parameters()) for e in self.experts]
        expert_params = [{k: v.data for k, v in d.items()} for d in expert_params]

        merged_params = mamba_grouped_merge(
            expert_params=expert_params,
            fishers=fisher_diagonals,
            memory_bank=active_memory_bank,
            difficulty_importance=difficulty_importance,
            policies=self.policies,
            ssm_fisher_boost=self.ssm_fisher_boost,
            seed=seed,
            eps=eps,
        )

        target = merged_block if merged_block is not None else copy.deepcopy(self.experts[0])
        apply_merged_mamba_params(target, merged_params)

        self.merged_mamba = target.to(device)
        self.memory_bank.copy_(active_memory_bank)
        self.is_merged = True


# =============================================================================
# 10. DAPH DECODER LAYER  (new: not present in either source draft)
# =============================================================================

class DAPHDecoderLayer(nn.Module):
    """
    Decoder block combining an (optional, externally supplied) attention
    path, an FFN-ExFusion path, a Mamba-ExFusion path, and a cheap FNet-style
    token-mixing fallback path, blended by a difficulty-predictive macro
    router.

    Design notes (new implementation; not present in either source draft,
    despite being referenced in this module's docstring/import example in
    every revision):

      - Attention is intentionally out of scope for this merge toolkit (it
        composes FFN/Mamba merging, not attention), so `attention_factory`
        is optional. If omitted, the attention path is simply disabled and
        contributes `torch.zeros_like(hidden)`, exactly like the FFN/Mamba
        paths when disabled.
      - The int/tensor crash flagged during review (`0 + tensor` raising a
        TypeError, and a subsequent `torch.stack` failing on a mix of
        Python ints and tensors) is fixed by using `torch.zeros_like(hidden)`
        for every disabled path rather than a bare `0`.
      - "Macro-routing softmax over attention/efficient/cheap paths": a
        small router produces 3 logits from the sequence-mean of `hidden`,
        softmax-normalized, and used as a convex combination weight over
        (attention, ffn+mamba "efficient", cheap-FNet) branch outputs. This
        gives the layer a difficulty-predictive way to lean on the cheap
        path when the macro router decides the input doesn't need full
        attention/Mamba compute.
      - The cheap path is an FNet-style unparameterized 2D Fourier mixing
        (real part of FFT over both the sequence and hidden dimensions)
        followed by a linear projection, as a nearly-free token-mixing
        fallback when the macro router selects it.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_exfusion_factory: Optional[Callable[[], MemoryBankExFusionFFN]] = None,
        mamba_exfusion_factory: Optional[Callable[[], MemoryBankExFusionMamba]] = None,
        attention_factory: Optional[Callable[[], nn.Module]] = None,
        macro_router_hidden: Optional[int] = None,
        use_cheap_path: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.attn_path = attention_factory() if attention_factory is not None else None
        self.ffn_path = ffn_exfusion_factory() if ffn_exfusion_factory is not None else None
        self.mamba_path = mamba_exfusion_factory() if mamba_exfusion_factory is not None else None

        self.use_cheap_path = use_cheap_path
        # FNet-inspired cheap path: layernorm -> 2D FFT (real) -> linear projection
        if use_cheap_path:
            self.cheap_norm = nn.LayerNorm(hidden_size)
            self.cheap_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.cheap_norm = None
            self.cheap_proj = None

        router_dim = macro_router_hidden or max(hidden_size // 4, 1)
        self.macro_router = nn.Sequential(
            nn.Linear(hidden_size, router_dim),
            nn.SiLU(),
            nn.Linear(router_dim, 3),  # [attention, efficient (ffn+mamba), cheap]
        )

    def _cheap_path(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute the cheap FNet-style path with 2D FFT.

        The input is first normalized, then a 2D FFT is applied across the last
        two dimensions (sequence and hidden), the real part is taken, and a
        linear projection produces the output.  A residual connection is not
        applied here because blending with the macro-router weights and
        residual addition occurs in the caller.
        """
        # Sanity: if cheap path is disabled, just return zeros
        if not self.use_cheap_path or self.cheap_proj is None or self.cheap_norm is None:
            return torch.zeros_like(hidden)
        x = self.cheap_norm(hidden)
        # compute 2D FFT over sequence and hidden dims
        x_fft = torch.fft.fft2(x, dim=(-2, -1)).real
        return self.cheap_proj(x_fft)

    def forward(self, hidden: torch.Tensor, attn_kwargs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        attn_kwargs = attn_kwargs or {}

        macro_logits = self.macro_router(hidden.mean(dim=1))
        macro_weights = F.softmax(macro_logits, dim=-1)  # (batch, 3)
        macro_context = macro_weights.mean(dim=0)  # scalar-ish context fed to ExFusion paths

        attn_out = self.attn_path(hidden, **attn_kwargs) if self.attn_path is not None else torch.zeros_like(hidden)

        ffn_out = self.ffn_path(hidden, macro_context=macro_context) if self.ffn_path is not None else torch.zeros_like(hidden)
        mamba_out = self.mamba_path(hidden, macro_context=macro_context) if self.mamba_path is not None else torch.zeros_like(hidden)
        efficient_out = ffn_out + mamba_out

        cheap_out = self._cheap_path(hidden) if self.use_cheap_path else torch.zeros_like(hidden)

        # Broadcast per-batch macro weights (batch, 3) over (batch, seq, hidden).
        w_attn = macro_weights[:, 0].view(-1, 1, 1)
        w_eff = macro_weights[:, 1].view(-1, 1, 1)
        w_cheap = macro_weights[:, 2].view(-1, 1, 1)

        output = w_attn * attn_out + w_eff * efficient_out + w_cheap * cheap_out
        return output

    def merge_exfusion_paths(
        self,
        path: Literal["ffn", "mamba", "both"] = "both",
        pipeline: List[Literal["dare", "ties", "fisher", "kfac"]] = ["dare", "ties", "fisher"],
        fisher_diagonals: Optional[List[Dict[str, torch.Tensor]]] = None,
        mamba_fisher_diagonals: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ):
        """Convenience wrapper to merge the FFN and/or Mamba ExFusion paths."""
        if path in ("ffn", "both") and self.ffn_path is not None:
            self.ffn_path.merge_to_dense(pipeline=pipeline, fisher_diagonals=fisher_diagonals, **kwargs)
        if path in ("mamba", "both") and self.mamba_path is not None:
            diagonals = mamba_fisher_diagonals if mamba_fisher_diagonals is not None else fisher_diagonals
            if diagonals is None:
                raise ValueError("fisher_diagonals (or mamba_fisher_diagonals) required to merge the Mamba path.")
            self.mamba_path.merge_to_dense(fisher_diagonals=diagonals, **{
                k: v for k, v in kwargs.items()
                if k in ("kfac_scores", "kfac_temperature", "difficulty_importance", "seed", "eps")
            })


# =============================================================================
# 11. UNIFIED CALIBRATION LOOP  (new: not present in either source draft)
# =============================================================================

def unified_calibration_loop(
    exfusion_modules: List[nn.Module],
    search_space: Dict[str, List[Any]],
    evaluate_fn: Callable[[List[nn.Module]], float],
    merge_kwargs_common: Optional[Dict[str, Any]] = None,
    rounds: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Greedy per-key hyperparameter search over merge settings shared across a
    set of ExFusion modules (e.g. every `MemoryBankExFusionFFN` /
    `MemoryBankExFusionMamba` inside a `DAPHDecoderLayer` stack).

    Coordinate-descent: for each key in `search_space` (in dict order),
    holding every other key fixed at its current-best value, try every
    candidate value, merge deep copies of `exfusion_modules` with that
    configuration, score them with `evaluate_fn`, and keep whichever
    candidate scores lowest. Repeats for `rounds` passes over all keys (a
    second pass can improve on a first-pass choice now that other keys have
    also moved). This is a search *helper*: it always merges disposable deep
    copies and only applies the winning configuration to the real
    `exfusion_modules` at the very end, since `merge_to_dense` is a
    destructive, one-way operation (`is_merged=True` short-circuits future
    merge calls).

    Args:
        exfusion_modules: modules exposing a `merge_to_dense(**kwargs)`
            method (both `MemoryBankExFusionFFN` and
            `MemoryBankExFusionMamba` qualify).
        search_space: e.g. {"dare_drop_rate": [0.1, 0.3, 0.5],
                             "ties_trim_ratio": [0.1, 0.2, 0.3],
                             "fisher_power": [0.5, 1.0, 1.5]}.
        evaluate_fn: given a list of *merged* modules (same order as
            `exfusion_modules`), returns a scalar loss (lower is better) —
            e.g. forward the owning model over a held-out calibration batch.
        merge_kwargs_common: extra kwargs passed to every `merge_to_dense`
            call regardless of the current search point (e.g.
            `fisher_diagonals`, since those aren't a searched hyperparameter).
        rounds: number of coordinate-descent passes over `search_space`.

    Returns:
        The winning kwargs dict (also already applied to `exfusion_modules`
        in place).
    """
    if not search_space:
        raise ValueError("search_space must contain at least one hyperparameter to search.")
    merge_kwargs_common = merge_kwargs_common or {}

    best_kwargs = {key: values[0] for key, values in search_space.items()}

    def _trial_score(trial_kwargs: Dict[str, Any]) -> float:
        trial_modules = [copy.deepcopy(m) for m in exfusion_modules]
        for m in trial_modules:
            m.merge_to_dense(**{**merge_kwargs_common, **trial_kwargs})
        return evaluate_fn(trial_modules)

    best_score = _trial_score(best_kwargs)
    if verbose:
        print(f"[calibration] initial config {best_kwargs} -> score {best_score:.6f}")

    for round_idx in range(rounds):
        improved = False
        for key, candidates in search_space.items():
            local_best_value = best_kwargs[key]
            local_best_score = best_score
            for candidate in candidates:
                if candidate == local_best_value:
                    continue
                trial_kwargs = dict(best_kwargs)
                trial_kwargs[key] = candidate
                score = _trial_score(trial_kwargs)
                if verbose:
                    print(f"[calibration] round {round_idx} key={key} candidate={candidate} -> score {score:.6f}")
                if score < local_best_score:
                    local_best_score = score
                    local_best_value = candidate
            if local_best_value != best_kwargs[key]:
                improved = True
            best_kwargs[key] = local_best_value
            best_score = local_best_score
        if not improved:
            break

    # Apply the winning configuration to the real modules exactly once.
    for m in exfusion_modules:
        m.merge_to_dense(**{**merge_kwargs_common, **best_kwargs})

    if verbose:
        print(f"[calibration] final config {best_kwargs} -> score {best_score:.6f}")

    return best_kwargs


# =============================================================================
# 12. LIGHTWEIGHT SELF-TEST
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    hidden_size, intermediate_size, num_experts, batch, seqlen = 16, 32, 3, 2, 5

    ffn = MemoryBankExFusionFFN(
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_experts=num_experts, activation="swiglu",
    )
    x = torch.randn(batch, seqlen, hidden_size)
    out = ffn(x)
    assert out.shape == x.shape, "FFN forward shape mismatch"

    fisher_diagonals = [
        {k: torch.rand_like(v) + 1e-3 for k, v in ffn._get_weights(e).items()}
        for e in ffn.experts
    ]
    ffn.merge_to_dense(pipeline=["dare", "ties", "fisher"], fisher_diagonals=fisher_diagonals, seed=0)
    assert ffn.is_merged
    merged_out = ffn(x)
    assert merged_out.shape == x.shape, "Merged FFN forward shape mismatch"

    # K-FAC ordering: pipeline ending in kfac after fisher should still
    # reflect kfac_scores, since kfac is resolved before any stage runs.
    ffn2 = MemoryBankExFusionFFN(
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_experts=num_experts, activation="swiglu",
    )
    kfac_scores = {f"layer_{i}": float(i + 1) for i in range(num_experts)}
    fisher_diagonals2 = [
        {k: torch.rand_like(v) + 1e-3 for k, v in ffn2._get_weights(e).items()}
        for e in ffn2.experts
    ]
    ffn2.merge_to_dense(
        pipeline=["dare", "ties", "fisher", "kfac"],
        fisher_diagonals=fisher_diagonals2,
        kfac_scores=kfac_scores,
        seed=0,
    )
    assert ffn2.is_merged
    print("MemoryBankExFusionFFN: forward + merge_to_dense OK")

    print("All lightweight self-tests passed.")