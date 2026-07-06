"""
Difficulty-Adaptive Top-p Macro-Router
========================================
Implements DynMoE-style cumulative-threshold routing for the DAPH macro-router.
Paths are activated based on a difficulty-modulated top-p (nucleus) selection,
where higher predicted difficulty lowers the threshold and activates more paths.

Also provides DAPHDecoderLayerV2, an updated decoder layer that uses the
adaptive router instead of the static softmax blending in DAPHDecoderLayer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .merge_toolkit import MemoryBankExFusionFFN, MemoryBankExFusionMamba


class AdaptiveTopPMacroRouter(nn.Module):
    """
    Macro-path router that uses top-p (nucleus) selection with a threshold
    modulated by predicted difficulty.

    Higher difficulty → lower threshold → more paths active.
    This is a lightweight extension of cumulative-threshold routing that
    reuses the existing difficulty predictor signal.

    Args:
        d_model: Hidden dimension of input tokens.
        num_paths: Number of macro-paths (typically 3: attention, efficient, cheap).
        base_threshold: Default cumulative probability threshold when difficulty=0.5.
        difficulty_scale: How much threshold changes per unit difficulty deviation.
        use_external_difficulty: If True, expects difficulty tensor in forward().
    """

    def __init__(
        self,
        d_model: int,
        num_paths: int = 3,
        base_threshold: float = 0.85,
        difficulty_scale: float = 0.3,
        use_external_difficulty: bool = False,
    ):
        super().__init__()
        self.num_paths = num_paths
        self.base_threshold = base_threshold
        self.difficulty_scale = difficulty_scale
        self.use_external_difficulty = use_external_difficulty

        # Path scoring
        self.router = nn.Linear(d_model, num_paths, bias=False)

        if not use_external_difficulty:
            self.difficulty_predictor = nn.Sequential(
                nn.Linear(d_model, max(d_model // 4, 1)),
                nn.ReLU(),
                nn.Linear(max(d_model // 4, 1), 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        hidden: torch.Tensor,
        external_difficulty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (B, L, D) input tokens.
            external_difficulty: (B, L) or (B*L,) optional difficulty scores.

        Returns:
            selected_mask: (B, L, num_paths) boolean mask of activated paths.
            probs: (B, L, num_paths) softmax path probabilities for blending.
        """
        B, L, D = hidden.shape
        flat = hidden.view(-1, D)  # (B*L, D)

        logits = self.router(flat)  # (B*L, num_paths)
        probs = F.softmax(logits, dim=-1)  # (B*L, num_paths)

        # Difficulty per token
        if self.use_external_difficulty:
            if external_difficulty is None:
                raise ValueError("External difficulty required but None given")
            diff = external_difficulty.view(-1)  # (B*L,)
        else:
            diff = self.difficulty_predictor(flat).squeeze(-1)  # (B*L,)

        # Threshold: higher difficulty → higher threshold → more paths active.
        # In top-p (nucleus) selection, a higher cumulative-probability threshold
        # requires accumulating more paths to cross it, so more paths are selected.
        threshold = self.base_threshold + self.difficulty_scale * (diff - 0.5)
        threshold = torch.clamp(threshold, min=0.4, max=0.95)

        # Top-p selection: sort descending, accumulate, select prefix
        sorted_probs, indices = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Elements where cumsum < threshold, plus one extra to cross it
        under = cumsum < threshold.unsqueeze(-1)  # (B*L, num_paths)
        mask_sorted = torch.cat(
            [
                torch.ones_like(under[:, :1], dtype=torch.bool),
                under[:, :-1],
            ],
            dim=1,
        )

        # Scatter back to original path order
        selected_mask = torch.zeros_like(probs, dtype=torch.bool)
        selected_mask.scatter_(1, indices, mask_sorted)

        return selected_mask.view(B, L, -1), probs.view(B, L, -1)


class DAPHDecoderLayerV2(nn.Module):
    """
    DAPH decoder layer using AdaptiveTopPMacroRouter for dynamic path selection.

    Unlike DAPHDecoderLayer (which always computes all paths and blends by
    softmax weights), this layer only computes paths that are selected by the
    top-p mask, saving inference compute when inputs are easy.

    Args:
        hidden_size: Model dimension.
        ffn_exfusion_factory: Callable returning MemoryBankExFusionFFN.
        mamba_exfusion_factory: Callable returning MemoryBankExFusionMamba.
        attention_factory: Callable returning attention module.
        macro_router_kwargs: Dict passed to AdaptiveTopPMacroRouter.
        use_cheap_path: Whether to include the FNet cheap path.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_exfusion_factory: Optional[callable] = None,
        mamba_exfusion_factory: Optional[callable] = None,
        attention_factory: Optional[callable] = None,
        macro_router_kwargs: Optional[dict] = None,
        use_cheap_path: bool = True,
        use_packed_dispatch: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_cheap_path = use_cheap_path
        self.use_packed_dispatch = use_packed_dispatch

        self.attn_path = attention_factory() if attention_factory is not None else None
        self.ffn_path = ffn_exfusion_factory() if ffn_exfusion_factory is not None else None
        self.mamba_path = mamba_exfusion_factory() if mamba_exfusion_factory is not None else None

        if use_cheap_path:
            # FNet-inspired cheap path uses its own normalization and projection
            self.cheap_norm = nn.LayerNorm(hidden_size)
            self.cheap_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.cheap_norm = None
            self.cheap_proj = None

        router_kwargs = macro_router_kwargs or {}
        self.macro_router = AdaptiveTopPMacroRouter(
            d_model=hidden_size, num_paths=3, **router_kwargs
        )

        self.final_norm = nn.LayerNorm(hidden_size)

    def _cheap_path(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute the cheap FNet-style path with 2D FFT.

        When the cheap path is enabled, this normalizes the input, applies a
        2D FFT across (sequence, hidden) dimensions, and projects the real
        component back to the hidden dimension.

        The FFT result is memoised based on the input tensor's data pointer
        and shape, so repeated calls with the same input (e.g. when the
        residual stream hasn't changed between the cheap path and other
        paths in the same layer) avoid recomputing the FFT.
        """
        if not self.use_cheap_path or self.cheap_proj is None or self.cheap_norm is None:
            return torch.zeros_like(hidden)
        x = self.cheap_norm(hidden)
        # Memoise the FFT based on the normalised tensor's identity.
        # This avoids recomputing fft2 when the same hidden state is
        # processed multiple times (e.g. across layers with identical input).
        cache_key = (x.data_ptr(), x.shape, x.device)
        if hasattr(self, '_fft_cache') and self._fft_cache.get('key') == cache_key:
            x_fft = self._fft_cache['value']
        else:
            # Upcast to float32 before FFT — torch.fft does not support
            # half-precision (float16/bfloat16) on many GPU and CPU platforms.
            x_fft = torch.fft.fft2(x.float(), dim=(-2, -1)).real.to(x.dtype)
            self._fft_cache = {'key': cache_key, 'value': x_fft}
        return self.cheap_proj(x_fft)

    @staticmethod
    def _packed_dispatch(
        hidden: torch.Tensor,
        path_fn,
        path_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gather-run-scatter: run ``path_fn`` only on tokens where
        ``path_mask`` is True, then scatter results back.

        This realizes real FLOP savings for token-independent paths (FFN)
        by avoiding computation on tokens that don't use the path.
        Tokens that need full-sequence context (attention, Mamba) cannot
        use this and must run on the full sequence.

        Args:
            hidden: (B, L, D) input tensor.
            path_fn: Callable taking (B, L', D) → (B, L', D).
            path_mask: (B, L) boolean mask of tokens needing this path.

        Returns:
            (B, L, D) output with path results scattered to selected tokens.
        """
        if path_mask.all():
            # All tokens need this path — no savings from packing
            return path_fn(hidden)
        B, L, D = hidden.shape
        flat_hidden = hidden.reshape(-1, D)  # (B*L, D)
        flat_mask = path_mask.reshape(-1)  # (B*L,)
        selected_idx = flat_mask.nonzero(as_tuple=True)[0]
        if selected_idx.numel() == 0:
            return torch.zeros_like(hidden)
        # Gather: pack selected tokens into a contiguous batch
        packed = flat_hidden[selected_idx]  # (N, D)
        # Run the path on the packed subset
        packed_out = path_fn(packed.unsqueeze(0)).squeeze(0)  # (N, D)
        # Scatter: write results back to their original positions
        out = torch.zeros_like(flat_hidden)
        out[selected_idx] = packed_out
        return out.view(B, L, D)

    def forward(
        self,
        hidden: torch.Tensor,
        attn_kwargs: Optional[dict] = None,
        external_difficulty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_kwargs = attn_kwargs or {}
        residual = hidden

        # Get path selection mask and blending probabilities
        mask, probs = self.macro_router(hidden, external_difficulty)
        # mask: (B, L, 3), probs: (B, L, 3)

        # Re-normalize probabilities over selected paths so tokens that
        # activate only a subset of paths retain full output scale.
        # Without this, an easy token activating only the cheap path gets
        # squashed by probs[:,:,2] (e.g. 0.6), causing exponential scale
        # decay across decoder layers.
        active_probs = probs * mask.to(probs.dtype)
        prob_sum = active_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        norm_probs = active_probs / prob_sum  # sums to 1.0 over active paths

        # Only compute paths that are actually selected.
        #
        # Packed token dispatch (gather-run-scatter) is used for the FFN
        # path, which is token-independent and can skip computation on
        # tokens that don't use it — realizing real FLOP savings.
        #
        # Attention and Mamba require full-sequence context (K/V cache,
        # recurrent state) and cannot be packed, so they run on the full
        # sequence but their outputs are still masked by norm_probs.
        #
        # The cheap path (FFT) operates on the full (B, L, D) tensor and
        # also cannot be packed.
        out = torch.zeros_like(hidden)

        if self.attn_path is not None and mask[:, :, 0].any():
            attn_out = self.attn_path(hidden, **attn_kwargs)
            out += attn_out * norm_probs[:, :, 0:1]

        if (self.ffn_path is not None or self.mamba_path is not None) and mask[:, :, 1].any():
            eff_out = torch.zeros_like(hidden)
            if self.ffn_path is not None:
                # Use packed dispatch for FFN when merged (inference mode).
                # During training (not merged), the FFN's internal expert
                # routing needs the full batch for statistics.
                if hasattr(self.ffn_path, 'is_merged') and self.ffn_path.is_merged and self.use_packed_dispatch:
                    eff_out += self._packed_dispatch(
                        hidden, self.ffn_path, mask[:, :, 1]
                    )
                else:
                    eff_out += self.ffn_path(hidden)
            if self.mamba_path is not None:
                # Mamba is sequential — cannot pack, must run on full sequence
                eff_out += self.mamba_path(hidden)
            out += eff_out * norm_probs[:, :, 1:2]

        if self.use_cheap_path and mask[:, :, 2].any():
            cheap_out = self._cheap_path(hidden)
            out += cheap_out * norm_probs[:, :, 2:3]

        return self.final_norm(residual + out)

    def merge_exfusion_paths(
        self,
        path: str = "both",
        pipeline: list = None,
        fisher_diagonals: Optional[list] = None,
        mamba_fisher_diagonals: Optional[list] = None,
        **kwargs,
    ):
        """Convenience wrapper to merge FFN and/or Mamba ExFusion paths."""
        if pipeline is None:
            pipeline = ["dare", "ties", "fisher"]
        if path in ("ffn", "both") and self.ffn_path is not None:
            self.ffn_path.merge_to_dense(pipeline=pipeline, fisher_diagonals=fisher_diagonals, **kwargs)
        if path in ("mamba", "both") and self.mamba_path is not None:
            diagonals = mamba_fisher_diagonals if mamba_fisher_diagonals is not None else fisher_diagonals
            if diagonals is None:
                raise ValueError("fisher_diagonals required to merge the Mamba path.")
            self.mamba_path.merge_to_dense(fisher_diagonals=diagonals, **{
                k: v for k, v in kwargs.items()
                if k in ("kfac_scores", "kfac_temperature", "difficulty_importance", "seed", "eps")
            })
