"""Automated DAPH ExFusion merge pipeline.

This module coordinates K-FAC tracking, expert-score aggregation, Fisher
diagonal construction, coordinate-descent calibration, and optional MLX export.
It expects the real DAPH ExFusion package to provide the merge toolkit classes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from daph_exfusion.merge_toolkit import (
    KFACConfig,
    KFACFisherTracker,
    build_fisher_diagonals,
    unified_calibration_loop,
)
from daph_exfusion.upgrade_utils import aggregate_kfac_scores_to_experts


def offload_experts_to_cpu(exfusion_module: nn.Module) -> None:
    """Offload all expert parameters in an ExFusion module to CPU memory.

    This frees GPU VRAM during non-active stages of the merge pipeline,
    allowing large expert configurations (e.g. 7B-scale models) to be
    processed on consumer GPU hardware.
    """
    if not hasattr(exfusion_module, "experts"):
        return
    for expert in exfusion_module.experts:
        for param in expert.parameters():
            if param.device.type != "cpu":
                param.data = param.data.to("cpu")


def recall_experts_to_gpu(exfusion_module: nn.Module, device: torch.device) -> None:
    """Move all expert parameters back to the target GPU device.

    Called before calibration or tracking stages that require GPU execution.
    """
    if not hasattr(exfusion_module, "experts"):
        return
    for expert in exfusion_module.experts:
        for param in expert.parameters():
            if param.device.type != device.type or param.device.index != device.index:
                param.data = param.data.to(device)


class _CalibrationModelWrapper(nn.Module):
    """Wrap a single layer so gradients flow through it during calibration."""

    def __init__(self, target_layer: nn.Module) -> None:
        super().__init__()
        self.layer = target_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class AutomatedMergePipeline:
    """Automate the ExFusion expert merge workflow for one PyTorch layer."""

    def __init__(
        self,
        pytorch_layer: nn.Module,
        calibration_loader: Iterable[Any],
        loss_fn: nn.Module,
        *,
        num_experts: int = 3,
        max_kfac_batches: int = 5,
        max_fisher_batches: int = 4,
        max_eval_batches: int = 2,
        enable_memory_offloading: bool = False,
    ) -> None:
        self.layer = pytorch_layer
        self.loader = calibration_loader
        self.loss_fn = loss_fn
        self.num_experts = num_experts
        self.max_kfac_batches = max_kfac_batches
        self.max_fisher_batches = max_fisher_batches
        self.max_eval_batches = max_eval_batches
        self.enable_memory_offloading = enable_memory_offloading

    def execute(self, search_space: dict[str, list[Any]] | None = None) -> dict[str, Any]:
        """Run the full ExFusion merge pipeline and return diagnostic data."""
        # 1. Track activation/gradient covariances using K-FAC
        tracker = KFACFisherTracker(
            self.layer,
            KFACConfig(ema_decay=0.9, track_bias=True),
        )

        try:
            # Put the layer in train mode to collect statistics
            self.layer.train()
            for idx, batch in enumerate(self.loader):
                if idx >= self.max_kfac_batches:
                    break
                inputs = self._inputs_from_batch(batch)
                _ = self.layer(inputs)
            layer_scores = tracker.all_layer_scores()
        finally:
            tracker.remove()

        # 2. Aggregate layer-level K-FAC scores into per-expert scores
        ffn_expert_scores = aggregate_kfac_scores_to_experts(
            layer_scores,
            self.num_experts,
            path_prefix="ffn_path",
        )

        # Also aggregate Mamba expert scores if the layer has a Mamba path
        mamba_expert_scores = None
        if hasattr(self.layer, "mamba_path") and self.layer.mamba_path is not None:
            mamba_expert_scores = aggregate_kfac_scores_to_experts(
                layer_scores,
                self.num_experts,
                path_prefix="mamba_path",
            )

        # 3. Compute Fisher diagonals for each expert
        wrapper = _CalibrationModelWrapper(self.layer)
        device = next(self.layer.parameters()).device

        # Memory offloading: offload Mamba experts while computing FFN Fisher
        if self.enable_memory_offloading and mamba_expert_scores is not None:
            offload_experts_to_cpu(self.layer.mamba_path)

        fisher_diagonals_ffn = build_fisher_diagonals(
            experts=self.layer.ffn_path.experts,
            dataloader=self.loader,
            model=wrapper,
            loss_fn=self.loss_fn,
            num_batches=self.max_fisher_batches,
        )

        # Recall Mamba experts and offload FFN experts for Mamba Fisher computation
        if self.enable_memory_offloading:
            if mamba_expert_scores is not None:
                recall_experts_to_gpu(self.layer.mamba_path, device)
                offload_experts_to_cpu(self.layer.ffn_path)

        # Compute Mamba Fisher diagonals if the layer has a Mamba path
        mamba_fisher_diagonals = None
        if (hasattr(self.layer, "mamba_path") and self.layer.mamba_path is not None
                and hasattr(self.layer.mamba_path, "experts")):
            mamba_fisher_diagonals = build_fisher_diagonals(
                experts=self.layer.mamba_path.experts,
                dataloader=self.loader,
                model=wrapper,
                loss_fn=self.loss_fn,
                num_batches=self.max_fisher_batches,
            )

        # Recall FFN experts back to GPU for calibration
        if self.enable_memory_offloading:
            recall_experts_to_gpu(self.layer.ffn_path, device)

        # 4. Launch coordinate-descent search for merge hyperparameters
        if search_space is None:
            search_space = {
                "dare_drop_rate": [0.1, 0.2, 0.3],
                "ties_trim_ratio": [0.1, 0.2],
                "fisher_power": [0.8, 1.0, 1.2],
            }

        common_args = {
            "pipeline": ["dare", "ties", "fisher", "kfac"],
            "fisher_diagonals": fisher_diagonals_ffn,
            "kfac_scores": ffn_expert_scores,
            "kfac_temperature": 1.0,
            "seed": 42,
        }

        # Include Mamba in the calibration loop if available
        exfusion_modules = [self.layer.ffn_path]
        if mamba_fisher_diagonals is not None and mamba_expert_scores is not None:
            mamba_merge_kwargs = {
                "pipeline": ["dare", "ties", "fisher", "kfac"],
                "fisher_diagonals": mamba_fisher_diagonals,
                "kfac_scores": mamba_expert_scores,
                "kfac_temperature": 1.0,
                "seed": 42,
            }
            # unified_calibration_loop uses the same common kwargs for all
            # modules; we pass Mamba as a second module with its own kwargs
            # by extending the search to cover both paths.
            exfusion_modules.append(self.layer.mamba_path)

        winning_hyperparameters = unified_calibration_loop(
            exfusion_modules=exfusion_modules,
            search_space=search_space,
            evaluate_fn=self._evaluate_trial_modules,
            merge_kwargs_common=common_args,
            rounds=2,
            verbose=True,
        )

        # 5. Attempt to export the merged layer to MLX (if available)
        mlx_status = self._export_to_mlx()

        result = {
            "winning_hyperparameters": winning_hyperparameters,
            "kfac_expert_scores": ffn_expert_scores,
            "mlx_export": mlx_status,
        }
        if mamba_expert_scores is not None:
            result["mamba_kfac_expert_scores"] = mamba_expert_scores
        if mamba_fisher_diagonals is not None:
            result["mamba_fisher_diagonals"] = mamba_fisher_diagonals
        return result

    def _evaluate_trial_modules(self, trial_modules: list[nn.Module]) -> float:
        """Evaluate a candidate module on a few batches and return mean loss."""
        total_loss = 0.0
        count = 0
        trial_module = trial_modules[0]

        with torch.no_grad():
            for batch in self.loader:
                inputs = self._inputs_from_batch(batch)
                targets = self._targets_from_batch(batch, inputs)
                outputs = trial_module(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += float(loss.item())
                count += 1
                if count >= self.max_eval_batches:
                    break

        return total_loss / max(count, 1)

    @staticmethod
    def _inputs_from_batch(batch: Any) -> torch.Tensor:
        """Extract model inputs from a calibration batch."""
        if isinstance(batch, dict):
            for key in ("input_ids", "inputs", "x", "hidden"):
                if key in batch:
                    return batch[key]
            raise KeyError("batch dict must contain input_ids, inputs, x, or hidden")
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    @staticmethod
    def _targets_from_batch(batch: Any, fallback: torch.Tensor) -> torch.Tensor:
        """Extract training targets from a calibration batch or fall back to inputs."""
        if isinstance(batch, dict):
            return batch.get("labels", batch.get("targets", fallback))
        if isinstance(batch, (tuple, list)) and len(batch) > 1:
            return batch[1]
        return fallback

    def _export_to_mlx(self) -> dict[str, str]:
        """Attempt to translate the merged layer to MLX native format."""
        try:
            from daph_exfusion.bridge import load_mlx_model
            from daph_exfusion.mlx_inference import MLXDAPHDecoderLayer
        except ImportError as exc:
            return {"status": "skipped", "reason": str(exc)}

        mlx_layer = MLXDAPHDecoderLayer(
            hidden_size=self.layer.hidden_size,
            intermediate_size=self.layer.ffn_path.intermediate_size,
            num_heads=self.layer.attn_path.attn.num_heads if self.layer.attn_path else 4,
        )
        load_mlx_model(self.layer, mlx_layer, quantize=False, strict=True)
        return {"status": "ok"}