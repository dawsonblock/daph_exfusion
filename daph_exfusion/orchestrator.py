"""Automated DAPH ExFusion merge pipeline.

This module coordinates K-FAC tracking, expert-score aggregation, Fisher
diagonal construction, coordinate-descent calibration, and optional MLX export.
It expects the real DAPH ExFusion package to provide the merge toolkit classes.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import torch
import torch.nn as nn

from daph_exfusion.merge_toolkit import (
    KFACConfig,
    KFACFisherTracker,
    build_fisher_diagonals,
    unified_calibration_loop,
)
from daph_exfusion.upgrade_utils import aggregate_kfac_scores_to_experts


# Global thread pool for asynchronous host-device memory transfers on MPS/CPU.
# Single worker ensures transfers are serialized and don't contend with each
# other; the pool is reused across pipeline instances to avoid thread creation
# overhead.  Created lazily to avoid spawning threads at import time.
_TRANSFER_POOL: Optional[ThreadPoolExecutor] = None


def _get_transfer_pool() -> ThreadPoolExecutor:
    """Return the global transfer thread pool, creating it on first use."""
    global _TRANSFER_POOL
    if _TRANSFER_POOL is None:
        _TRANSFER_POOL = ThreadPoolExecutor(max_workers=1)
    return _TRANSFER_POOL


class AsyncMemoryOffloader:
    """Device-agnostic asynchronous parameter pre-fetcher.

    Uses CUDA streams on NVIDIA hardware for overlapped non-blocking
    transfers.  On Apple Silicon (MPS) and CPU, where no stream API is
    available, falls back to a background Python thread pool to prevent
    CPU stalls during parameter offloading/recall.

    On UMA (Unified Memory Architecture) systems like Apple Silicon,
    CPU and GPU share the same physical memory, so the actual copy
    latency is lower than PCIe-based systems.  The thread pool still
    helps by allowing the main thread to continue computation while
    the parameter moves execute in the background.

    Args:
        device: Target compute device for recall operations.
        use_async: If True, use async transfers (CUDA stream or thread
            pool).  If False, all transfers are synchronous.
    """

    def __init__(self, device: torch.device, use_async: bool = True):
        self.device = device
        self.use_async = use_async
        self.cuda_stream = None
        self.thread_future = None

        if use_async and device.type == "cuda":
            self.cuda_stream = torch.cuda.Stream()

    def offload_async(self, module: nn.Module) -> None:
        """Asynchronously push expert parameters to CPU memory."""
        if not hasattr(module, "experts"):
            return

        if not self.use_async:
            for expert in module.experts:
                for param in expert.parameters():
                    if param.device.type != "cpu":
                        param.data = param.data.to("cpu")
            return

        if self.cuda_stream is not None:
            # NVIDIA pathway: asynchronous CUDA stream
            for expert in module.experts:
                for param in expert.parameters():
                    if param.device.type != "cpu":
                        if not param.data.is_pinned():
                            param.data = param.data.pin_memory()
                        with torch.cuda.stream(self.cuda_stream):
                            param.data = param.data.to("cpu", non_blocking=True)
        else:
            # Apple Silicon / CPU pathway: background thread pool
            def _thread_offload():
                for expert in module.experts:
                    for param in expert.parameters():
                        if param.device.type != "cpu":
                            param.data = param.data.to("cpu")

            # Wait for previous transfer, then submit new job
            self.synchronize()
            self.thread_future = _get_transfer_pool().submit(_thread_offload)

    def recall_async(self, module: nn.Module) -> None:
        """Asynchronously pull expert parameters to GPU memory."""
        if not hasattr(module, "experts"):
            return

        if not self.use_async:
            for expert in module.experts:
                for param in expert.parameters():
                    if param.device.type != self.device.type:
                        param.data = param.data.to(self.device)
            return

        if self.cuda_stream is not None:
            # NVIDIA pathway
            for expert in module.experts:
                for param in expert.parameters():
                    if param.device.type == "cpu":
                        with torch.cuda.stream(self.cuda_stream):
                            param.data = param.data.to(self.device, non_blocking=True)
        else:
            # Apple Silicon / CPU pathway
            def _thread_recall():
                for expert in module.experts:
                    for param in expert.parameters():
                        if param.device.type == "cpu":
                            param.data = param.data.to(self.device)

            self.synchronize()
            self.thread_future = _get_transfer_pool().submit(_thread_recall)

    def synchronize(self) -> None:
        """Wait for any active background memory transfers to complete."""
        if self.cuda_stream is not None:
            torch.cuda.current_stream().wait_stream(self.cuda_stream)
        elif self.thread_future is not None:
            self.thread_future.result()
            self.thread_future = None


# ── Backward-compatible function wrappers ────────────────────────────────
# These preserve the v4.6.0 API for callers that use the standalone
# functions directly.  Internally they delegate to AsyncMemoryOffloader.

def _get_transfer_stream() -> Optional[Any]:
    """Return a CUDA stream for async transfers, or None if unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.Stream()
    return None


def _sync_stream(stream: Optional[Any]) -> None:
    """Synchronize the transfer stream with the current compute stream."""
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def offload_experts_to_cpu(exfusion_module: nn.Module,
                           stream: Optional[Any] = None) -> None:
    """Offload all expert parameters in an ExFusion module to CPU memory.

    Backward-compatible wrapper.  Prefer using ``AsyncMemoryOffloader``
    directly for new code, as it supports thread-pool-based async
    transfers on MPS/CPU.
    """
    if not hasattr(exfusion_module, "experts"):
        return
    for expert in exfusion_module.experts:
        for param in expert.parameters():
            if param.device.type != "cpu":
                if stream is not None:
                    with torch.cuda.stream(stream):
                        param.data = param.data.to("cpu", non_blocking=True)
                else:
                    param.data = param.data.to("cpu", non_blocking=True)


def recall_experts_to_gpu(exfusion_module: nn.Module, device: torch.device,
                          stream: Optional[Any] = None) -> None:
    """Move all expert parameters back to the target GPU device.

    Backward-compatible wrapper.  Prefer using ``AsyncMemoryOffloader``
    directly for new code.
    """
    if not hasattr(exfusion_module, "experts"):
        return
    for expert in exfusion_module.experts:
        for param in expert.parameters():
            if param.device.type != device.type or param.device.index != device.index:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        param.data = param.data.to(device, non_blocking=True)
                else:
                    param.data = param.data.to(device, non_blocking=True)


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

        # v4.7.0: Device-agnostic async offloading via AsyncMemoryOffloader.
        # Uses CUDA streams on NVIDIA, thread-pool pipelining on MPS/CPU.
        offloader = AsyncMemoryOffloader(
            device, use_async=self.enable_memory_offloading
        )

        # Memory offloading: offload Mamba experts while computing FFN Fisher
        if self.enable_memory_offloading and mamba_expert_scores is not None:
            offloader.offload_async(self.layer.mamba_path)

        fisher_diagonals_ffn = build_fisher_diagonals(
            experts=self.layer.ffn_path.experts,
            dataloader=self.loader,
            model=wrapper,
            loss_fn=self.loss_fn,
            num_batches=self.max_fisher_batches,
        )

        # Recall Mamba experts and offload FFN experts for Mamba Fisher computation
        if self.enable_memory_offloading:
            offloader.synchronize()
            if mamba_expert_scores is not None:
                offloader.recall_async(self.layer.mamba_path)
                offloader.offload_async(self.layer.ffn_path)

        # Compute Mamba Fisher diagonals if the layer has a Mamba path
        mamba_fisher_diagonals = None
        if (hasattr(self.layer, "mamba_path") and self.layer.mamba_path is not None
                and hasattr(self.layer.mamba_path, "experts")):
            offloader.synchronize()
            mamba_fisher_diagonals = build_fisher_diagonals(
                experts=self.layer.mamba_path.experts,
                dataloader=self.loader,
                model=wrapper,
                loss_fn=self.loss_fn,
                num_batches=self.max_fisher_batches,
            )

        # Recall FFN experts back to GPU for calibration
        if self.enable_memory_offloading:
            offloader.synchronize()
            offloader.recall_async(self.layer.ffn_path)
            offloader.synchronize()

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

        # Include Mamba in a separate merge step if available.
        # unified_calibration_loop applies the same merge_kwargs_common to
        # all modules, so we cannot mix FFN and Mamba (they need different
        # Fisher diagonals).  Instead, calibrate FFN first, then manually
        # merge Mamba with its own kwargs.
        exfusion_modules = [self.layer.ffn_path]

        winning_hyperparameters = unified_calibration_loop(
            exfusion_modules=exfusion_modules,
            search_space=search_space,
            evaluate_fn=self._evaluate_trial_modules,
            merge_kwargs_common=common_args,
            rounds=2,
            verbose=True,
        )

        # Manually merge Mamba with its own Fisher diagonals and the
        # winning hyperparameters from the FFN calibration.
        if mamba_fisher_diagonals is not None and mamba_expert_scores is not None:
            mamba_merge_kwargs = {
                "pipeline": ["dare", "ties", "fisher", "kfac"],
                "fisher_diagonals": mamba_fisher_diagonals,
                "kfac_scores": mamba_expert_scores,
                "kfac_temperature": 1.0,
                "seed": 42,
                **winning_hyperparameters,
            }
            self.layer.mamba_path.merge_to_dense(**mamba_merge_kwargs)

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