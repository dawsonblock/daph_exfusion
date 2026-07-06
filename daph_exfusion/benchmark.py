"""
DAPH ExFusion Benchmark Suite
===============================
Rigorous evaluation protocol for MoE merging and dynamic routing.

This module provides harnesses for:
  - Per-domain perplexity evaluation
  - Long-range copy accuracy (LRA-Copy)
  - Inference latency measurement
  - Routing efficiency statistics
  - Ablation studies

Usage:
    from daph_exfusion.benchmark import MoEBenchmarkSuite, run_ablation_study

Note: This is a research prototype. Full integration with real datasets
(Wikitext-103, GovReport, QMSum) requires external data loaders.
"""

import time
import math
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def _try_mlx():
    try:
        import mlx.core as mx
        return mx
    except ImportError:
        return None


@dataclass
class BenchmarkResult:
    """Container for a single benchmark run."""
    method_name: str
    domain_ppl: Dict[str, float]
    lra_copy_acc: Dict[int, float]
    avg_latency_ms: float
    avg_active_paths: float
    activated_params_ratio: float
    notes: str = ""


class MoEBenchmarkSuite:
    """
    Evaluation harness for DAPH ExFusion models.

    Args:
        model: The model to evaluate (PyTorch nn.Module or MLX nn.Module).
        device: torch device for PyTorch models.
    """

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.mx = _try_mlx()
        self._is_mlx = hasattr(model, "__call__") and not isinstance(model, nn.Module)

    def evaluate_perplexity(self, dataloader) -> float:
        """Calculate token-level perplexity from a data loader."""
        if self._is_mlx:
            raise NotImplementedError("MLX perplexity evaluation requires custom implementation")

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids)

                logits = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += labels.numel()

        return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

    def measure_average_latency_ms(self, prompt_shape: tuple, steps: int = 100) -> float:
        """Measure average inference latency per token."""
        if self._is_mlx and self.mx is not None:
            return self._measure_mlx_latency(prompt_shape, steps)
        return self._measure_pytorch_latency(prompt_shape, steps)

    def _measure_pytorch_latency(self, prompt_shape: tuple, steps: int) -> float:
        self.model.eval()
        dummy = torch.randn(*prompt_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy)

        latencies = []
        with torch.no_grad():
            for _ in range(steps):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = self.model(dummy)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        return float(np.mean(latencies))

    def _measure_mlx_latency(self, prompt_shape: tuple, steps: int) -> float:
        dummy = self.mx.random.normal(prompt_shape)

        # Warmup
        for _ in range(10):
            out = self.model(dummy)
            self.mx.eval(out)

        latencies = []
        for _ in range(steps):
            t0 = time.perf_counter()
            out = self.model(dummy)
            self.mx.eval(out)
            latencies.append((time.perf_counter() - t0) * 1000)

        return float(np.mean(latencies))

    def collect_routing_metrics(self, dataloader, max_batches: int = 10) -> dict:
        """Collect path activation statistics from dynamic routing layers."""
        active_counts = []
        total_tokens = 0

        if self._is_mlx:
            return {"error": "MLX routing metrics not yet implemented"}

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                x = batch["input_ids"].to(self.device)
                _ = self.model(x)

                # Inspect layers for routing info
                for module in self.model.modules():
                    if hasattr(module, "macro_router") and hasattr(module, "_last_mask"):
                        mask = module._last_mask  # (B, L, 3)
                        active_counts.append(mask.sum(dim=-1).float().mean().item())
                        total_tokens += mask.size(0) * mask.size(1)

        if not active_counts:
            return {"avg_active_paths": None, "note": "No dynamic routing layers found"}

        return {
            "avg_active_paths": float(np.mean(active_counts)),
            "total_tokens_evaluated": total_tokens,
        }

    def evaluate_lra_copy(self, seq_lengths: List[int] = None, num_samples: int = 100) -> Dict[int, float]:
        """
        Evaluate long-range copy accuracy at various sequence lengths.
        Returns accuracy per sequence length.
        """
        if seq_lengths is None:
            seq_lengths = [512, 1024, 2048]

        results = {}
        for length in seq_lengths:
            correct = 0
            for _ in range(num_samples):
                # Generate synthetic copy task: [prefix] [marker] [prefix]
                prefix_len = length // 3
                vocab_size = 1000

                prefix = torch.randint(0, vocab_size, (1, prefix_len))
                marker = torch.tensor([[vocab_size - 1]])  # special marker token
                target = prefix.clone()

                input_seq = torch.cat([prefix, marker, prefix], dim=1)  # (1, length)

                if not self._is_mlx:
                    input_seq = input_seq.to(self.device)
                    with torch.no_grad():
                        logits = self.model(input_seq)
                    pred = logits[:, -prefix_len:, :].argmax(dim=-1)
                    if torch.equal(pred.cpu(), target):
                        correct += 1
                else:
                    # MLX implementation would go here
                    correct += 0

            results[length] = correct / num_samples
        return results


def run_ablation_study(
    base_model_fn: Callable,
    expert_models: List[nn.Module],
    dataloaders: Dict[str, Any],
    methods: List[str],
) -> List[BenchmarkResult]:
    """
    Run a controlled ablation study across multiple merge methods.

    Args:
        base_model_fn: Callable that returns a fresh base model instance.
        expert_models: List of fine-tuned expert models to merge.
        dataloaders: Dict of {domain_name: dataloader}.
        methods: List of method names to compare.

    Returns:
        List of BenchmarkResult, one per method.
    """
    results = []

    for method in methods:
        print(f"Evaluating: {method}")
        model = base_model_fn()

        # Apply merge method
        if method == "average":
            # Simple parameter averaging
            with torch.no_grad():
                for name, param in model.named_parameters():
                    expert_params = [e.state_dict()[name] for e in expert_models if name in e.state_dict()]
                    if expert_params:
                        param.copy_(sum(expert_params) / len(expert_params))
        elif method == "daph_exfusion":
            # Requires DAPH-specific merge pipeline
            print("  [Note] Full DAPH merge requires manual setup of memory banks and Fisher diagonals")
        else:
            print(f"  [Note] Method {method} not fully automated in this prototype")

        suite = MoEBenchmarkSuite(model)

        # Per-domain PPL
        domain_ppl = {}
        for domain_name, loader in dataloaders.items():
            try:
                ppl = suite.evaluate_perplexity(loader)
                domain_ppl[domain_name] = ppl
            except Exception as e:
                domain_ppl[domain_name] = float("nan")
                print(f"  Error on {domain_name}: {e}")

        # LRA-Copy
        lra_acc = suite.evaluate_lra_copy(num_samples=10)

        # Latency
        latency = suite.measure_average_latency_ms((2, 128, model.hidden_size if hasattr(model, "hidden_size") else 512))

        results.append(BenchmarkResult(
            method_name=method,
            domain_ppl=domain_ppl,
            lra_copy_acc=lra_acc,
            avg_latency_ms=latency,
            avg_active_paths=1.0,  # Static for non-dynamic methods
            activated_params_ratio=1.0,
        ))

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 80)
    header = f"{"Method":<25} {"Avg PPL":<12} {"LRA-512":<10} {"LRA-2048":<10} {"Latency(ms)":<12}"
    print(header)
    print("-" * 80)
    for r in results:
        avg_ppl = np.mean([v for v in r.domain_ppl.values() if not math.isnan(v)])
        lra_512 = r.lra_copy_acc.get(512, 0.0)
        lra_2048 = r.lra_copy_acc.get(2048, 0.0)
        print(f"{r.method_name:<25} {avg_ppl:<12.2f} {lra_512:<10.3f} {lra_2048:<10.3f} {r.avg_latency_ms:<12.2f}")
    print("=" * 80)