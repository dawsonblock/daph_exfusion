"""Tests for benchmark suite."""
import math
import pytest
import torch
import torch.nn as nn

from daph_exfusion.benchmark import (
    MoEBenchmarkSuite,
    BenchmarkResult,
    run_ablation_study,
    print_results_table,
)


class DummyModel(nn.Module):
    """Causal dummy model that handles both integer tokens and continuous embeddings."""
    def __init__(self, vocab_size=1000, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # If input is integer token IDs, project through embedding first
        if x.dtype in (torch.long, torch.int32, torch.int64):
            x = self.embed(x)
        return self.linear(x)


def test_benchmark_result_dataclass():
    r = BenchmarkResult(
        method_name="test",
        domain_ppl={"wiki": 10.0},
        lra_copy_acc={512: 0.5},
        avg_latency_ms=5.0,
        avg_active_paths=1.5,
        activated_params_ratio=0.8,
    )
    assert r.method_name == "test"
    assert r.domain_ppl["wiki"] == 10.0


def test_suite_init():
    model = DummyModel()
    suite = MoEBenchmarkSuite(model)
    assert suite.model is model
    assert suite.device == "cpu"


def test_latency_measurement():
    model = DummyModel(hidden_size=16)
    suite = MoEBenchmarkSuite(model)
    latency = suite.measure_average_latency_ms((2, 8, 16), steps=20)
    assert latency > 0
    assert not math.isnan(latency)


def test_lra_copy_eval():
    model = DummyModel(vocab_size=1000, hidden_size=16)
    suite = MoEBenchmarkSuite(model)
    results = suite.evaluate_lra_copy(seq_lengths=[16, 32], num_samples=5)
    assert 16 in results
    assert 32 in results
    assert 0 <= results[16] <= 1


def test_ablation_study_runs():
    """Ablation study should run without crashing."""
    def make_model():
        return DummyModel(hidden_size=16)

    experts = [DummyModel(hidden_size=16) for _ in range(2)]

    # Create dummy dataloaders
    class DummyLoader:
        def __init__(self, batches=2):
            self.batches = batches
        def __iter__(self):
            for _ in range(self.batches):
                yield {
                    "input_ids": torch.randint(0, 1000, (2, 8)),
                    "labels": torch.randint(0, 1000, (2, 8)),
                }
        def __len__(self):
            return self.batches

    dataloaders = {
        "wiki": DummyLoader(),
        "gov": DummyLoader(),
    }

    methods = ["average", "daph_exfusion"]
    results = run_ablation_study(make_model, experts, dataloaders, methods)

    assert len(results) == 2
    assert results[0].method_name == "average"
    assert results[1].method_name == "daph_exfusion"


def test_print_results_table():
    results = [
        BenchmarkResult(
            method_name="avg",
            domain_ppl={"a": 15.0, "b": 20.0},
            lra_copy_acc={512: 0.6, 2048: 0.3},
            avg_latency_ms=10.0,
            avg_active_paths=1.0,
            activated_params_ratio=1.0,
        ),
    ]
    # Should not raise
    print_results_table(results)
