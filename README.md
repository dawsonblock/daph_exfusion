<div align="center">

# DAPH ExFusion

**Difficulty-Aware Mixture-of-Experts merging, with native MLX inference on Apple Silicon.**

[![Version](https://img.shields.io/badge/version-2026.07.4.5.0-blue)](./PATCH_NOTES.md)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](./pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)](#known-limitations)

</div>

---

DAPH ExFusion is a research system for **merging Mixture-of-Experts (MoE) models into a single dense model** and running the merged weights through a **native Metal-accelerated MLX inference stack** on Apple Silicon.

The merge toolkit (PyTorch, CPU/CUDA) implements a difficulty-aware DARE → TIES → Fisher pipeline with per-group Mamba policies and K-FAC preconditioning. The inference stack (MLX) provides fused Metal kernels for SwiGLU, Mamba selective scan, flash attention, and an FNet cheap path, glued together by an adaptive top-p macro-router.

> **This is a research prototype, not a production system.** It is importable, tested, and honest about its limitations — see [Known Limitations](#known-limitations).

---

## Highlights

- **Difficulty-aware merging** — DARE drop rates and Fisher weighting are modulated by per-expert difficulty, so easy experts are pruned harder and hard experts are preserved.
- **Per-group Mamba policies** — `A_log` / `D` are protected with low drop rates and unanimous-only sign elections; projections (`in_proj`, `out_proj`, `x_proj`, `dt_proj`) get standard rates. Sign elections use pure **majority** voting (not magnitude-weighted) so large outliers can't swamp the vote.
- **K-FAC Fisher tracker** — runs in **diagonal-only mode by default** (a few MB per layer per expert instead of ~1.6 GB at `d_model=4096`), with a full-covariance ablation path.
- **Compiled MLX pre-fill** — the SSM pre-fill recurrence is wrapped in `@mx.compile`, fusing each timestep's elementwise chain into a single kernel call.
- **Strict PT→MLX bridge** — `validate_architecture_compatibility` fails loudly on missing keys or shape mismatches instead of silently producing nonsense inference.
- **Adaptive top-p macro-router** — predicts per-token difficulty and activates only the paths needed (attention / ExFusion / cheap FNet), so easy tokens skip the expensive paths.
- **Fused Metal kernels** — SwiGLU epilogue and Mamba selective scan are implemented as `mx.fast.metal_kernel` for single-pass computation.

---

## Architecture

```text
PyTorch Training
   ├─ MemoryBankExFusionFFN   (N experts)
   ├─ MemoryBankExFusionMamba (N experts)
   └─ DAPHDecoderLayer[V2]    (macro router)
          │
          ▼  merge_exfusion_paths()
   ├─ DARE → TIES → Fisher pipeline
   ├─ K-FAC score pre-modulation (diagonal-only by default)
   └─ Mamba per-group policies (A_log/D protected, majority sign election)
          │
          ▼  bridge.py  (strict key + shape validation)
   Merged dense state_dict
          │
          ▼  load_mlx_model()
MLX Inference (Apple Silicon)
   ├─ MLXSwiGLUFFN            (fused Metal kernel)
   ├─ MLXMergedMamba          (selective scan kernel + @mx.compile pre-fill)
   ├─ MLXFlashAttention       (separate Q/K/V/O projections)
   ├─ MLXFNetBlock            (FFT cheap path)
   ├─ MLXDAPHDecoderLayer     (trace-safe mx.where)
   └─ MLXStatefulDAPHDecoderLayer (KV-cache + SSM state)
```

---

## Quick Start

### Install

```bash
# CPU / CUDA — merge toolkit only
pip install -e ".[dev]"

# Apple Silicon — merge toolkit + native MLX inference
pip install -e ".[dev,mlx]"
```

### Run the demo

```bash
python -m daph_exfusion.demo
```

### Run the tests

```bash
pytest tests/ -v
```

### Merge experts (PyTorch)

```python
from daph_exfusion import MemoryBankExFusionMamba

mamba = MemoryBankExFusionMamba(block_factory, num_experts=3, hidden_size=8)
fishers = [
    {name: torch.rand_like(p) + 1e-3 for name, p in e.named_parameters()}
    for e in mamba.experts
]
mamba.merge_to_dense(fisher_diagonals=fishers, seed=42)
```

### Export to MLX (Apple Silicon)

```python
from daph_exfusion.bridge import load_mlx_model

mlx_model = build_mlx_from_merged_layer(...)
load_mlx_model(pytorch_module, mlx_model, strict=True)  # raises on any mismatch
```

---

## Macro-Router Variants

| Layer | Router Style | Use Case |
|-------|-------------|----------|
| `DAPHDecoderLayer` | Static softmax blending (always computes all paths) | Baseline, deterministic latency |
| `DAPHDecoderLayerV2` | **Adaptive top-p** (difficulty-modulated threshold, computes only selected paths) | Dynamic inference, variable latency |

**Adaptive top-p routing** predicts input difficulty per token. Higher difficulty → higher cumulative-probability threshold → more paths active (more probability mass must be accumulated). Easy tokens may use only the cheap FNet path; hard tokens activate attention + ExFusion + cheap.

Expert Choice is not used at the macro level (only 3 paths, qualitatively different compute types).

---

## Benchmark Protocol

```python
from daph_exfusion.benchmark import MoEBenchmarkSuite, run_ablation_study, print_results_table

suite = MoEBenchmarkSuite(model)
ppl  = suite.evaluate_perplexity(dataloader)
lra  = suite.evaluate_lra_copy(seq_lengths=[512, 1024, 2048])
lat  = suite.measure_average_latency_ms((2, 128, 512))
```

### Recommended Evaluation

**Models & Tasks**
- Base model: Mamba-130M fine-tuned on three domains:
  - Expert A: Wikitext-103 (standard LM)
  - Expert B: GovReport (long-document summarization)
  - Expert C: QMSum (query-based meeting summarization)
- SSM-sensitive task: Long-Range Arena Copy (LRA-Copy) at 512 / 1024 / 2048

**Compared Methods**
1. Simple averaging (lower bound)
2. TIES only (trim + sign election, no difficulty)
3. DARE + TIES (standard pipeline, uniform drop)
4. Fisher merging (uniform memory-bank weights + Fisher diagonals)
5. DAPH ExFusion — full difficulty-aware pipeline with SSM group policies (fixed top-1 macro router)
6. DAPH ExFusion + adaptive top-p macro routing (proposed)

**Ablations of DAPH**
- Remove difficulty modulation (all `d_i = 1`)
- Remove SSM soft merge (strict TIES on `A_log` / `D`)
- Remove SSM Fisher boost (`γ_ssm = 1`)
- Remove macro-routing entirely (always use ExFusion path)

**Metrics**

| Metric | Why |
|--------|-----|
| Per-domain perplexity (PPL) | Retention of each expert's specialization |
| LRA-Copy accuracy | Preservation of long-range SSM dynamics |
| Avg latency per token (ms) | Inference speed gain from dynamic routing |
| Activated parameters per token | Compute reduction quantification |
| Memory-bank divergence | Online adaptation quality |

---

## Project Structure

```text
daph_exfusion/
├── __init__.py                 # Lazy imports (MLX optional)
├── merge_toolkit.py            # K-FAC, DARE, TIES, Fisher, Mamba merge, calibration
├── adaptive_top_p_router.py    # Adaptive top-p macro-router + DAPHDecoderLayerV2
├── mlx_inference.py            # Metal kernels, MLX layers, adaptive router, stateful decoder
├── bridge.py                   # Strict PyTorch → MLX key + shape validation
├── benchmark.py                # Evaluation harness + ablation study runner
├── demo.py                     # End-to-end example
├── orchestrator.py             # Automated merge pipeline
└── upgrade_utils.py            # Robust policy lookup + K-FAC score aggregation
tests/
├── test_import.py              # compileall + import smoke tests
├── test_merge_toolkit.py       # Bias preservation, K-FAC mismatch, seed determinism
├── test_adaptive_router.py     # Top-p routing, DAPHDecoderLayerV2 forward + merge
├── test_mlx_adaptive_router.py # MLX adaptive router, KV-cache, SSM state, stateful decoder
├── test_benchmark.py           # Benchmark suite + ablation study tests
├── test_bridge.py              # Key mapping + validation tests
└── test_safety_gates.py        # v4.2.1 regression gates (sign-mode, pre-fill, K-FAC, bridge)
```

---

## Known Limitations

This is a research prototype. The limitations below are real and documented rather than hidden.

1. **K-FAC is layer-level, not expert-level.** `incorporate_kfac_scores()` requires pre-aggregated per-expert scores; use `aggregate_kfac_scores_to_experts()` from `upgrade_utils.py`.
2. **Macro router is a simple difficulty predictor**, not a learned gating network.
3. **Mamba `block_factory` must expose standard parameters** (`in_proj`, `out_proj`, `x_proj`, `dt_proj`, `A_log`, `D`) for grouped merge policies to apply.
4. **MLX conversion requires exact architecture parity.** The bridge uses strict key mapping; non-standard PyTorch factories require updating the key map in `bridge.py`.
5. **No distributed training support** — single-device merge toolkit only.
6. **Stateful decoder is skeletal** — `MLXStatefulDAPHDecoderLayer` provides the container (KV-cache + SSM state), but full autoregressive state management needs wiring for your specific attention/Mamba implementations.

---

## Changelog

See [PATCH_NOTES.md](./PATCH_NOTES.md) for the full history.

### v4.2.1 (2026-07-05)

- **Sign-election fix** — all Mamba groups default to `sign_mode="majority"`.
- **Compiled pre-fill** — SSM pre-fill step wrapped in `@mx.compile`.
- **Diagonal-only K-FAC** — `KFACConfig.diagonal_only=True` by default (~1.6 GB → few MB per layer).
- **Strict bridge validation** — missing keys raise instead of silently `continue`-ing.
- Regression gates added in `tests/test_safety_gates.py`.

---

## License

MIT License — see [LICENSE](./LICENSE).
