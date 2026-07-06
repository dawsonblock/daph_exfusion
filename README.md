# DAPH ExFusion — MoE Merge Toolkit & Native Inference

**Version:** 2026.07.4.2  
**Status:** Research prototype — importable and tested, but not production-hardened.

---

## What this is

A research system for training, merging, and deploying Mixture-of-Experts (MoE) models. The PyTorch merge toolkit works on any CPU/CUDA machine. MLX inference requires Apple Silicon and the `mlx` package.

This is **not** a production system. It is a consolidated research prototype with explicit known limitations.

---

## Architecture

```
PyTorch Training
   ├─ MemoryBankExFusionFFN  (N experts)
   ├─ MemoryBankExFusionMamba (N experts)
   └─ DAPHDecoderLayer / DAPHDecoderLayerV2 (macro router)
          │
          ▼  merge_exfusion_paths()
   ├─ DARE → TIES → Fisher pipeline
   ├─ K-FAC score pre-modulation
   └─ Mamba per-group policies (A_log/D protected)
          │
          ▼  bridge.py
   Merged dense state_dict
          │
          ▼  load_mlx_model()
MLX Inference (Apple Silicon only)
   ├─ MLXSwiGLUFFN (fused Metal kernel)
   ├─ MLXMergedMamba (selective scan kernel)
   ├─ MLXFlashAttention (separate Q/K/V/O projections)
   ├─ MLXFNetBlock (FFT cheap path)
   ├─ MLXDAPHDecoderLayer (trace-safe mx.where)
   └─ MLXStatefulDAPHDecoderLayer (stateful: KV-cache + SSM state)
```

---

## Macro-Router Variants

Two decoder layer implementations are provided:

| Layer | Router Style | Use Case |
|-------|-------------|----------|
| `DAPHDecoderLayer` | Static softmax blending (always computes all paths) | Baseline, deterministic latency |
| `DAPHDecoderLayerV2` | **Adaptive top-p** (difficulty-modulated threshold, computes only selected paths) | Dynamic inference, variable latency |

**Adaptive top-p routing** (`AdaptiveTopPMacroRouter` / `MLXAdaptiveTopPMacroRouter`):
- Predicts input difficulty per token.
- Higher difficulty → lower cumulative-probability threshold → more paths active.
- Easy tokens may use only the cheap FNet path; hard tokens activate attention + ExFusion + cheap.
- This is a lightweight extension of DynMoE-style cumulative-threshold routing.

**Expert Choice** is not used at the macro level (only 3 paths, qualitatively different compute types). It may be added later inside the ExFusion paths if scaling to 8–16 sub-experts.

---

## Quick Start

```bash
pip install -e ".[dev]"          # CPU-only
# On Apple Silicon:
pip install -e ".[dev,mlx]"
```

Run tests:
```bash
pytest tests/ -v
```

Run the demo:
```bash
python -m daph_exfusion.demo
```

---

## Benchmark Protocol

To evaluate DAPH ExFusion rigorously, use the provided benchmark suite:

```python
from daph_exfusion.benchmark import MoEBenchmarkSuite, run_ablation_study, print_results_table

suite = MoEBenchmarkSuite(model)
ppl = suite.evaluate_perplexity(dataloader)
lra = suite.evaluate_lra_copy(seq_lengths=[512, 1024, 2048])
latency = suite.measure_average_latency_ms((2, 128, 512))
```

### Recommended Evaluation

**Models & Tasks:**
- Base model: Mamba-130M fine-tuned on three domains:
  - Expert A: Wikitext-103 (standard LM)
  - Expert B: GovReport (long-document summarization)
  - Expert C: QMSum (query-based meeting summarization)
- SSM-sensitive task: Long-Range Arena Copy (LRA-Copy) at 512/1024/2048

**Compared Methods:**
1. Simple averaging (lower bound)
2. TIES only (trim + sign election, no difficulty)
3. DARE + TIES (standard pipeline, uniform drop)
4. Fisher merging (uniform memory-bank weights + Fisher diagonals)
5. DAPH ExFusion — full difficulty-aware pipeline with SSM group policies (fixed top-1 macro router)
6. DAPH ExFusion + adaptive top-p macro routing (proposed)

**Ablations of DAPH:**
- Remove difficulty modulation (all d_i = 1)
- Remove SSM soft merge (strict TIES on A_log/D)
- Remove SSM Fisher boost (γ_ssm = 1)
- Remove macro-routing entirely (always use ExFusion path)

**Metrics:**
| Metric | Why |
|--------|-----|
| Per-domain perplexity (PPL) | Retention of each expert's specialization |
| LRA-Copy accuracy | Preservation of long-range SSM dynamics |
| Avg latency per token (ms) | Inference speed gain from dynamic routing |
| Activated parameters per token | Compute reduction quantification |
| Memory-bank divergence | Online adaptation quality |

---

## Known Limitations (Honest)

1. **K-FAC is layer-level, not expert-level.**  
   `incorporate_kfac_scores()` requires you to pre-aggregate K-FAC layer scores into per-expert scores. We do not yet provide that aggregation.

2. **Macro router is a simple difficulty predictor.**  
   It is sufficient for research prototyping but is not a learned gating network.

3. **Mamba block_factory must expose standard parameters.**  
   `in_proj`, `out_proj`, `x_proj`, `dt_proj`, `A_log`, `D` are required for grouped merge policies to apply.

4. **MLX conversion requires exact architecture parity.**  
   The bridge uses strict key mapping. If your PyTorch attention or Mamba factories deviate from the demo conventions, you must update `_PYT_MLX_KEY_MAP` in `bridge.py`.

5. **No distributed training support.**  
   This is a single-device merge toolkit.

6. **Stateful decoder (KV-cache, SSM state) is skeletal.**  
   `MLXStatefulDAPHDecoderLayer` provides the container but full autoregressive state management requires additional wiring for your specific attention and Mamba implementations.

---

## File Structure

```
daph_exfusion/
├── __init__.py                   # Lazy imports (MLX optional)
├── merge_toolkit.py              # K-FAC, DARE, TIES, Fisher, Mamba merge, calibration
├── adaptive_top_p_router.py      # Adaptive top-p macro-router + DAPHDecoderLayerV2
├── mlx_inference.py              # Metal kernels, MLX native layers, adaptive router, stateful decoder
├── bridge.py                     # Explicit PyTorch → MLX key mapping
├── benchmark.py                  # Evaluation harness + ablation study runner
├── demo.py                       # Honest end-to-end example
tests/
├── test_import.py                # compileall + import smoke tests
├── test_merge_toolkit.py         # Bias preservation, K-FAC mismatch, seed determinism
├── test_adaptive_router.py       # Top-p routing, DAPHDecoderLayerV2 forward + merge
├── test_mlx_adaptive_router.py   # MLX adaptive router, KV-cache, SSM state, stateful decoder
├── test_benchmark.py             # Benchmark suite + ablation study tests
└── test_bridge.py                # Key mapping + validation tests
```

---

## License

MIT License
