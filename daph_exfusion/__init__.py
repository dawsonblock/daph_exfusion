"""DAPH ExFusion — v4.3.6 (research prototype, not production-hardened)."""

__version__ = "2026.07.4.3.6"

# Core toolkit (always available)
from .merge_toolkit import (
    MemoryBankExFusionFFN,
    MemoryBankExFusionMamba,
    DAPHDecoderLayer,
    KFACFisherTracker,
    KFACConfig,
    GroupMergePolicy,
    DEFAULT_MAMBA_POLICIES,
    unified_calibration_loop,
    build_fisher_diagonals,
    incorporate_kfac_scores,
    mamba_grouped_merge,
    difficulty_weighted_fisher_merge_from_deltas,
    compute_ties_aligned_deltas,
    difficulty_weighted_dare_deltas,
    apply_dare_to_delta,
    elect_sign_mask,
    SwiGLUFFN,
)

# Adaptive top-p macro-router (PyTorch only, no MLX dependency)
from .adaptive_top_p_router import (
    AdaptiveTopPMacroRouter,
    DAPHDecoderLayerV2,
)

# Benchmark suite (always available)
from .benchmark import (
    MoEBenchmarkSuite,
    BenchmarkResult,
    run_ablation_study,
    print_results_table,
)

# Optional MLX stack (non-fatal if absent)
_mlx_available = False
try:
    from .mlx_inference import (
        MLXSwiGLUFFN,
        MLXMergedMamba,
        MLXFlashAttention,
        MLXRotaryEmbedding,
        MLXFNetBlock,
        MLXMacroRouter,
        MLXDAPHDecoderLayer,
        MLXAttentionPath,
        MLXAdaptiveTopPMacroRouter,
        KVCache,
        SSMState,
        ConvState,
        MLXStatefulDAPHDecoderLayer,
        MLXStatefulCausalLM,
        fused_swiglu_epilogue,
        mamba_selective_scan,
        mamba_selective_scan_reference,
        _ssm_prefill_step,
        ssm_prefill_loop,
        clean_pytorch_keys,
        pytorch_to_mlx,
        test_scan_correctness,
        test_scan_with_real_weights,
    )
    from .bridge import (
        extract_merged_state_dict,
        pt_to_mlx_array,
        load_mlx_model,
        validate_architecture_compatibility,
        build_mlx_from_merged_layer,
    )
    _mlx_available = True
except Exception:
    MLXSwiGLUFFN = None
    MLXMergedMamba = None
    MLXFlashAttention = None
    MLXRotaryEmbedding = None
    MLXFNetBlock = None
    MLXMacroRouter = None
    MLXDAPHDecoderLayer = None
    MLXAttentionPath = None
    MLXAdaptiveTopPMacroRouter = None
    KVCache = None
    SSMState = None
    ConvState = None
    MLXStatefulDAPHDecoderLayer = None
    MLXStatefulCausalLM = None
    fused_swiglu_epilogue = None
    mamba_selective_scan = None
    mamba_selective_scan_reference = None
    _ssm_prefill_step = None
    ssm_prefill_loop = None
    clean_pytorch_keys = None
    pytorch_to_mlx = None
    test_scan_correctness = None
    test_scan_with_real_weights = None
    extract_merged_state_dict = None
    pt_to_mlx_array = None
    load_mlx_model = None
    validate_architecture_compatibility = None
    build_mlx_from_merged_layer = None
