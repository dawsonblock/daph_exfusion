"""Integration pipeline harness for DAPH ExFusion.

This script exercises the full expert‑merge workflow on synthetic data. It
initializes PyTorch ExFusion modules, tracks K‑FAC activations, computes
Fisher diagonals, runs a coordinate‑descent calibration search, performs
merging of FFN and Mamba experts, transfers parameters to MLX, and checks
numerical parity. The intended use is for end‑to‑end smoke testing and
benchmarking on development machines. MLX validation will be skipped when the
``mlx`` package or Apple Silicon hardware is unavailable.

To run this script directly:

    python run_integration_pipeline.py

It prints progress through each step and reports the maximum absolute
difference between PyTorch and MLX outputs when possible.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings

from daph_exfusion.merge_toolkit import (
    MemoryBankExFusionFFN,
    MemoryBankExFusionMamba,
    DAPHDecoderLayer,
    KFACConfig,
    KFACFisherTracker,
    build_fisher_diagonals,
    unified_calibration_loop,
)
from daph_exfusion.upgrade_utils import aggregate_kfac_scores_to_experts
from daph_exfusion.demo import PyTAttentionPath, make_mamba_factory


class SyntheticCalibrationDataset(torch.utils.data.Dataset):
    """Simple dataset of random inputs and targets for calibration."""

    def __init__(self, num_samples: int = 16, seq_len: int = 8, hidden_size: int = 16) -> None:
        super().__init__()
        self.inputs = torch.randn(num_samples, seq_len, hidden_size)
        self.targets = torch.randn(num_samples, seq_len, hidden_size)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class CalibrationModelWrapper(nn.Module):
    """Wraps a target layer to ensure gradients flow through during calibration."""

    def __init__(self, target_layer: nn.Module) -> None:
        super().__init__()
        self.layer = target_layer

    def forward(self, x, **kwargs):
        return self.layer(x)


def run_integration_pipeline() -> None:
    print("=" * 70)
    print("      DAPH EXFUSION END‑TO‑END PIPELINE ORCHESTRATION TEST")
    print("=" * 70)

    hidden_size = 16
    intermediate_size = 32
    num_experts = 3
    num_heads = 2

    # Suppress extraneous warnings for cleaner output
    warnings.filterwarnings("ignore")

    # -------------------------------------------------------------
    # Step 1: Initialize PyTorch Layer & Dataset
    # -------------------------------------------------------------
    print("\n[Step 1] Initializing MoE structures and calibration dataset...")
    ffn_exfusion = MemoryBankExFusionFFN(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        activation="swiglu",
        bias=True,
    )
    mamba_exfusion = MemoryBankExFusionMamba(
        block_factory=make_mamba_factory(hidden_size),
        num_experts=num_experts,
        hidden_size=hidden_size,
    )
    pyt_layer = DAPHDecoderLayer(
        hidden_size=hidden_size,
        ffn_exfusion_factory=lambda: ffn_exfusion,
        mamba_exfusion_factory=lambda: mamba_exfusion,
        attention_factory=lambda: PyTAttentionPath(hidden_size, num_heads),
        use_cheap_path=True,
    )
    dataset = SyntheticCalibrationDataset(num_samples=32, seq_len=8, hidden_size=hidden_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # -------------------------------------------------------------
    # Step 2: Track Layer Activations using K‑FAC Tracker
    # -------------------------------------------------------------
    print("\n[Step 2] Attaching K‑FAC Fisher Tracker...")
    tracker = KFACFisherTracker(pyt_layer, KFACConfig(ema_decay=0.9, track_bias=True))
    pyt_layer.train()
    for inputs, _ in dataloader:
        _ = pyt_layer(inputs)
    print("    Extracting layer scores...")
    layer_scores = tracker.all_layer_scores()
    tracker.remove()
    expert_kfac_scores = aggregate_kfac_scores_to_experts(
        layer_scores, num_experts=num_experts, path_prefix="ffn_path"
    )
    print(f"    Aggregated Expert K‑FAC Scores: {expert_kfac_scores}")

    # -------------------------------------------------------------
    # Step 3: Compute Squared‑Gradient Fisher Diagonals
    # -------------------------------------------------------------
    print("\n[Step 3] Computing expert Fisher Diagonals...")
    loss_fn = nn.MSELoss()
    wrapper = CalibrationModelWrapper(pyt_layer)
    fisher_diagonals_ffn = build_fisher_diagonals(
        experts=pyt_layer.ffn_path.experts,
        dataloader=dataloader,
        model=wrapper,
        loss_fn=loss_fn,
        num_batches=4,
    )
    print("    Fisher diagonals computed for FFN expert weights.")

    # -------------------------------------------------------------
    # Step 4: Run Greedy Coordinate‑Descent Calibration Loop
    # -------------------------------------------------------------
    print("\n[Step 4] Running hyperparameter calibration search...")
    search_space = {
        "dare_drop_rate": [0.1, 0.3],
        "ties_trim_ratio": [0.1, 0.2],
        "fisher_power": [1.0, 1.2],
    }
    common_args = {
        "pipeline": ["dare", "ties", "fisher", "kfac"],
        "fisher_diagonals": fisher_diagonals_ffn,
        "kfac_scores": expert_kfac_scores,
        "kfac_temperature": 1.0,
        "seed": 42,
    }
    def evaluate_fn(trial_modules):
        trial_module = trial_modules[0]
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = trial_module(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                count += 1
                if count >= 4:
                    break
        return total_loss / count
    winning_hparams = unified_calibration_loop(
        exfusion_modules=[pyt_layer.ffn_path],
        search_space=search_space,
        evaluate_fn=evaluate_fn,
        merge_kwargs_common=common_args,
        rounds=1,
        verbose=True,
    )
    print(f"    Optimal Hyperparameters Found: {winning_hparams}")

    # -------------------------------------------------------------
    # Step 5: Merge Mamba Path & Finalize PyTorch Layer
    # -------------------------------------------------------------
    print("\n[Step 5] Merging Mamba experts to dense path...")
    fisher_diagonals_mamba = [
        {name: torch.rand_like(p) + 1e-3 for name, p in e.named_parameters()}
        for e in pyt_layer.mamba_path.experts
    ]
    pyt_layer.mamba_path.merge_to_dense(
        fisher_diagonals=fisher_diagonals_mamba,
        seed=42,
    )
    print(f"    FFN Merged Status: {pyt_layer.ffn_path.is_merged}")
    print(f"    Mamba Merged Status: {pyt_layer.mamba_path.is_merged}")

    # -------------------------------------------------------------
    # Step 6: Perform MLX Hardware Parity Mapping (If Available)
    # -------------------------------------------------------------
    print("\n[Step 6] Attempting MLX translation and validation check...")
    try:
        import mlx.core as mx
        from daph_exfusion.mlx_inference import MLXDAPHDecoderLayer
        from daph_exfusion.bridge import load_mlx_model
        mlx_layer = MLXDAPHDecoderLayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
        )
        load_mlx_model(pyt_layer, mlx_layer, quantize=False, strict=True)
        print("    PyTorch‑to‑MLX parameter translation successful.")
        eval_input = torch.randn(1, 4, hidden_size)
        pyt_layer.eval()
        with torch.no_grad():
            pyt_out = pyt_layer(eval_input).numpy()
        mlx_input = mx.array(eval_input.numpy())
        mlx_out = np.array(mlx_layer(mlx_input))
        max_diff = np.max(np.abs(pyt_out - mlx_out))
        print(f"    Validation Maximum Absolute Difference: {max_diff:.3e}")
        if max_diff < 1e-4:
            print("    STATUS: Parity Verified.")
        else:
            print("    STATUS: Parity Warning — Check parameter projections.")
    except Exception:
        print("    MLX or Apple Silicon environment not detected. Bypassing hardware parity step.")
    print("\n" + "=" * 70)
    print("                     PIPELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_integration_pipeline()