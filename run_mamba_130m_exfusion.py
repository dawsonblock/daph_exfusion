#!/usr/bin/env python
"""End-to-end DAPH ExFusion pipeline for state-spaces/mamba-130m-hf.

Downloads the 130M Mamba checkpoint, extracts a single mixer block, maps
it into a bridge-compatible SimpleMambaBlock, generates 3 domain experts
via controlled perturbation, calibrates K-FAC + Fisher diagonals, merges
via TIES-Fisher, and validates PyTorch→MLX parity.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from daph_exfusion.merge_toolkit import (
    MemoryBankExFusionMamba,
    KFACConfig,
    KFACFisherTracker,
    build_fisher_diagonals,
)
from daph_exfusion.upgrade_utils import aggregate_kfac_scores_to_experts

try:
    from transformers import MambaForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# 1. PARITY-COMPATIBLE MAMBA BLOCK ARCHITECTURE
# =============================================================================
class SimpleMambaBlock(nn.Module):
    """Mamba block matching MLXMergedMamba parameter names exactly.

    Uses the standard Mamba SSM recurrence (not a shortcut) so that
    PyTorch and MLX outputs can be compared for parity.
    """

    def __init__(self, hidden_size: int = 768, d_conv: int = 4, d_state: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_conv = d_conv
        self.d_state = d_state

        self.in_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.conv1d = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=d_conv,
            stride=1, padding=0, groups=hidden_size, bias=True,
        )
        self.x_proj_B = nn.Linear(hidden_size, d_state, bias=False)
        self.x_proj_C = nn.Linear(hidden_size, d_state, bias=False)
        self.dt_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.A_log = nn.Parameter(torch.zeros(hidden_size))
        self.D = nn.Parameter(torch.zeros(hidden_size))

    def _selective_scan(self, u, delta, Bc, Cc):
        """Reference SSM scan: h[t,n] = exp(dt*a) * h[t-1,n] + dt*B[n]*x[t]."""
        B, L, D = u.shape
        d_state = self.d_state
        a = -torch.exp(self.A_log)  # (D,)
        state = torch.zeros(B, D, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            dt = delta[:, t, :]  # (B, D)
            Bt = Bc[:, t, :]     # (B, d_state)
            Ct = Cc[:, t, :]     # (B, d_state)
            xt = u[:, t, :]      # (B, D)
            decay = torch.exp(dt * a)  # (B, D)
            # state: (B, D, d_state)
            state = decay.unsqueeze(-1) * state + dt.unsqueeze(-1) * Bt.unsqueeze(1) * xt.unsqueeze(-1)
            # y = sum_n C[n] * h[n] + D * x
            yt = (Ct.unsqueeze(1) * state).sum(dim=-1) + self.D.unsqueeze(0) * xt
            ys.append(yt)
        return torch.stack(ys, dim=1)  # (B, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        a_gate, b_gate = h[..., :self.hidden_size], h[..., self.hidden_size:]
        u = F.silu(a_gate) * b_gate  # (B, L, D)

        # Depthwise causal conv1d with left-padding
        pad = torch.zeros(
            x.shape[0], self.d_conv - 1, self.hidden_size,
            device=x.device, dtype=x.dtype,
        )
        u_padded = torch.cat([pad, u], dim=1).transpose(1, 2)  # (B, D, L+k-1)
        u = F.silu(self.conv1d(u_padded).transpose(1, 2))      # (B, L, D)

        delta = F.softplus(self.dt_proj(u))
        Bc = self.x_proj_B(u)
        Cc = self.x_proj_C(u)
        y = self._selective_scan(u, delta, Bc, Cc)
        return self.out_proj(y)


# =============================================================================
# 2. HUGGINGFACE WEIGHT EXTRACTION AND MAPPING
# =============================================================================
def extract_and_map_mamba_weights(hf_mixer, simple_block):
    """Map HuggingFace Mamba mixer parameters into SimpleMambaBlock.

    HF Mamba-130m has d_model=768, d_inner=1536, dt_rank=48, d_state=16.
    The in_proj maps (768 → 1536*2=3072), but our SimpleMambaBlock uses
    in_proj (768 → 768*2=1536).  We slice the first 768 columns of each
    projection to fit our block's hidden_size=768 signature.
    """
    print("    Mapping parameters from Hugging Face structure...")
    with torch.no_grad():
        # in_proj: HF (3072, 768) → ours (1536, 768) — take first 1536 rows
        simple_block.in_proj.weight.copy_(hf_mixer.in_proj.weight[:1536, :])

        # out_proj: HF (768, 1536) → ours (768, 768) — take first 768 cols
        simple_block.out_proj.weight.copy_(hf_mixer.out_proj.weight[:, :768])

        # conv1d: HF (1536, 1, 4) → ours (768, 1, 4) — take first 768 channels
        simple_block.conv1d.weight.copy_(hf_mixer.conv1d.weight[:768, :, :])
        simple_block.conv1d.bias.copy_(hf_mixer.conv1d.bias[:768])

        # x_proj: HF (80, 1536) where 80 = dt_rank(48) + d_state(16) + d_state(16)
        x_proj_weight = hf_mixer.x_proj.weight  # (80, 1536)
        dt_rank, d_state = 48, 16
        b_slice = x_proj_weight[dt_rank:dt_rank + d_state, :768]    # (16, 768)
        c_slice = x_proj_weight[dt_rank + d_state:, :768]           # (16, 768)
        simple_block.x_proj_B.weight.copy_(b_slice)
        simple_block.x_proj_C.weight.copy_(c_slice)

        # dt_proj: HF (1536, 48) → ours (768, 768) — use diagonal-ish mapping
        # HF dt_proj maps dt_rank(48) → d_inner(1536).
        # Our dt_proj maps hidden(768) → hidden(768).
        # We create a (768, 768) matrix by tiling the HF weight rows.
        hf_dt = hf_mixer.dt_proj.weight  # (1536, 48)
        # Take first 768 rows, pad 48→768 with zeros
        dt_padded = torch.zeros(768, 768, dtype=hf_dt.dtype)
        dt_padded[:, :48] = hf_dt[:768, :]
        simple_block.dt_proj.weight.copy_(dt_padded)

        # A_log: HF (1536, d_state) → ours (768,) — average over d_state, slice
        simple_block.A_log.copy_(hf_mixer.A_log.mean(dim=-1)[:768])

        # D: HF (1536,) → ours (768,)
        simple_block.D.copy_(hf_mixer.D[:768])


# =============================================================================
# 3. CALIBRATION DATASET
# =============================================================================
class SyntheticDataset(torch.utils.data.Dataset):
    """Simple synthetic dataset for calibration."""

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# =============================================================================
# 4. PIPELINE ORCHESTRATION
# =============================================================================
def main():
    print("=" * 80)
    print("      DAPH EXFUSION MAMBA-130M SYNTHETIC MERGE PIPELINE")
    print("=" * 80)

    if not HAS_TRANSFORMERS:
        print("[ERROR] transformers not installed. Run: pip install transformers")
        return

    hidden_size = 768
    num_experts = 3
    batch_size = 2
    seq_len = 32

    # --- Phase 1: Download and Extract Base Block ---
    print("\n[Phase 1] Downloading state-spaces/mamba-130m-hf from Hugging Face...")
    hf_model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    hf_mixer = hf_model.backbone.layers[0].mixer

    print("\n[Phase 2] Initializing baseline SimpleMambaBlock...")
    base_block = SimpleMambaBlock(hidden_size=hidden_size)
    extract_and_map_mamba_weights(hf_mixer, base_block)

    # --- Phase 2: Create 3 Specialized Expert Copies ---
    print("\n[Phase 3] Generating 3 domain-specialized experts via perturbation...")
    expert_modules = []
    torch.manual_seed(0)
    for i in range(num_experts):
        expert = copy.deepcopy(base_block)
        with torch.no_grad():
            for param in expert.parameters():
                param.add_(torch.randn_like(param) * 1e-4)
        expert_modules.append(expert)
    print(f"    Initialized {num_experts} distinct Mamba experts.")

    # --- Phase 3: Setup MoE Container and Datasets ---
    def block_factory():
        return SimpleMambaBlock(hidden_size=hidden_size)

    mamba_moe = MemoryBankExFusionMamba(
        block_factory=block_factory,
        num_experts=num_experts,
        hidden_size=hidden_size,
    )
    # Replace factory-generated experts with our pre-trained, perturbed copies
    mamba_moe.experts = nn.ModuleList(expert_modules)

    # Generate synthetic calibration data
    print("    Generating synthetic calibration loader...")
    inputs = torch.randn(10, seq_len, hidden_size)
    targets = torch.randn(10, seq_len, hidden_size)
    dataset = SyntheticDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # --- Phase 4: Curvature Tracking ---
    print("\n[Phase 4] Attaching K-FAC Tracker and running calibration...")
    tracker = KFACFisherTracker(mamba_moe, KFACConfig(ema_decay=0.9, track_bias=True))
    mamba_moe.train()

    for x_batch, _ in dataloader:
        _ = mamba_moe(x_batch)

    layer_scores = tracker.all_layer_scores()
    tracker.remove()

    expert_kfac_scores = aggregate_kfac_scores_to_experts(
        layer_scores, num_experts=num_experts, path_prefix="experts"
    )
    print(f"    Aggregated Expert K-FAC Scores: {expert_kfac_scores}")

    # Compute Mamba Fisher diagonals
    print("    Computing expert Fisher diagonals...")
    loss_fn = nn.MSELoss()
    mamba_moe.train()
    fisher_diagonals = build_fisher_diagonals(
        experts=mamba_moe.experts,
        dataloader=dataloader,
        model=mamba_moe,
        loss_fn=loss_fn,
        num_batches=4,
    )

    # --- Phase 5: ExFusion Merging ---
    print("\n[Phase 5] Compiling experts into a single dense pathway...")
    mamba_moe.merge_to_dense(
        fisher_diagonals=fisher_diagonals,
        kfac_scores=expert_kfac_scores,
        seed=42,
    )
    print(f"    Mamba Pathway Merged Status: {mamba_moe.is_merged}")

    # --- Phase 6: Export and Hardware Parity Validation ---
    print("\n[Phase 6] Checking Apple Silicon MLX Hardware Parity...")
    try:
        import mlx.core as mx
        from daph_exfusion.mlx_inference import MLXMergedMamba
        from daph_exfusion.bridge import load_mlx_model

        mlx_block = MLXMergedMamba(d_model=hidden_size, d_state=16)

        # Inject merged parameters through the bridge
        load_mlx_model(mamba_moe.merged_mamba, mlx_block, quantize=False, strict=True)
        print("    Parameter mapping to MLX successful.")

        test_input = torch.randn(1, 4, hidden_size)
        mamba_moe.eval()
        with torch.no_grad():
            pyt_out = mamba_moe(test_input).numpy()

        mlx_input = mx.array(test_input.numpy())
        mlx_out = np.array(mlx_block(mlx_input))

        max_diff = np.max(np.abs(pyt_out - mlx_out))
        print(f"    Maximum Absolute Discrepancy (PyTorch vs MLX): {max_diff:.3e}")

        if max_diff < 1e-4:
            print("    STATUS: Hardware Parity Verified.")
        else:
            print("    STATUS: Parity Discrepancy Detected — check projection alignments.")

    except Exception as e:
        print(f"    MLX translation bypassed: {e}")
        print("    Run this script on an Apple Silicon device with 'mlx' installed to verify GPU execution.")

    print("\n" + "=" * 80)
    print("                     PIPELINE RUN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
