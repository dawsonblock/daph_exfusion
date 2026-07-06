"""
Honest end-to-end DAPH ExFusion Demo
======================================
1. Build a PyTorch DAPH model with expert FFN + Mamba paths.
2. Run forward passes to populate memory banks.
3. Merge experts using DARE -> TIES -> Fisher.
4. (Optional) Convert to MLX — only on Apple Silicon with mlx installed.

This demo proves the PyTorch merge pipeline works. It does NOT claim to
prove MLX numerical parity unless you are running on Apple Silicon.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from daph_exfusion.merge_toolkit import (
    MemoryBankExFusionFFN,
    MemoryBankExFusionMamba,
    DAPHDecoderLayer,
)


class SimpleAttention(nn.Module):
    """Clean attention block with independent Q/K/V/O projections."""
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class PyTAttentionPath(nn.Module):
    """Wrapper matching MLX namespace layout exactly."""
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = SimpleAttention(hidden_size, num_heads)

    def forward(self, x, mask=None):
        return self.attn(self.norm(x), mask)


def make_mamba_factory(hidden_size: int, d_conv: int = 4):
    def factory():
        class SimpleMambaBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
                self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                # Standard depthwise 1D convolution (kernel=4, causal)
                self.conv1d = nn.Conv1d(
                    hidden_size, hidden_size, kernel_size=d_conv,
                    stride=1, padding=0, groups=hidden_size, bias=True,
                )
                self.x_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.dt_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.A_log = nn.Parameter(torch.zeros(hidden_size))
                self.D = nn.Parameter(torch.zeros(hidden_size))

            def forward(self, x):
                h = self.in_proj(x)
                a_gate, b_gate = h[..., :hidden_size], h[..., hidden_size:]
                u = torch.silu(a_gate) * b_gate  # (B, L, D)
                # Causal depthwise conv1d: left-pad, conv, SiLU
                pad = torch.zeros(x.shape[0], d_conv - 1, hidden_size,
                                  device=x.device, dtype=x.dtype)
                u_padded = torch.cat([pad, u], dim=1).transpose(1, 2)  # (B, D, L+k-1)
                u = torch.silu(self.conv1d(u_padded).transpose(1, 2))  # (B, L, D)
                delta = F.softplus(self.dt_proj(u))
                y = u + self.D * u
                return self.out_proj(y)
        return SimpleMambaBlock()
    return factory


def build_demo_model(hidden_size=64, intermediate_size=128, num_experts=3, num_heads=4):
    ffn = MemoryBankExFusionFFN(
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_experts=num_experts, activation="swiglu", bias=True,
    )
    mamba = MemoryBankExFusionMamba(
        block_factory=make_mamba_factory(hidden_size),
        num_experts=num_experts, hidden_size=hidden_size,
    )
    layer = DAPHDecoderLayer(
        hidden_size=hidden_size,
        ffn_exfusion_factory=lambda: ffn,
        mamba_exfusion_factory=lambda: mamba,
        attention_factory=lambda: PyTAttentionPath(hidden_size, num_heads),
        use_cheap_path=True,
    )
    return layer


def generate_dummy_fisher_diagonals(module, num_experts: int):
    diagonals = []
    for e in module.experts:
        d = {}
        for name, param in e.named_parameters():
            d[name] = torch.rand_like(param.data) + 1e-3
        diagonals.append(d)
    return diagonals


def main():
    hidden_size, intermediate_size = 64, 128
    num_experts, num_heads = 3, 4
    batch_size, seq_len = 2, 8

    print("=" * 60)
    print("DAPH ExFusion v4.2 — End-to-End Execution Demo")
    print("=" * 60)

    print("\n[1] Instantiating MoE PyTorch Model Layer...")
    layer = build_demo_model(hidden_size, intermediate_size, num_experts, num_heads)
    print(f"    hidden_size={hidden_size}, experts={num_experts}, heads={num_heads}")

    print("\n[2] Accumulating Expert Activations over 5 forward iterations...")
    x = torch.randn(batch_size, seq_len, hidden_size)
    for _ in range(5):
        out = layer(x)
    print(f"    PyTorch execution complete. Output tensor shape: {out.shape}")
    print(f"    FFN Routing metrics: {layer.ffn_path.memory_bank.detach().cpu().numpy().round(3)}")
    print(f"    Mamba Routing metrics: {layer.mamba_path.memory_bank.detach().cpu().numpy().round(3)}")

    print("\n[3] Triggering DARE -> TIES -> Fisher merging pipeline...")
    fisher_ffn = generate_dummy_fisher_diagonals(layer.ffn_path, num_experts)
    layer.ffn_path.merge_to_dense(
        pipeline=["dare", "ties", "fisher"],
        fisher_diagonals=fisher_ffn,
        seed=0,
    )
    print("    FFN merged (bias preserved)" if layer.ffn_path.merged_ffn.up.bias is not None else "    FFN merged")

    fisher_mamba = generate_dummy_fisher_diagonals(layer.mamba_path, num_experts)
    layer.mamba_path.merge_to_dense(fisher_diagonals=fisher_mamba, seed=0)
    print("    Mamba merged")

    with torch.no_grad():
        merged_out = layer(x)
    print(f"    Dense Model output shape: {merged_out.shape}")

    # --- MLX Hardware Mapping ---
    try:
        import mlx.core as mx
        from daph_exfusion.mlx_inference import MLXDAPHDecoderLayer
        from daph_exfusion.bridge import load_mlx_model

        print("\n[4] Transferring structural weights to MLX layout...")
        mlx_layer = MLXDAPHDecoderLayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
        )
        mlx_layer = load_mlx_model(layer, mlx_layer, quantize=False, strict=True)
        print("    MLX mapping completed successfully")

        print("\n[5] Calculating maximum numerical distance check...")
        x_mlx = mx.array(x.detach().cpu().numpy())
        mlx_out = mlx_layer(x_mlx)
        print(f"    MLX execution complete. Output tensor shape: {mlx_out.shape}")

        pt_np = merged_out.detach().cpu().numpy()
        mlx_np = np.array(mlx_out)
        diff = float(np.abs(pt_np - mlx_np).max())
        print(f"    Max Absolute Discrepancy (PT vs MLX): {diff:.3e}")
        if diff < 1e-3:
            print("    Parity validation verified")
        else:
            print("    Parity check warnings: investigate structural alignments")

    except ImportError as e:
        print(f"\n[4-5] Bypassing Apple Silicon translation steps: {e}")

    print("\n" + "=" * 60)
    print("Demo execution finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
