"""
DAPH ExFusion -> MLX Native Inference Pipeline
==============================================
Provides dense, hardware-accelerated MLX component layers on Apple Silicon.
Ensures trace compatibility with graph compilers and proper state structure registry.
"""

import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# =============================================================================
# 1. CUSTOM METAL KERNELS (Precise Single-Precision Calculation Boundaries)
# =============================================================================

# Fused SwiGLU Epilogue
_swiglu_epilogue_kernel = mx.fast.metal_kernel(
    name="fused_swiglu_epilogue",
    input_names=["gate_out", "up_out"],
    output_names=["fused"],
    source="""
        uint elem = thread_position_in_grid.x;
        float g = (float)gate_out[elem];
        float u = (float)up_out[elem];
        float sig = 1.0f / (1.0f + metal::exp(-g));
        fused[elem] = T(g * sig * u);
    """,
)

def fused_swiglu_epilogue(gate_out: mx.array, up_out: mx.array) -> mx.array:
    assert gate_out.shape == up_out.shape
    return _swiglu_epilogue_kernel(
        inputs=[gate_out, up_out],
        template=[("T", gate_out.dtype)],
        grid=(gate_out.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[gate_out.shape],
        output_dtypes=[gate_out.dtype],
    )[0]


# Mamba Selective Scan
_mamba_scan_kernel = mx.fast.metal_kernel(
    name="mamba_selective_scan",
    input_names=["delta", "A_log", "B", "C", "D", "x"],
    output_names=["y"],
    source="""
        uint elem = thread_position_in_grid.x;   
        uint bsz = x_shape[0];
        uint L   = x_shape[1];
        uint d   = x_shape[2];

        uint b_idx = elem / d;
        uint c_idx = elem % d;

        float a_f = -metal::exp((float)A_log[c_idx]);
        float state_f = 0.0f;

        for (uint t = 0; t < L; t++) {
            uint idx = (b_idx * L + t) * d + c_idx;
            float dt_f = (float)delta[idx];
            float Bt_f = (float)B[idx];
            float Ct_f = (float)C[idx];
            float xt_f = (float)x[idx];

            float decay = metal::exp(dt_f * a_f);
            state_f = decay * state_f + dt_f * Bt_f * xt_f;

            y[idx] = T(Ct_f * state_f + (float)D[c_idx] * xt_f);
        }
    """,
)

def mamba_selective_scan(delta: mx.array, A_log: mx.array, B: mx.array, C: mx.array, D: mx.array, x: mx.array) -> mx.array:
    dtype = x.dtype
    delta = delta.astype(dtype)
    A_log = A_log.astype(dtype)
    B = B.astype(dtype)
    C = C.astype(dtype)
    D = D.astype(dtype)
    
    return _mamba_scan_kernel(
        inputs=[delta, A_log, B, C, D, x],
        template=[("T", dtype)],
        grid=(x.shape[0] * x.shape[2], 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[dtype],
    )[0]


def mamba_selective_scan_reference(delta: mx.array, A_log: mx.array, B: mx.array, C: mx.array, D: mx.array, x: mx.array) -> mx.array:
    """Pure-MLX Python loop reference for correctness checks."""
    bsz, L, d = x.shape
    a = -mx.exp(A_log)
    state = mx.zeros((bsz, d))
    ys = []
    for t in range(L):
        dt = delta[:, t, :]
        Bt = B[:, t, :]
        Ct = C[:, t, :]
        xt = x[:, t, :]

        decay = mx.exp(dt * a)
        state = decay * state + dt * Bt * xt
        yt = Ct * state + D * xt
        ys.append(yt)
    return mx.stack(ys, axis=1)


@mx.compile
def _ssm_prefill_step(state: mx.array, dt: mx.array, Bt: mx.array,
                     xt: mx.array, a: mx.array) -> mx.array:
    """Compiled single-timestep SSM state update.

    Fuses the ``decay * state + dt * Bt * xt`` elementwise chain into a
    single compiled graph node instead of ~5 eager ops.  Called once per
    timestep inside the pre-fill loop of ``MLXStatefulDAPHDecoderLayer``;
    the Python loop remains (the sequential scan has a data dependency
    across timesteps) but each iteration is now one compiled kernel call
    rather than a chain of eager allocations.
    """
    decay = mx.exp(dt * a)
    return decay * state + dt * Bt * xt


def ssm_prefill_loop(delta: mx.array, Bc: mx.array, u: mx.array,
                     a: mx.array, state: mx.array) -> mx.array:
    """Run the compiled per-step kernel over the full sequence.

    Equivalent to ``mamba_selective_scan_reference``'s state recurrence but
    without computing per-step outputs (the pre-fill path only needs the
    final state).  Returns the final ``state`` of shape ``(B, D)``.
    """
    L = int(u.shape[1])
    for t in range(L):
        state = _ssm_prefill_step(state, delta[:, t, :], Bc[:, t, :], u[:, t, :], a)
    return state


# =============================================================================
# 2. STATELESS MLX LAYER STACK
# =============================================================================

class MLXSwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.up = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        g = self.gate(x)
        u = self.up(x)
        fused = fused_swiglu_epilogue(g, u)
        return self.down(fused)


class MLXMergedMamba(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.x_proj_B = nn.Linear(d_model, d_model, bias=False)
        self.x_proj_C = nn.Linear(d_model, d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=False)
        self.A_log = mx.zeros((d_model,))
        self.D = mx.zeros((d_model,))

    def __call__(self, x: mx.array) -> mx.array:
        h = self.in_proj(x)
        a_gate, b_gate = h[..., :self.d_model], h[..., self.d_model:]
        u = nn.silu(a_gate) * b_gate  

        delta = nn.softplus(self.dt_proj(u))
        Bc = self.x_proj_B(u)
        Cc = self.x_proj_C(u)

        y = mamba_selective_scan(delta, self.A_log, Bc, Cc, self.D, u)
        return self.out_proj(y)


class MLXFlashAttention(nn.Module):
    """Separate Q/K/V/O projections to match PyTorch DAPHDecoderLayer exactly."""
    def __init__(self, hidden_size: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.o_proj(out)


class MLXAttentionPath(nn.Module):
    """Proper MLX module wrapper so parameters register cleanly in structural layers."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = MLXFlashAttention(hidden_size, num_heads)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        return self.attn(self.norm(x), mask=mask)


class MLXFNetBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.norm(x)
        x_fft = mx.real(mx.fft.fft2(x, axes=(-2, -1)))
        return residual + self.ff(x_fft)


class MLXMacroRouter(nn.Module):
    def __init__(self, hidden_size: int, num_paths: int = 3):
        super().__init__()
        self.difficulty_pred = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_paths)
        )

    def __call__(self, hidden: mx.array, theta_t: float = 1.0) -> mx.array:
        pooled = hidden.mean(axis=1)
        difficulty = self.difficulty_pred(pooled)
        logits = self.router(pooled)
        
        zeros = mx.zeros_like(difficulty)
        modifier = mx.concatenate([zeros, difficulty * 2.0, zeros], axis=-1)
        
        logits = logits + modifier
        return mx.softmax(logits / max(theta_t, 1e-8), axis=-1)


class MLXDAPHDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.macro_router = MLXMacroRouter(hidden_size)
        self.attention_path = MLXAttentionPath(hidden_size, num_heads)
        self.ffn_path = MLXSwiGLUFFN(hidden_size, intermediate_size)
        self.mamba_path = MLXMergedMamba(hidden_size)
        self.cheap_path = MLXFNetBlock(hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)

    def __call__(self, hidden: mx.array, theta_t: float = 1.0) -> mx.array:
        residual = hidden
        macro_probs = self.macro_router(hidden, theta_t=theta_t)
        path_idx = mx.argmax(macro_probs, axis=-1)
        
        counts = mx.array([mx.sum(path_idx == i) for i in range(3)])
        dominant = mx.argmax(counts) # Shape () - trace-safe, no sync boundary

        # Compute paths unconditionally to support safe JIT graph caching
        out_attn = self.attention_path(hidden)
        out_eff = self.ffn_path(hidden) + self.mamba_path(hidden)
        out_cheap = self.cheap_path(hidden)

        # Dynamic trace-safe conditional switch
        routed_hidden = mx.where(
            dominant == 0, out_attn,
            mx.where(dominant == 1, out_eff, out_cheap)
        )

        return self.final_norm(residual + routed_hidden)


# =============================================================================
# 3. CLEAN WEIGHT LOADING MECHANICS
# =============================================================================

def clean_pytorch_keys(state_dict: dict) -> dict:
    """Corrects structural wrapper changes in nested classes for dense mapping."""
    cleaned = {}
    for key, value in state_dict.items():
        if ".ffn_path.merged_ffn." in key:
            key = key.replace(".ffn_path.merged_ffn.", ".ffn_path.")
        elif ".mamba_path.merged_mamba." in key:
            key = key.replace(".mamba_path.merged_mamba.", ".mamba_path.")
        elif ".attn_path.q_proj." in key:
            key = key.replace(".attn_path.q_proj.", ".attention_path.attn.q_proj.")
        elif ".attn_path.k_proj." in key:
            key = key.replace(".attn_path.k_proj.", ".attention_path.attn.k_proj.")
        elif ".attn_path.v_proj." in key:
            key = key.replace(".attn_path.v_proj.", ".attention_path.attn.v_proj.")
        elif ".attn_path.o_proj." in key:
            key = key.replace(".attn_path.o_proj.", ".attention_path.attn.o_proj.")
        elif ".attn_path.norm." in key:
            key = key.replace(".attn_path.norm.", ".attention_path.norm.")
            
        # Target state parameters of experts to drop during dense map initialization
        if any(ignore in key for ignore in [
            "memory_bank", "step_count", ".experts.", 
            ".ffn_path.router.", ".ffn_path.router_out.",
            ".mamba_path.router.", ".mamba_path.router_out."
        ]):
            continue
            
        cleaned[key] = value
    return cleaned


# =============================================================================
# 4. MLX DIFFICULTY-ADAPTIVE TOP-P MACRO-ROUTER (Trace-Safe)
# =============================================================================

class MLXAdaptiveTopPMacroRouter(nn.Module):
    """
    Macro-router executing difficulty-scaled nucleus path selection.
    Compatible with mx.compile tracing boundaries.
    """
    def __init__(self, d_model: int, num_paths: int = 3,
                 base_threshold: float = 0.85,
                 difficulty_scale: float = 0.3):
        super().__init__()
        self.num_paths = num_paths
        self.base_threshold = base_threshold
        self.difficulty_scale = difficulty_scale

        self.router = nn.Linear(d_model, num_paths, bias=False)
        self.difficulty_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def __call__(self, hidden: mx.array) -> tuple:
        """
        Args:
            hidden: (B, L, D) intermediate state representation.
        Returns:
            selected_mask: (B, L, num_paths) Boolean activation bitmask.
            probs:         (B, L, num_paths) path softmax probability layout.
            difficulty:    (B, L, 1) predicted local difficulty.
        """
        B, L, D = hidden.shape
        flat = hidden.reshape(-1, D)                      # (B*L, D)

        logits = self.router(flat)                        # (B*L, num_paths)
        probs = mx.softmax(logits, axis=-1)               # (B*L, num_paths)

        difficulty = self.difficulty_predictor(flat)        # (B*L, 1)
        diff_sq = difficulty.squeeze(-1)                  # (B*L,)

        threshold = self.base_threshold - self.difficulty_scale * (diff_sq - 0.5)
        threshold = mx.clip(threshold, min_val=0.4, max_val=0.95)  # (B*L,)

        # Descending sort via negation
        sorted_indices = mx.argsort(-probs, axis=-1)      # (B*L, num_paths)

        rows = mx.arange(probs.shape[0])[:, None]         # (B*L, 1)
        sorted_probs = probs[rows, sorted_indices]        # (B*L, num_paths)
        cumsum = mx.cumsum(sorted_probs, axis=-1)         # (B*L, num_paths)

        threshold_unsqueezed = threshold[:, None]           # (B*L, 1)
        under_threshold = cumsum < threshold_unsqueezed   # (B*L, num_paths)

        # Ensure at least the top-1 path is active
        shifted_under = under_threshold[:, :-1]           # (B*L, num_paths-1)
        first_column = mx.ones((probs.shape[0], 1), dtype=mx.bool_)
        mask_sorted = mx.concatenate([first_column, shifted_under], axis=1)

        # Scatter back to original path order using flat indexing
        flat_indices = (sorted_indices + rows * self.num_paths).flatten()
        flat_mask_sorted = mask_sorted.flatten()

        flat_selected = mx.zeros(probs.size, dtype=mx.bool_)
        flat_selected = flat_selected.at[flat_indices].set(flat_mask_sorted)
        selected_mask = flat_selected.reshape(probs.shape)

        return (
            selected_mask.reshape(B, L, -1),
            probs.reshape(B, L, -1),
            difficulty.reshape(B, L, 1),
        )


# =============================================================================
# 5. STATEFUL DECODER COMPONENTS
# =============================================================================

class KVCache:
    """Simple key-value cache for autoregressive attention."""
    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, k: mx.array, v: mx.array) -> tuple:
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        return self.keys, self.values

    def clear(self):
        self.keys = None
        self.values = None


class SSMState:
    """Mamba SSM hidden state container."""
    def __init__(self, bsz: int, d_model: int, dtype=mx.float32):
        self.h = mx.zeros((bsz, d_model), dtype=dtype)

    def update(self, h_new: mx.array):
        self.h = h_new


class MLXStatefulDAPHDecoderLayer(nn.Module):
    """
    Stateful DAPH decoder layer with adaptive top-p routing,
    KV-cache for attention, and SSM state for Mamba.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.macro_router = MLXAdaptiveTopPMacroRouter(hidden_size, num_paths=3)
        self.attention_path = MLXAttentionPath(hidden_size, num_heads)
        self.ffn_path = MLXSwiGLUFFN(hidden_size, intermediate_size)
        self.mamba_path = MLXMergedMamba(hidden_size)
        self.cheap_path = MLXFNetBlock(hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)

    def __call__(self, hidden: mx.array,
                 kv_cache: KVCache = None,
                 ssm_state: SSMState = None,
                 mask: mx.array = None) -> mx.array:
        """
        Forward pass for the stateful DAPH decoder layer.

        Args:
            hidden: (B, L, D) input tensor.
            kv_cache: Optional KVCache to accumulate keys/values for attention.
            ssm_state: Optional SSMState to track recurrent Mamba hidden state.
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (B, L, D) after adaptive routing and merging.
        """
        residual = hidden
        B, L, D = hidden.shape

        # Compute macro routing masks and probabilities
        macro_mask, macro_probs, _ = self.macro_router(hidden)
        use_attn = macro_mask[:, :, 0:1]
        use_eff = macro_mask[:, :, 1:2]
        use_cheap = macro_mask[:, :, 2:3]

        # 1. Attention Path with KV-cache support
        if kv_cache is not None:
            # Normalize input and project Q, K, V using the underlying FlashAttention projections
            norm_hidden = self.attention_path.norm(hidden)
            q = self.attention_path.attn.q_proj(norm_hidden)
            k = self.attention_path.attn.k_proj(norm_hidden)
            v = self.attention_path.attn.v_proj(norm_hidden)

            num_heads = self.attention_path.attn.num_heads
            head_dim = D // num_heads

            # Reshape to (B, num_heads, L, head_dim) to match scaled_dot_product_attention API
            q = q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)

            # Update cache with new keys/values (concatenate along sequence dimension)
            keys, values = kv_cache.update(k, v)

            # Scaled dot product attention over full cached sequence
            scale = 1.0 / math.sqrt(head_dim)
            attn_out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale, mask=mask)

            # Reshape back to (B, L, D)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, D)
            attn_out = self.attention_path.attn.o_proj(attn_out)
        else:
            # Stateless attention path
            attn_out = self.attention_path(hidden, mask=mask)

        # 2. Mamba path with optional step-wise SSM recurrence
        if ssm_state is not None and L == 1:
            # Perform efficient step-wise update of the Mamba state
            # Project input to obtain gating vectors
            h = self.mamba_path.in_proj(hidden)  # (B, 1, 2D)
            a_gate, b_gate = h[..., :self.mamba_path.d_model], h[..., self.mamba_path.d_model:]
            u = nn.silu(a_gate) * b_gate  # (B, 1, D)

            # Delta, B and C projections
            delta = nn.softplus(self.mamba_path.dt_proj(u)).squeeze(1)  # (B, D)
            Bc = self.mamba_path.x_proj_B(u).squeeze(1)               # (B, D)
            Cc = self.mamba_path.x_proj_C(u).squeeze(1)               # (B, D)
            u_sq = u.squeeze(1)                                       # (B, D)

            # Compute decay factor and update recurrent state
            decay = mx.exp(delta * -mx.exp(self.mamba_path.A_log))      # (D,) broadcasted
            h_new = decay * ssm_state.h + (delta * Bc * u_sq)           # (B, D)
            ssm_state.update(h_new)

            # Output step: C * h_new + D * u
            y_step = Cc * h_new + self.mamba_path.D * u_sq
            mamba_out = self.mamba_path.out_proj(y_step[:, None, :])    # (B, 1, D)
        else:
            # Full sequence selective scan for stateless mode
            # Run the Mamba kernel over the entire sequence to produce outputs
            mamba_out = self.mamba_path(hidden)
            # During pre-fill (L > 1) we must also capture the final recurrent state
            # for subsequent autoregressive decoding. The Metal kernel returns
            # only the output sequence, so we reconstruct the last hidden state
            # via a trace-safe reference recurrence. This incurs an O(L) loop
            # but is required for correct state initialization.
            if ssm_state is not None:
                # Project hidden to gating vectors for all time steps
                h_proj = self.mamba_path.in_proj(hidden)  # (B, L, 2D)
                a_gate, b_gate = h_proj[..., :self.mamba_path.d_model], h_proj[..., self.mamba_path.d_model:]
                u = nn.silu(a_gate) * b_gate  # (B, L, D)

                # Compute delta, B and C projections for all steps
                delta = nn.softplus(self.mamba_path.dt_proj(u))  # (B, L, D)
                Bc = self.mamba_path.x_proj_B(u)  # (B, L, D)
                # We do not need C projection for state update

                # Initialize zero state for each batch
                a = -mx.exp(self.mamba_path.A_log)
                # state has shape (B, D)
                state = mx.zeros((B, self.mamba_path.d_model), dtype=delta.dtype)
                # Iterate through sequence dimension to accumulate final state.
                # Each step is a single compiled kernel call (_ssm_prefill_step)
                # instead of a chain of eager ops, keeping the per-step graph
                # node count to ~1 instead of ~5.
                state = ssm_prefill_loop(delta, Bc, u, a, state)
                # Update the provided SSMState with the final state
                ssm_state.update(state)

        # 3. Feed-forward network output
        ffn_out = self.ffn_path(hidden)
        eff_out = ffn_out + mamba_out

        # 4. Cheap path via FNet
        cheap_out = self.cheap_path(hidden)

        # 5. Blend outputs based on macro masks and probabilities
        routed_out = (
            attn_out * use_attn * macro_probs[:, :, 0:1] +
            eff_out * use_eff * macro_probs[:, :, 1:2] +
            cheap_out * use_cheap * macro_probs[:, :, 2:3]
        )

        return self.final_norm(residual + routed_out)
