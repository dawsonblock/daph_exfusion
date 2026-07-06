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
from typing import Optional

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


# Mamba Selective Scan — fused output + final-state capture
# Cache of compiled Metal kernels keyed by d_state, so we only compile
# once per state dimension.
_mamba_kernel_cache: dict[int, object] = {}
_cooperative_kernel_cache: dict[int, object] = {}

# Threshold above which we switch to the cooperative (multi-thread) kernel.
# Below this, the single-thread-per-channel kernel is faster (no sync overhead).
_COOPERATIVE_D_STATE_THRESHOLD = 128

# SIMD group width for cooperative state reduction.
# Must match the Apple GPU SIMD group width (32 threads) so that
# simd_sum reduces within a single channel's thread group.
_SIMD_WIDTH = 32

def _get_mamba_scan_kernel(d_state: int):
    """Return a compiled single-thread Metal kernel for the given d_state.

    The kernel source is generated with a statically-sized register array
    matching the actual d_state.  Compiled kernels are cached per d_state.
    Supports d_state up to _COOPERATIVE_D_STATE_THRESHOLD; larger values
    should use _get_cooperative_mamba_scan_kernel instead.
    """
    if d_state > _COOPERATIVE_D_STATE_THRESHOLD:
        raise ValueError(
            f"Single-thread kernel supports d_state <= "
            f"{_COOPERATIVE_D_STATE_THRESHOLD}. Got d_state = {d_state}. "
            f"Use the cooperative kernel path instead."
        )
    if d_state in _mamba_kernel_cache:
        return _mamba_kernel_cache[d_state]

    source = f"""
        uint elem = thread_position_in_grid.x;
        uint bsz = x_shape[0];
        uint L   = x_shape[1];
        uint d   = x_shape[2];       // Model dimension (D)
        uint d_state = B_shape[2];   // State dimension

        // Guard: the GPU driver may round up the grid to a multiple of the
        // threadgroup size (256).  Padded threads with elem >= bsz * d must
        // exit immediately to prevent out-of-bounds memory reads/writes.
        if (elem >= bsz * d) return;

        uint b_idx = elem / d;
        uint c_idx = elem % d;       // channel index in [0, d)

        float a_f = -metal::exp((float)A_log[c_idx]);

        // Standard Mamba SSM: each channel c has a d_state-dimensional state.
        // h[t, n] = decay * h[t-1, n] + dt[t] * B[t, n] * x[t, c]
        // y[t, c] = sum_n C[t, n] * h[t, n] + D[c] * x[t, c]
        float h_states[{d_state}];
        for (uint n = 0; n < d_state; n++) h_states[n] = 0.0f;

        for (uint t = 0; t < L; t++) {{
            uint idx_x = (b_idx * L + t) * d + c_idx;
            float dt_f = (float)delta[idx_x];
            float xt_f = (float)x[idx_x];
            float decay = metal::exp(dt_f * a_f);

            float y_acc = (float)D[c_idx] * xt_f;
            for (uint n = 0; n < d_state; n++) {{
                uint idx_s = (b_idx * L + t) * d_state + n;
                float Bt_f = (float)B[idx_s];
                float Ct_f = (float)C[idx_s];
                h_states[n] = decay * h_states[n] + dt_f * Bt_f * xt_f;
                y_acc += Ct_f * h_states[n];
            }}
            y[idx_x] = T(y_acc);
        }}
        // Write final state: (B, D, d_state) layout
        for (uint n = 0; n < d_state; n++) {{
            h_last[(b_idx * d + c_idx) * d_state + n] = T(h_states[n]);
        }}
    """

    kernel = mx.fast.metal_kernel(
        name=f"mamba_selective_scan_d{d_state}",
        input_names=["delta", "A_log", "B", "C", "D", "x"],
        output_names=["y", "h_last"],
        source=source,
    )
    _mamba_kernel_cache[d_state] = kernel
    return kernel


def _get_cooperative_mamba_scan_kernel(d_state: int):
    """Return a cooperative (multi-thread) Metal kernel for large d_state.

    Uses a SIMD-group parallelization strategy: W threads collaborate on
    each channel, each handling S = ceil(d_state / W) state elements.
    Partial dot products are reduced via simd_sum, keeping per-thread
    register footprint small and constant regardless of d_state.

    Supports d_state up to 8192 (W=32, S<=256 floats per thread).
    """
    W = _SIMD_WIDTH
    S = (d_state + W - 1) // W  # ceil division

    if S > 256:
        raise ValueError(
            f"Cooperative kernel supports d_state <= {W * 256}. "
            f"Got d_state = {d_state} (S = {S} > 256)."
        )

    if d_state in _cooperative_kernel_cache:
        return _cooperative_kernel_cache[d_state]

    source = f"""
        uint elem = thread_position_in_grid.x;
        uint bsz = x_shape[0];
        uint L   = x_shape[1];
        uint d   = x_shape[2];       // Model dimension (D)
        uint d_state = B_shape[2];   // State dimension

        // Each channel is handled by W cooperative threads.
        // elem encodes both the channel and the thread-within-channel.
        uint total_channels = bsz * d;
        uint channel_idx = elem / {W};
        uint thread_idx_in_channel = elem % {W};

        // Determine if this thread is active (not padding).
        // Do NOT return early — inactive threads must still participate
        // in simd_sum with a zero contribution, otherwise the reduction
        // produces undefined results for the last channel in a threadgroup.
        bool active = (channel_idx < total_channels);

        uint b_idx = active ? (channel_idx / d) : 0;
        uint c_idx = active ? (channel_idx % d) : 0;

        float a_f = active ? -metal::exp((float)A_log[c_idx]) : 0.0f;

        // Each thread holds S state elements — small constant register footprint.
        float h_local[{S}];
        for (uint i = 0; i < {S}; i++) h_local[i] = 0.0f;

        for (uint t = 0; t < L; t++) {{
            // Compute local partial dot product (0 for inactive threads).
            float partial_dot = 0.0f;
            float dt_f = 0.0f;
            float xt_f = 0.0f;
            uint idx_x = 0;
            if (active) {{
                idx_x = (b_idx * L + t) * d + c_idx;
                dt_f = (float)delta[idx_x];
                xt_f = (float)x[idx_x];
                float decay = metal::exp(dt_f * a_f);

                for (uint i = 0; i < {S}; i++) {{
                    uint n = thread_idx_in_channel * {S} + i;
                    if (n >= d_state) break;  // Handle non-multiple d_state
                    uint idx_s = (b_idx * L + t) * d_state + n;
                    float Bt_f = (float)B[idx_s];
                    float Ct_f = (float)C[idx_s];
                    h_local[i] = decay * h_local[i] + dt_f * Bt_f * xt_f;
                    partial_dot += Ct_f * h_local[i];
                }}
            }}

            // SIMD-group reduction: ALL threads (including inactive with 0)
            // must call simd_sum at the same point — no divergence here.
            float final_dot = simd_sum(partial_dot);

            // Only active thread 0 writes the output (includes D * x term).
            if (active && thread_idx_in_channel == 0) {{
                y[idx_x] = T(final_dot + (float)D[c_idx] * xt_f);
            }}
        }}

        // Coalesced write-back of final state segments (active threads only).
        if (active) {{
            for (uint i = 0; i < {S}; i++) {{
                uint n = thread_idx_in_channel * {S} + i;
                if (n >= d_state) break;
                h_last[(b_idx * d + c_idx) * d_state + n] = T(h_local[i]);
            }}
        }}
    """

    kernel = mx.fast.metal_kernel(
        name=f"mamba_cooperative_scan_d{d_state}",
        input_names=["delta", "A_log", "B", "C", "D", "x"],
        output_names=["y", "h_last"],
        source=source,
    )
    _cooperative_kernel_cache[d_state] = kernel
    return kernel

def mamba_selective_scan(delta: mx.array, A_log: mx.array, B: mx.array,
                         C: mx.array, D: mx.array, x: mx.array) -> tuple:
    """Fused Metal selective scan returning both outputs and final state.

    Args:
        delta: (B, L, D) — discretization step
        A_log: (D,) — log of diagonal A matrix
        B: (B, L, d_state) — input-dependent B matrix
        C: (B, L, d_state) — input-dependent C matrix
        D: (D,) — feedthrough
        x: (B, L, D) — input sequence

    Returns ``(y, h_last)`` where ``y`` has shape ``(B, L, D)`` and ``h_last``
    has shape ``(B, D, d_state)`` — the final recurrent state after processing
    the full sequence, captured in the same GPU pass with no Python loop.

    .. note::
        For ``d_state <= 128``, a single-thread-per-channel kernel is used
        (minimal sync overhead).  For ``d_state > 128``, a cooperative
        SIMD-group kernel parallelizes the state dimension across W=16
        threads, keeping per-thread register footprint at S = ceil(d_state/W)
        floats.  Kernels are cached per ``d_state``.
    """
    d_state = B.shape[2]
    dtype = x.dtype
    delta = delta.astype(dtype)
    A_log = A_log.astype(dtype)
    B = B.astype(dtype)
    C = C.astype(dtype)
    D = D.astype(dtype)

    y_shape = x.shape
    h_last_shape = (x.shape[0], x.shape[2], d_state)  # (B, D, d_state)

    if d_state <= _COOPERATIVE_D_STATE_THRESHOLD:
        # Single-thread-per-channel kernel — faster for small d_state.
        kernel = _get_mamba_scan_kernel(d_state)
        y, h_last = kernel(
            inputs=[delta, A_log, B, C, D, x],
            template=[("T", dtype)],
            grid=(x.shape[0] * x.shape[2], 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[y_shape, h_last_shape],
            output_dtypes=[dtype, dtype],
        )
    else:
        # Cooperative kernel: W threads per channel, SIMD-group reduction.
        kernel = _get_cooperative_mamba_scan_kernel(d_state)
        W = _SIMD_WIDTH
        y, h_last = kernel(
            inputs=[delta, A_log, B, C, D, x],
            template=[("T", dtype)],
            grid=(x.shape[0] * x.shape[2] * W, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[y_shape, h_last_shape],
            output_dtypes=[dtype, dtype],
        )
    return y, h_last


def mamba_selective_scan_reference(delta: mx.array, A_log: mx.array, B: mx.array, C: mx.array, D: mx.array, x: mx.array) -> mx.array:
    """Pure-MLX Python loop reference for correctness checks.

    Args:
        delta: (B, L, D), A_log: (D,), B: (B, L, d_state), C: (B, L, d_state)
        D: (D,), x: (B, L, D)
    """
    bsz, L, d = x.shape
    d_state = B.shape[2]
    a = -mx.exp(A_log)  # (D,)
    # State: (B, D, d_state)
    state = mx.zeros((bsz, d, d_state))
    ys = []
    for t in range(L):
        dt = delta[:, t, :]  # (B, D)
        Bt = B[:, t, :]      # (B, d_state)
        Ct = C[:, t, :]      # (B, d_state)
        xt = x[:, t, :]      # (B, D)

        # decay: (B, D, 1) — broadcast over d_state
        # delta * a correctly broadcasts (B,D)*(D,)→(B,D), then expand for d_state
        decay = mx.exp((dt * a)[:, :, None])  # (B, D, 1)
        # state = decay * state + dt * Bt * xt
        # dt: (B, D), Bt: (B, d_state), xt: (B, D)
        # contribution: (B, D, d_state) = dt[:,:,None] * Bt[:,None,:] * xt[:,:,None]
        state = decay * state + dt[:, :, None] * Bt[:, None, :] * xt[:, :, None]
        # y = sum_n C[t,n] * h[t,n] + D * x[t]
        yt = (Ct[:, None, :] * state).sum(axis=-1) + D[None, :] * xt  # (B, D)
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

    Args:
        state: (B, D, d_state)
        dt: (B, D) — discretization step
        Bt: (B, d_state) — input-dependent B
        xt: (B, D) — input
        a: (D,) — diagonal A
    """
    decay = mx.exp((dt * a)[:, :, None])  # (B, D, 1)
    return decay * state + dt[:, :, None] * Bt[:, None, :] * xt[:, :, None]


def ssm_prefill_loop(delta: mx.array, Bc: mx.array, u: mx.array,
                     a: mx.array, state: mx.array) -> mx.array:
    """Run the compiled per-step kernel over the full sequence.

    Equivalent to ``mamba_selective_scan_reference``'s state recurrence but
    without computing per-step outputs (the pre-fill path only needs the
    final state).  Returns the final ``state`` of shape ``(B, D, d_state)``.

    Args:
        delta: (B, L, D), Bc: (B, L, d_state), u: (B, L, D)
        a: (D,), state: (B, D, d_state)
    """
    L = int(u.shape[1])
    for t in range(L):
        state = _ssm_prefill_step(state, delta[:, t, :], Bc[:, t, :], u[:, t, :], a)
    return state


def test_scan_correctness(B: int = 2, L: int = 8, D: int = 16, d_state: int = 16, atol: float = 1e-5) -> bool:
    """Verify the fused Metal kernel matches the pure-Python reference loop.

    Checks both the output sequence ``y`` and the final state ``h_last``.
    """
    delta = mx.random.normal((B, L, D))
    A_log = mx.random.normal((D,))
    Bv = mx.random.normal((B, L, d_state))
    C = mx.random.normal((B, L, d_state))
    Dv = mx.random.normal((D,))
    x = mx.random.normal((B, L, D))

    y_ref = mamba_selective_scan_reference(delta, A_log, Bv, C, Dv, x)
    y_ker, h_last = mamba_selective_scan(delta, A_log, Bv, C, Dv, x)

    y_match = np.allclose(np.array(y_ref), np.array(y_ker), atol=atol)

    # Verify h_last by replaying the reference recurrence to get the final state.
    a = -mx.exp(A_log)
    state_ref = mx.zeros((B, D, d_state))
    for t in range(L):
        decay = mx.exp((delta[:, t, :] * a)[:, :, None])  # (B, D, 1)
        state_ref = decay * state_ref + delta[:, t, :, None] * Bv[:, t, None, :] * x[:, t, :, None]
    h_match = np.allclose(np.array(state_ref), np.array(h_last), atol=atol)

    return bool(y_match and h_match)


def test_scan_with_real_weights(d_model: int = 64, L: int = 16, d_state: int = 16, atol: float = 1e-4) -> bool:
    """End-to-end scan correctness with a real MLXMergedMamba module."""
    mamba = MLXMergedMamba(d_model, d_state=d_state)
    x = mx.random.normal((1, L, d_model))
    y = mamba(x)
    # Also run via the reference path for comparison
    h = mamba.in_proj(x)
    a_gate, b_gate = h[..., :d_model], h[..., d_model:]
    u = nn.silu(a_gate) * b_gate
    delta = nn.softplus(mamba.dt_proj(u))
    Bc = mamba.x_proj_B(u)
    Cc = mamba.x_proj_C(u)
    y_ref = mamba_selective_scan_reference(delta, mamba.A_log, Bc, Cc, mamba.D, u)
    y_ref_out = mamba.out_proj(y_ref)
    return bool(np.allclose(np.array(y), np.array(y_ref_out), atol=atol))


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
    """Mamba SSM block with standard 1D causal convolution.

    Matches the architecture of Mamba-1/Mamba-2/Jamba: a depthwise conv1d
    (kernel width 4) is applied after the input projection gating and before
    the selective scan.  This is required for loading real-world pre-trained
    Mamba weights.

    Args:
        d_model: Model hidden dimension (D).
        d_conv: Convolution kernel size (default 4).
        d_state: SSM state dimension (default 16, matching Mamba-1).
            B and C projections output d_state, not d_model.  The SSM state
            has shape (B, D, d_state).
    """
    def __init__(self, d_model: int, d_conv: int = 4, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Standard depthwise 1D convolution (kernel=4, causal left-padding)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=d_conv,
                                stride=1, padding=0, groups=d_model, bias=True)
        # Project to state dimension (e.g. 16), not d_model — this matches
        # the standard Mamba architecture where B/C operate in d_state space.
        self.x_proj_B = nn.Linear(d_model, d_state, bias=False)
        self.x_proj_C = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=False)
        self.A_log = mx.zeros((d_model,))
        self.D = mx.zeros((d_model,))

    def _causal_conv1d(self, u: mx.array) -> mx.array:
        """Apply causal depthwise conv1d with left-padding.

        MLX Conv1d uses (N, L, C) format.  We left-pad with ``d_conv - 1``
        zeros and use ``padding=0`` so each output position depends only on
        current and past inputs (causal).
        """
        B, L, D = u.shape
        pad = mx.zeros((B, self.d_conv - 1, D), dtype=u.dtype)
        u_padded = mx.concatenate([pad, u], axis=1)  # (B, L + k - 1, D)
        return self.conv1d(u_padded)  # (B, L, D)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.in_proj(x)
        a_gate, b_gate = h[..., :self.d_model], h[..., self.d_model:]
        u = nn.silu(a_gate) * b_gate

        # Standard Mamba: depthwise causal conv1d + SiLU before projections
        u = nn.silu(self._causal_conv1d(u))

        delta = nn.softplus(self.dt_proj(u))
        Bc = self.x_proj_B(u)
        Cc = self.x_proj_C(u)

        y, _ = mamba_selective_scan(delta, self.A_log, Bc, Cc, self.D, u)
        return self.out_proj(y)


class MLXRotaryEmbedding:
    """Rotary Position Embeddings (RoPE) for MLX attention.

    Applies rotary embeddings to query and key tensors before the scaled
    dot-product attention, encoding relative position information.  This is
    required for compatibility with standard autoregressive Transformer
    architectures (Llama, Mistral, Qwen, etc.).

    The cos/sin caches are built lazily and extended as needed.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int = 4096,
                 base: float = 10000.0):
        self.head_dim = head_dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        """Build cos/sin caches up to ``seq_len`` positions."""
        inv_freq = 1.0 / (self.base ** (
            mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim
        ))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)  # (seq_len, head_dim // 2)
        # Duplicate to match head_dim: [f0, f1, ..., f0, f1, ...]
        emb = mx.concatenate([freqs, freqs], axis=-1)  # (seq_len, head_dim)
        self._cos = mx.cos(emb)
        self._sin = mx.sin(emb)
        self._cached_len = seq_len

    def _ensure_cache(self, offset: int, L: int):
        needed = offset + L
        if needed > self._cached_len:
            self._build_cache(max(needed, self._cached_len * 2))

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        """Rotate the second half of the last dimension."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return mx.concatenate([-x2, x1], axis=-1)

    def apply_rope(self, q: mx.array, k: mx.array,
                   offset: int = 0) -> tuple:
        """Apply RoPE to q and k.

        Args:
            q, k: (B, num_heads, L, head_dim) tensors.
            offset: Position offset (for use with KV-cache during decoding).

        Returns:
            (q_rotated, k_rotated) with the same shapes.
        """
        L = q.shape[2]
        self._ensure_cache(offset, L)
        cos = self._cos[offset:offset + L, :]  # (L, head_dim)
        sin = self._sin[offset:offset + L, :]
        # Broadcast cos/sin to (1, 1, L, head_dim) for elementwise mul.
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q_embed = q * cos + self._rotate_half(q) * sin
        k_embed = k * cos + self._rotate_half(k) * sin
        return q_embed, k_embed


class MLXFlashAttention(nn.Module):
    """Separate Q/K/V/O projections with optional Rotary Position Embeddings.

    Supports Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)
    via the ``num_kv_heads`` parameter.  When ``num_kv_heads < num_heads``,
    K and V projections produce fewer heads and MLX's
    ``scaled_dot_product_attention`` natively broadcasts them across query
    groups, matching the architecture of modern causal Transformers
    (Llama-3, Mistral, Qwen).  Set ``use_rope=False`` to disable RoPE.
    """
    def __init__(self, hidden_size: int, num_heads: int,
                 num_kv_heads: Optional[int] = None, bias: bool = False,
                 use_rope: bool = True, max_position_embeddings: int = 4096):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        # K/V projections produce num_kv_heads * head_dim, not hidden_size
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.use_rope = use_rope
        if use_rope:
            self.rope = MLXRotaryEmbedding(self.head_dim, max_position_embeddings)
        else:
            self.rope = None

    def __call__(self, x: mx.array, mask: mx.array = None,
                 offset: int = 0) -> mx.array:
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply Rotary Position Embeddings to Q and K
        if self.rope is not None:
            q, k = self.rope.apply_rope(q, k, offset=offset)

        scale = 1.0 / math.sqrt(self.head_dim)
        # MLX SDPA natively broadcasts K/V for GQA/MQA (num_kv_heads < num_heads)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.o_proj(out)


class MLXAttentionPath(nn.Module):
    """Proper MLX module wrapper so parameters register cleanly in structural layers."""
    def __init__(self, hidden_size: int, num_heads: int,
                 num_kv_heads: Optional[int] = None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = MLXFlashAttention(hidden_size, num_heads,
                                      num_kv_heads=num_kv_heads)

    def __call__(self, x: mx.array, mask: mx.array = None,
                 offset: int = 0) -> mx.array:
        return self.attn(self.norm(x), mask=mask, offset=offset)


class MLXFNetBlock(nn.Module):
    """FNet-inspired cheap path: LayerNorm → FFT2 → Linear.

    No internal residual — the outer decoder layer handles residual
    blending, matching the PyTorch ``DAPHDecoderLayerV2._cheap_path``
    structure for bridge parity.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x_fft = mx.real(mx.fft.fft2(x, axes=(-2, -1)))
        return self.ff(x_fft)


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
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int,
                 num_kv_heads: Optional[int] = None,
                 skip_inactive_paths: bool = False):
        super().__init__()
        self.skip_inactive_paths = skip_inactive_paths
        self.macro_router = MLXMacroRouter(hidden_size)
        self.attention_path = MLXAttentionPath(hidden_size, num_heads,
                                               num_kv_heads=num_kv_heads)
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

        if self.skip_inactive_paths:
            # Sync to read dominant path, then compute only that path.
            # Trades a small device-to-host sync for FLOP savings when
            # one path dominates.  Disabled by default to preserve JIT
            # graph caching.
            mx.eval(dominant)
            d = int(dominant)
            if d == 0:
                routed_hidden = self.attention_path(hidden)
            elif d == 1:
                routed_hidden = self.ffn_path(hidden) + self.mamba_path(hidden)
            else:
                routed_hidden = self.cheap_path(hidden)
        else:
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
    """Corrects structural wrapper changes in nested classes for dense mapping.

    Handles both root-level keys (e.g. ``ffn_path.merged_ffn.up.weight``) and
    nested keys (e.g. ``layer.0.ffn_path.merged_ffn.up.weight``) by checking
    for the pattern with and without a leading dot.
    """
    cleaned = {}
    # (old_pattern, new_pattern) — applied as substring replacement
    replacements = [
        ("ffn_path.merged_ffn.", "ffn_path."),
        ("mamba_path.merged_mamba.", "mamba_path."),
        ("attn_path.q_proj.", "attention_path.attn.q_proj."),
        ("attn_path.k_proj.", "attention_path.attn.k_proj."),
        ("attn_path.v_proj.", "attention_path.attn.v_proj."),
        ("attn_path.o_proj.", "attention_path.attn.o_proj."),
        ("attn_path.norm.", "attention_path.norm."),
    ]
    for key, value in state_dict.items():
        for old, new in replacements:
            # Root-level: key starts with the pattern
            if key.startswith(old):
                key = new + key[len(old):]
                break
            # Nested: pattern appears after a dot
            dot_old = "." + old
            if dot_old in key:
                key = key.replace(dot_old, "." + new, 1)
                break

        # Drop runtime state and expert parameters that don't exist in the
        # dense (merged) model.  Patterns work for both root-level and nested
        # keys since we use ``in`` substring matching.
        if any(ignore in key for ignore in [
            "memory_bank", "step_count", "experts",
            "ffn_path.router", "ffn_path.router_out",
            "mamba_path.router", "mamba_path.router_out"
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

    v4.5.0 enhancements mirror the PyTorch router:
      - Multi-signal difficulty (entropy + norm fusion).
      - Cost-aware routing via per-path logit bias.
      - Learnable threshold parameters.
      - Prefill/decode mode specialisation via ``decode_mode`` flag.
    """

    def __init__(self, d_model: int, num_paths: int = 3,
                 base_threshold: float = 0.85,
                 difficulty_scale: float = 0.3,
                 path_costs: Optional[tuple] = None,
                 cost_penalty: float = 0.1,
                 multi_signal_difficulty: bool = False,
                 learnable_threshold: bool = False):
        super().__init__()
        self.num_paths = num_paths
        self.multi_signal_difficulty = multi_signal_difficulty

        # Learnable or fixed threshold parameters
        if learnable_threshold:
            self.base_threshold = mx.array(base_threshold)
            self.difficulty_scale = mx.array(difficulty_scale)
        else:
            self.base_threshold = base_threshold
            self.difficulty_scale = difficulty_scale

        self.router = nn.Linear(d_model, num_paths, bias=False)
        self.difficulty_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        if multi_signal_difficulty:
            self.difficulty_combiner = nn.Sequential(
                nn.Linear(3, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )

        # Cost-aware routing: penalise expensive paths in logit space
        if path_costs is not None:
            assert len(path_costs) == num_paths
            cost_log = mx.array([math.log(c) for c in path_costs])
            self.cost_log_bias = -cost_penalty * cost_log
        else:
            self.cost_log_bias = None

    def __call__(self, hidden: mx.array,
                 decode_mode: bool = False) -> tuple:
        """
        Args:
            hidden: (B, L, D) intermediate state representation.
            decode_mode: If True, use a more aggressive (lower) threshold
                during single-token decoding to save compute, since
                cached state already captures context.
        Returns:
            selected_mask: (B, L, num_paths) Boolean activation bitmask.
            probs:         (B, L, num_paths) path softmax probability layout.
            difficulty:    (B, L, 1) predicted local difficulty.
        """
        B, L, D = hidden.shape
        flat = hidden.reshape(-1, D)                      # (B*L, D)

        logits = self.router(flat)                        # (B*L, num_paths)

        # Cost-aware routing: subtract log(cost) penalty from logits
        if self.cost_log_bias is not None:
            logits = logits + self.cost_log_bias

        probs = mx.softmax(logits, axis=-1)               # (B*L, num_paths)

        difficulty = self.difficulty_predictor(flat)        # (B*L, 1)
        diff_sq = difficulty.squeeze(-1)                  # (B*L,)

        if self.multi_signal_difficulty:
            # Router logit entropy: high entropy = uncertain/hard
            log_probs = mx.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum(axis=-1)   # (B*L,)
            max_entropy = math.log(self.num_paths)
            entropy_norm = entropy / max_entropy           # [0, 1]
            # Token norm (normalised by sqrt(D))
            token_norm = mx.sqrt((flat * flat).sum(axis=-1)) / math.sqrt(D)
            combined = mx.stack([diff_sq, entropy_norm, token_norm], axis=-1)
            diff_sq = self.difficulty_combiner(combined).squeeze(-1)

        # Threshold: higher difficulty → higher threshold → more paths active.
        threshold = self.base_threshold + self.difficulty_scale * (diff_sq - 0.5)

        # Decode mode: lower threshold to favour cheaper paths during
        # single-token generation (cached state already has context).
        if decode_mode:
            threshold = threshold - 0.1

        threshold = mx.clip(threshold, 0.4, 0.95)  # (B*L,)

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

        # Scatter back to original path order.  MLX 0.31's ArrayAt doesn't
        # support .set(), so we use the inverse argsort to unscramble the
        # sorted mask back to the original path order.
        inverse_indices = mx.argsort(sorted_indices, axis=-1)  # (B*L, num_paths)
        mask_unsorted = mx.take_along_axis(mask_sorted, inverse_indices, axis=-1)
        selected_mask = mask_unsorted.reshape(B, L, -1)

        return (
            selected_mask,
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
    """Mamba SSM hidden state container.

    The state has shape (B, D, d_state) where D is the model dimension and
    d_state is the SSM state dimension (typically 16).
    """
    def __init__(self, bsz: int, d_model: int, d_state: int = 16, dtype=mx.float32):
        self.h = mx.zeros((bsz, d_model, d_state), dtype=dtype)

    def update(self, h_new: mx.array):
        self.h = h_new


class ConvState:
    """Rolling history buffer for causal depthwise conv1d during decoding.

    Stores the last ``kernel_size - 1`` projected inputs so that single-token
    decoding steps can apply the conv1d without reprocessing the full sequence.
    """
    def __init__(self, bsz: int, d_model: int, kernel_size: int = 4, dtype=mx.float32):
        self.kernel_size = kernel_size
        self.history = mx.zeros((bsz, kernel_size - 1, d_model), dtype=dtype)

    def update(self, x_new: mx.array) -> mx.array:
        """Append ``x_new`` (B, 1, D) and return the full conv window (B, k, D).

        The returned window has ``kernel_size`` elements in the sequence
        dimension: the ``kernel_size - 1`` stored history items plus the
        new input.  The history is then trimmed to the last
        ``kernel_size - 1`` items for the next call.
        """
        window = mx.concatenate([self.history, x_new], axis=1)  # (B, k, D)
        self.history = window[:, 1:, :]  # keep last k-1 for next step
        return window


class MLXStatefulDAPHDecoderLayer(nn.Module):
    """
    Stateful DAPH decoder layer with adaptive top-p routing,
    KV-cache for attention, and SSM state for Mamba.

    Passes ``decode_mode=True`` to the router during single-token
    decoding (L==1) so the router uses a more aggressive threshold
    that favours cheaper paths, since cached state already captures
    context from the prefill phase.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int,
                 num_kv_heads: Optional[int] = None,
                 path_costs: Optional[tuple] = None,
                 cost_penalty: float = 0.1,
                 multi_signal_difficulty: bool = False,
                 learnable_threshold: bool = False):
        super().__init__()
        self.macro_router = MLXAdaptiveTopPMacroRouter(
            hidden_size, num_paths=3,
            path_costs=path_costs,
            cost_penalty=cost_penalty,
            multi_signal_difficulty=multi_signal_difficulty,
            learnable_threshold=learnable_threshold,
        )
        self.attention_path = MLXAttentionPath(hidden_size, num_heads,
                                               num_kv_heads=num_kv_heads)
        self.ffn_path = MLXSwiGLUFFN(hidden_size, intermediate_size)
        self.mamba_path = MLXMergedMamba(hidden_size)
        self.cheap_path = MLXFNetBlock(hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)

    def __call__(self, hidden: mx.array,
                 kv_cache: KVCache = None,
                 ssm_state: SSMState = None,
                 conv_state: ConvState = None,
                 mask: mx.array = None) -> mx.array:
        """
        Forward pass for the stateful DAPH decoder layer.

        Args:
            hidden: (B, L, D) input tensor.
            kv_cache: Optional KVCache to accumulate keys/values for attention.
            ssm_state: Optional SSMState to track recurrent Mamba hidden state.
            conv_state: Optional ConvState for Mamba conv1d during single-token decoding.
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (B, L, D) after adaptive routing and merging.
        """
        residual = hidden
        B, L, D = hidden.shape

        # Compute macro routing masks and probabilities.
        # During single-token decoding (L==1), pass decode_mode=True so
        # the router uses a more aggressive threshold favouring cheaper
        # paths, since cached state already captures context.
        decode_mode = (L == 1)
        macro_mask, macro_probs, _ = self.macro_router(hidden, decode_mode=decode_mode)
        use_attn = macro_mask[:, :, 0:1]
        use_eff = macro_mask[:, :, 1:2]
        use_cheap = macro_mask[:, :, 2:3]

        # Re-normalize probabilities over selected paths so tokens that
        # activate only a subset of paths retain full output scale.
        # Without this, an easy token activating only the cheap path gets
        # squashed by its raw softmax probability (e.g. 0.6), causing
        # exponential scale decay across decoder layers.
        active_probs = macro_probs * macro_mask.astype(macro_probs.dtype)
        prob_sum = active_probs.sum(axis=-1, keepdims=True)
        prob_sum = mx.maximum(prob_sum, 1e-8)
        norm_probs = active_probs / prob_sum  # sums to 1.0 over active paths

        # 1. Attention Path with KV-cache support
        if kv_cache is not None:
            # Normalize input and project Q, K, V using the underlying FlashAttention projections
            norm_hidden = self.attention_path.norm(hidden)
            q = self.attention_path.attn.q_proj(norm_hidden)
            k = self.attention_path.attn.k_proj(norm_hidden)
            v = self.attention_path.attn.v_proj(norm_hidden)

            num_heads = self.attention_path.attn.num_heads
            num_kv_heads = self.attention_path.attn.num_kv_heads
            head_dim = D // num_heads

            # Reshape to (B, num_heads, L, head_dim) for Q and
            # (B, num_kv_heads, L, head_dim) for K/V (GQA/MQA support)
            q = q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

            # Apply RoPE before caching, using the current cache length as
            # the position offset so cached keys keep their original positions.
            if self.attention_path.attn.rope is not None:
                offset = 0 if kv_cache.keys is None else kv_cache.keys.shape[2]
                q, k = self.attention_path.attn.rope.apply_rope(q, k, offset=offset)

            # Update cache with new keys/values (concatenate along sequence dimension)
            keys, values = kv_cache.update(k, v)

            # Scaled dot product attention over full cached sequence.
            # MLX SDPA natively broadcasts K/V for GQA (num_kv_heads < num_heads).
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
            # Single-token decoding: step-wise update of the Mamba state
            h = self.mamba_path.in_proj(hidden)  # (B, 1, 2D)
            a_gate, b_gate = h[..., :self.mamba_path.d_model], h[..., self.mamba_path.d_model:]
            u = nn.silu(a_gate) * b_gate  # (B, 1, D)

            # Apply causal conv1d using the rolling ConvState buffer.
            # conv_state stores the last (d_conv - 1) projected inputs;
            # we append the current one and apply the depthwise conv.
            if conv_state is not None:
                window = conv_state.update(u)  # (B, d_conv, D)
                u_conv = self.mamba_path.conv1d(window)  # (B, 1, D)
                u = nn.silu(u_conv)

            # Delta, B and C projections
            # B and C now project to d_state (e.g. 16), not d_model
            d_state = self.mamba_path.d_state
            delta = nn.softplus(self.mamba_path.dt_proj(u)).squeeze(1)  # (B, D)
            Bc = self.mamba_path.x_proj_B(u).squeeze(1)               # (B, d_state)
            Cc = self.mamba_path.x_proj_C(u).squeeze(1)               # (B, d_state)
            u_sq = u.squeeze(1)                                       # (B, D)

            # Compute decay factor and update recurrent state.
            # State has shape (B, D, d_state).
            # decay: (B, D, 1) — broadcast over d_state
            # delta * a correctly broadcasts (B,D)*(D,)→(B,D), then expand for d_state
            a = -mx.exp(self.mamba_path.A_log)  # (D,)
            decay = mx.exp((delta * a)[:, :, None])  # (B, D, 1)
            # contribution: dt[:,:,None] * Bt[:,None,:] * xt[:,:,None]
            h_new = decay * ssm_state.h + delta[:, :, None] * Bc[:, None, :] * u_sq[:, :, None]
            ssm_state.update(h_new)

            # Output step: y = sum_n C[n] * h[n] + D * x
            y_step = (Cc[:, None, :] * h_new).sum(axis=-1) + self.mamba_path.D * u_sq  # (B, D)
            mamba_out = self.mamba_path.out_proj(y_step[:, None, :])    # (B, 1, D)
        else:
            # Full sequence selective scan for stateless mode
            if ssm_state is not None:
                # Pre-fill path: compute outputs AND capture the final recurrent
                # state in a single GPU pass via the fused metal kernel, which
                # now returns (y, h_last).  This eliminates the sequential O(L)
                # Python pre-fill loop entirely.
                h_proj = self.mamba_path.in_proj(hidden)  # (B, L, 2D)
                a_gate, b_gate = h_proj[..., :self.mamba_path.d_model], h_proj[..., self.mamba_path.d_model:]
                u = nn.silu(a_gate) * b_gate  # (B, L, D)

                # Apply causal conv1d + SiLU (standard Mamba block)
                u = nn.silu(self.mamba_path._causal_conv1d(u))

                delta = nn.softplus(self.mamba_path.dt_proj(u))  # (B, L, D)
                Bc = self.mamba_path.x_proj_B(u)  # (B, L, d_state)
                Cc = self.mamba_path.x_proj_C(u)  # (B, L, d_state)

                y_seq, h_last = mamba_selective_scan(
                    delta, self.mamba_path.A_log, Bc, Cc, self.mamba_path.D, u
                )
                mamba_out = self.mamba_path.out_proj(y_seq)
                ssm_state.update(h_last)

                # Seed the ConvState with the last (d_conv - 1) projected
                # inputs so subsequent single-token decoding steps have the
                # correct conv window.  When the pre-fill sequence is shorter
                # than the conv history window (L < d_conv - 1), left-pad
                # with zeros to preserve the (B, d_conv - 1, D) shape;
                # otherwise the slice collapses the history dimension and
                # crashes the first decoding step.
                if conv_state is not None:
                    k = self.mamba_path.d_conv
                    history_len = k - 1
                    if u.shape[1] >= history_len:
                        conv_state.history = u[:, -history_len:, :].astype(
                            conv_state.history.dtype
                        )
                    else:
                        pad_len = history_len - u.shape[1]
                        pad = mx.zeros((B, pad_len, D), dtype=u.dtype)
                        conv_state.history = mx.concatenate(
                            [pad, u], axis=1
                        ).astype(conv_state.history.dtype)
            else:
                # Pure stateless mode — no state capture needed
                mamba_out = self.mamba_path(hidden)

        # 3. Feed-forward network output
        # Packed token dispatch (gather-run-scatter) saves FLOPs by only
        # running the FFN on tokens that use the efficient path.  This is
        # implemented in the PyTorch DAPHDecoderLayerV2.  In MLX, dynamic
        # tensor sizing from data (required for gather) would break lazy
        # evaluation, so we compute on all tokens and mask the output.
        # The FLOP savings are realized in the PyTorch inference path.
        ffn_out = self.ffn_path(hidden)
        eff_out = ffn_out + mamba_out

        # 4. Cheap path via FNet
        cheap_out = self.cheap_path(hidden)

        # 5. Blend outputs based on macro masks and re-normalized probabilities
        routed_out = (
            attn_out * use_attn * norm_probs[:, :, 0:1] +
            eff_out * use_eff * norm_probs[:, :, 1:2] +
            cheap_out * use_cheap * norm_probs[:, :, 2:3]
        )

        return self.final_norm(residual + routed_out)


# =============================================================================
# 6. TOP-LEVEL STATEFUL CAUSAL LM
# =============================================================================

class MLXStatefulCausalLM(nn.Module):
    """Top-level causal LM container for multi-layer autoregressive generation.

    Manages per-layer KV-caches, SSM states, and conv states, and handles
    the transition from pre-fill (L > 1) to single-token decoding (L = 1).

    Usage::

        model = MLXStatefulCausalLM(num_layers=4, vocab_size=32000, ...)
        caches = [KVCache() for _ in range(num_layers)]
        ssm_states = [SSMState(B, D) for _ in range(num_layers)]
        conv_states = [ConvState(B, D) for _ in range(num_layers)]

        # Pre-fill
        logits = model(tokens, caches=caches, ssm_states=ssm_states,
                       conv_states=conv_states)

        # Decode one token at a time
        for _ in range(max_new_tokens):
            logits = model(next_token[:, None], caches=caches,
                           ssm_states=ssm_states, conv_states=conv_states,
                           offset=current_len)
            next_token = sample(logits[:, -1, :])
            current_len += 1
    """

    def __init__(self, num_layers: int, vocab_size: int, hidden_size: int,
                 intermediate_size: int, num_heads: int,
                 num_kv_heads: Optional[int] = None, d_state: int = 16):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            MLXStatefulDAPHDecoderLayer(hidden_size, intermediate_size,
                                        num_heads, num_kv_heads=num_kv_heads)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, tokens: mx.array,
                 caches: list = None,
                 ssm_states: list = None,
                 conv_states: list = None,
                 offset: int = 0) -> mx.array:
        """Forward pass returning logits ``(B, L, vocab_size)``.

        Args:
            tokens: ``(B, L)`` integer token IDs.
            caches: Optional list of ``KVCache`` per layer.
            ssm_states: Optional list of ``SSMState`` per layer.
            conv_states: Optional list of ``ConvState`` per layer.
            offset: Position offset for RoPE (use current sequence length
                during single-token decoding with a KV-cache).
        """
        x = self.embed_tokens(tokens)  # (B, L, D)
        B, L, D = x.shape

        # Build causal attention mask for pre-fill (L > 1)
        mask = None
        if L > 1:
            mask = mx.triu(mx.full((L, L), -float("inf")), k=1)

        for i, layer in enumerate(self.layers):
            layer_cache = caches[i] if caches is not None else None
            layer_ssm = ssm_states[i] if ssm_states is not None else None
            layer_conv = conv_states[i] if conv_states is not None else None
            x = layer(x, kv_cache=layer_cache, ssm_state=layer_ssm,
                      conv_state=layer_conv, mask=mask)

        x = self.norm(x)
        return self.lm_head(x)

    def generate(self, prompt_tokens: list, max_new_tokens: int = 64,
                 temperature: float = 0.0, seed: int = 42) -> list:
        """Generate tokens autoregressively from a prompt.

        Ensures memory stability across generation cycles by triggering
        ``mx.eval()`` on step outputs, pruning the lazy evaluation graph to
        prevent VRAM accumulation during long generation runs.

        Args:
            prompt_tokens: List of integer token IDs to seed generation.
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature; 0.0 means greedy argmax.
            seed: Random seed for reproducible sampling.

        Returns:
            List of generated token IDs (length ``max_new_tokens``).
        """
        mx.random.seed(seed)

        B = 1
        D = self.hidden_size

        # Instantiate per-layer states
        caches = [KVCache() for _ in range(self.num_layers)]
        ssm_states = [SSMState(B, D, self.d_state) for _ in range(self.num_layers)]
        conv_states = [ConvState(B, D) for _ in range(self.num_layers)]

        # --- Phase 1: Pre-fill ------------------------------------------------
        tokens_arr = mx.array([prompt_tokens])  # (1, L_prompt)
        logits = self(tokens_arr, caches=caches, ssm_states=ssm_states,
                       conv_states=conv_states)

        last_logits = logits[:, -1, :]
        if temperature > 0:
            next_token = mx.random.categorical(
                last_logits / temperature, num_samples=1
            )
        else:
            next_token = mx.argmax(last_logits, axis=-1, keepdims=True)

        generated_tokens = [int(next_token.item())]
        current_len = len(prompt_tokens)

        # Force evaluation of the pre-fill graph to release intermediate tensors.
        # Consolidate all arrays into a single mx.eval() call so MLX compiles
        # and executes them in one GPU dispatch pass instead of 1 + 3*N_layers
        # sequential dispatches.
        eval_targets = [next_token]
        for c in caches:
            if c.keys is not None:
                eval_targets.extend([c.keys, c.values])
        eval_targets.extend([s.h for s in ssm_states])
        eval_targets.extend([c.history for c in conv_states])
        mx.eval(*eval_targets)

        # --- Phase 2: Autoregressive decoding --------------------------------
        for _ in range(max_new_tokens - 1):
            logits = self(next_token, caches=caches, ssm_states=ssm_states,
                          conv_states=conv_states, offset=current_len)

            last_logits = logits[:, -1, :]
            if temperature > 0:
                next_token = mx.random.categorical(
                    last_logits / temperature, num_samples=1
                )
            else:
                next_token = mx.argmax(last_logits, axis=-1, keepdims=True)

            generated_tokens.append(int(next_token.item()))
            current_len += 1

            # Prune the lazy evaluation graph to prevent VRAM accumulation.
            # Single consolidated mx.eval() call — one GPU dispatch per token.
            eval_targets = [next_token]
            for c in caches:
                if c.keys is not None:
                    eval_targets.extend([c.keys, c.values])
            eval_targets.extend([s.h for s in ssm_states])
            eval_targets.extend([c.history for c in conv_states])
            mx.eval(*eval_targets)

        return generated_tokens
