"""Key mapping and structure parity tests."""
import pytest
import torch
import numpy as np
from mlx.utils import tree_flatten
from daph_exfusion.mlx_inference import clean_pytorch_keys, MLXDAPHDecoderLayer
from daph_exfusion.bridge import validate_architecture_compatibility


def test_map_key_drops_runtime_state():
    dummy_dict = {
        "ffn_path.memory_bank": torch.zeros(1),
        "mamba_path.step_count": torch.zeros(1),
        "ffn_path.router.weight": torch.zeros(2, 2),
    }
    cleaned = clean_pytorch_keys(dummy_dict)
    assert len(cleaned) == 0


def test_map_key_ffn_merged():
    dummy_dict = {
        "ffn_path.merged_ffn.up.weight": torch.zeros(8, 8),
        "ffn_path.merged_ffn.up.bias": torch.zeros(8),
    }
    cleaned = clean_pytorch_keys(dummy_dict)
    assert "ffn_path.up.weight" in cleaned
    assert "ffn_path.up.bias" in cleaned


def test_map_key_attention_nesting():
    dummy_dict = {
        "attn_path.q_proj.weight": torch.zeros(8, 8),
        "attn_path.norm.weight": torch.zeros(8),
    }
    cleaned = clean_pytorch_keys(dummy_dict)
    assert "attention_path.attn.q_proj.weight" in cleaned
    assert "attention_path.norm.weight" in cleaned


def test_validate_raises_on_shape_mismatch():
    # Build a complete, shape-compatible state dict from the MLX model itself,
    # then corrupt one weight's shape so the *only* defect is a shape mismatch
    # (the new missing-key guard would otherwise fire first on a partial dict).
    mlx_model = MLXDAPHDecoderLayer(hidden_size=64, intermediate_size=128, num_heads=4)
    full_state = {}
    for k, v in tree_flatten(mlx_model.parameters()):
        full_state[k] = torch.from_numpy(np.array(v))
    # Corrupt one weight: swap its dimensions so it can't be transposed back.
    bad_key = "attention_path.attn.q_proj.weight"
    if bad_key not in full_state:
        # Fall back to any weight key present in the model.
        bad_key = next(k for k in full_state if "weight" in k)
    w = full_state[bad_key]
    full_state[bad_key] = torch.randn(w.shape[1], w.shape[0] * 2)  # neither equal nor transposable
    with pytest.raises(RuntimeError, match="Architecture mismatch"):
        validate_architecture_compatibility(full_state, mlx_model, raise_on_mismatch=True)


def test_validate_raises_on_missing_keys():
    """A partial state dict must fail loudly under raise_on_mismatch=True."""
    bad_state = {"attention_path.attn.q_proj.weight": torch.randn(64, 32)}
    mlx_model = MLXDAPHDecoderLayer(hidden_size=64, intermediate_size=128, num_heads=4)
    with pytest.raises(RuntimeError, match="Bridge Parity Violation"):
        validate_architecture_compatibility(bad_state, mlx_model, raise_on_mismatch=True)


def test_validate_warns_on_missing_keys_when_lenient():
    """Under raise_on_mismatch=False, missing keys warn and return False."""
    import warnings as _warnings
    bad_state = {"attention_path.attn.q_proj.weight": torch.randn(64, 32)}
    mlx_model = MLXDAPHDecoderLayer(hidden_size=64, intermediate_size=128, num_heads=4)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = validate_architecture_compatibility(bad_state, mlx_model, raise_on_mismatch=False)
    assert result is False
    assert any("Bridge Parity Violation" in str(w.message) for w in caught)
