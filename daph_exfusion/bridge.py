"""
PyTorch -> MLX Bridge Layer
==========================
Defines validation mechanisms and explicit conversion wrappers for graph mapping.
"""

import warnings
import torch
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .mlx_inference import clean_pytorch_keys

def _mlx_flat_parameters(mlx_model: nn.Module) -> dict:
    """Return a flat ``{dotted_key: mx.array}`` view of an MLX module's params.

    MLX 0.31 dropped ``Module.iter_flat_views``; ``mlx.utils.tree_flatten`` is
    the supported way to enumerate parameters as ``(key, array)`` pairs.
    """
    return dict(tree_flatten(mlx_model.parameters()))

def validate_architecture_compatibility(state_dict: dict, mlx_model: nn.Module, raise_on_mismatch: bool = True) -> bool:
    """Performs defensive shape validation checks before writing target state values.

    Catches two classes of defect that previously passed silently:
      * Missing keys — a PT key with no MLX counterpart, or an MLX parameter
        with no PT source.  A single missing projection weight produces
        nonsense inference but went unnoticed under the old ``continue``.
      * Shape mismatches — including the (out, in) vs (in, out) transpose case.
    """
    mlx_parameters = _mlx_flat_parameters(mlx_model)
    cleaned_state = clean_pytorch_keys(state_dict)

    # --- Missing-key detection (both directions) ---------------------------
    missing_in_mlx = sorted(set(cleaned_state) - set(mlx_parameters))
    missing_in_pt = sorted(set(mlx_parameters) - set(cleaned_state))
    if missing_in_mlx or missing_in_pt:
        diff = {
            "missing_in_mlx": missing_in_mlx,
            "missing_in_pt": missing_in_pt,
        }
        msg = f"Parameter mismatch: {diff}"
        if raise_on_mismatch:
            raise RuntimeError(msg)
        warnings.warn(msg)
        return False

    # --- Shape validation --------------------------------------------------
    for key, tensor in cleaned_state.items():
        if key not in mlx_parameters:
            continue

        pt_shape = tuple(tensor.shape)
        mlx_shape = tuple(mlx_parameters[key].shape)

        # Address transpose configuration mismatch patterns between PyTorch and MLX linear elements
        if "weight" in key and pt_shape != mlx_shape:
            # Linear weights inside MLX are organized (out_features, in_features) matching PT,
            # but transpose patterns can emerge depending on projection factories.
            if pt_shape == (mlx_shape[1], mlx_shape[0]):
                cleaned_state[key] = tensor.t()
                pt_shape = tuple(cleaned_state[key].shape)

        if pt_shape != mlx_shape:
            mismatch_msg = (
                f"Architecture mismatch on key '{key}': "
                f"PyTorch shape {pt_shape} != MLX shape {mlx_shape}."
            )
            if raise_on_mismatch:
                raise RuntimeError(mismatch_msg)
            return False

    return True


def load_mlx_model(pytorch_module: torch.nn.Module, mlx_model: nn.Module, quantize: bool = False, strict: bool = True) -> nn.Module:
    """Executes the translation pipeline, updating MLX model parameters directly."""
    pt_state = pytorch_module.state_dict()
    
    # Defensive structural assertions
    validate_architecture_compatibility(pt_state, mlx_model, raise_on_mismatch=strict)
    
    cleaned_state = clean_pytorch_keys(pt_state)
    mlx_weights = []
    
    for k, v in cleaned_state.items():
        arr = mx.array(v.detach().cpu().numpy())
        # Address possible matrix transpose cases during numpy allocation
        mlx_params = dict(mlx_model.parameters())
        mlx_param = mlx_params.get(k)
        if mlx_param is not None and arr.shape != mlx_param.shape:
            if arr.shape == (mlx_param.shape[1], mlx_param.shape[0]):
                arr = arr.T
        mlx_weights.append((k, arr))
        
    mlx_model.load_weights(mlx_weights, strict=False)
    
    if quantize:
        # MLX built-in parameter group-wise quantization
        nn.quantize(mlx_model, group_size=64, bits=4)
        
    return mlx_model