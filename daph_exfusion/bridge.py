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

    Returns the cleaned (and transposed) state dict on success (truthy), or
    ``False`` on failure under ``raise_on_mismatch=False``.  Callers can
    reuse the returned state dict for loading, avoiding a second
    clean/transpose pass.
    """
    mlx_parameters = _mlx_flat_parameters(mlx_model)
    cleaned_state = clean_pytorch_keys(state_dict)

    # --- Missing-key detection (both directions) ---------------------------
    missing_in_mlx = sorted(set(cleaned_state) - set(mlx_parameters))
    missing_in_pt = sorted(set(mlx_parameters) - set(cleaned_state))
    if missing_in_mlx or missing_in_pt:
        err_lines = ["Bridge Parity Violation:"]
        if missing_in_mlx:
            err_lines.append(
                f"  Parameters in PyTorch weights but missing in MLX: {missing_in_mlx}"
            )
        if missing_in_pt:
            err_lines.append(
                f"  Parameters in MLX model but missing in PyTorch weights: {missing_in_pt}"
            )
        msg = "\n".join(err_lines)
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

        # Address transpose configuration mismatch patterns between PyTorch and MLX
        if "weight" in key and pt_shape != mlx_shape:
            # Linear weights: (out, in) vs (in, out) — 2D transpose
            if len(pt_shape) == 2 and pt_shape == (mlx_shape[1], mlx_shape[0]):
                cleaned_state[key] = tensor.t()
                pt_shape = tuple(cleaned_state[key].shape)
            # Conv1d weights: PyTorch (out_ch, in_ch/groups, kernel) vs
            # MLX (out_ch, kernel, in_ch/groups) — swap axes 1 and 2
            elif (len(pt_shape) == 3 and pt_shape[0] == mlx_shape[0]
                  and pt_shape[1] == mlx_shape[2] and pt_shape[2] == mlx_shape[1]):
                cleaned_state[key] = tensor.permute(0, 2, 1).contiguous()
                pt_shape = tuple(cleaned_state[key].shape)

        if pt_shape != mlx_shape:
            mismatch_msg = (
                f"Architecture mismatch on key '{key}': "
                f"PyTorch shape {pt_shape} != MLX shape {mlx_shape}."
            )
            if raise_on_mismatch:
                raise RuntimeError(mismatch_msg)
            return False

    return cleaned_state


def load_mlx_model(pytorch_module: torch.nn.Module, mlx_model: nn.Module, quantize: bool = False, strict: bool = True) -> nn.Module:
    """Executes the translation pipeline, updating MLX model parameters directly.

    Under ``strict=True`` (default), any missing key or shape mismatch raises
    ``RuntimeError``.  Under ``strict=False``, validation failures emit
    warnings and the load is **aborted** — the MLX model retains its
    initialized weights rather than silently loading a partial state dict
    that would produce nonsense inference.

    Reuses the cleaned and transposed state dict from
    ``validate_architecture_compatibility`` to avoid a second clean/transpose
    pass that could diverge from the validated state.
    """
    pt_state = pytorch_module.state_dict()

    # Defensive structural assertions.  Under strict=False this returns False
    # (with a warning) instead of raising; we must not proceed to load weights
    # if validation failed, as that would silently produce a broken model.
    # validate_architecture_compatibility returns the cleaned (and transposed)
    # state dict on success, or False on failure.
    cleaned_state = validate_architecture_compatibility(pt_state, mlx_model, raise_on_mismatch=strict)
    if not cleaned_state and not strict:
        warnings.warn(
            "load_mlx_model: validation failed under strict=False; "
            "aborting weight transfer to avoid silent corruption."
        )
        return mlx_model

    mlx_weights = []

    for k, v in cleaned_state.items():
        arr = mx.array(v.detach().cpu().numpy())
        mlx_weights.append((k, arr))

    mlx_model.load_weights(mlx_weights, strict=False)

    if quantize:
        # MLX built-in parameter group-wise quantization
        nn.quantize(mlx_model, group_size=64, bits=4)

    return mlx_model