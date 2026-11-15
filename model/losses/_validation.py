"""Controlli di validazione condivisi dalle loss degli outfit."""

import torch
from torch import Tensor


def floating_tensor(value: Tensor, name: str, ndim: int) -> Tensor:
    """Valida gli input e promuove i tensori a precisione ridotta in float32."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != ndim or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty rank-{ndim} tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    if value.dtype in (torch.float16, torch.bfloat16):
        return value.float()
    return value


def presence_mask(targets: Tensor, shape: torch.Size, device: torch.device) -> Tensor:
    """Verifica che ogni coppia outfit/categoria abbia un target di presenza binario."""
    if not isinstance(targets, Tensor):
        raise TypeError("category_targets must be a tensor")
    if targets.shape != shape:
        raise ValueError(f"category_targets must have shape {tuple(shape)}")
    if targets.device != device:
        raise ValueError("category_targets and predictions must be on the same device")
    if targets.is_complex() or bool(((targets != 0) & (targets != 1)).any()):
        raise ValueError("category_targets must contain only 0 or 1")
    return targets.bool()
