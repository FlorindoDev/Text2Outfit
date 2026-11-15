"""Encode an outfit's seed image with the visual encoder and an MLP."""

import torch
from torch import Tensor, nn

from ..mlp import MLP
from ..multimodal_representation import VisualEncoder


class OutfitSeedEncoder(nn.Module):
    """Project a seed's visual embedding through a configurable MLP."""

    def __init__(self, visual_encoder: VisualEncoder | None, mlp: MLP) -> None:
        super().__init__()
        if visual_encoder is not None and visual_encoder.output_dim != mlp.input_dim:
            raise ValueError("MLP input_dim must match visual encoder output_dim")
        self.visual_encoder = visual_encoder
        self.mlp = mlp

    @property
    def output_dim(self) -> int:
        return self.mlp.output_dim

    def forward(self, images: Tensor | None = None, *, precomputed: Tensor | None = None) -> Tensor:
        """Encode preprocessed ``[3, H, W]`` or ``[N, 3, H, W]`` as ``[N, D]``."""
        if precomputed is not None:
            if images is not None:
                raise ValueError("provide images or precomputed visual features")
            if precomputed.ndim != 2 or precomputed.size(1) != self.mlp.input_dim:
                raise ValueError(f"seed features must have shape [B, {self.mlp.input_dim}]")
            if not torch.is_floating_point(precomputed) or not bool(torch.isfinite(precomputed).all()):
                raise ValueError("seed features must be finite floating-point values")
            features = precomputed
        elif images is not None and self.visual_encoder is not None:
            if images.ndim == 3:
                images = images.unsqueeze(0)
            features = self.visual_encoder(images)
        else:
            raise ValueError("provide images with a visual encoder, or precomputed features")
        weights = next(self.mlp.parameters())
        return self.mlp(features.to(device=weights.device, dtype=weights.dtype))
