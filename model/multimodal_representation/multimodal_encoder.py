"""Project concatenated visual and text embeddings through an MLP head."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ..mlp import MLP
from .image_encoder import VisualEncoder
from .text_encoder import TextEncoder


class MultimodalEncoder(nn.Module):
    """Feed visual-then-text embeddings to a configurable MLP head."""

    def __init__(
        self, visual_encoder: VisualEncoder | None, text_encoder: TextEncoder | None, mlp: MLP
    ) -> None:
        super().__init__()
        if (visual_encoder is None) != (text_encoder is None):
            raise ValueError("provide both encoders, or neither for precomputed features")
        if visual_encoder is not None and text_encoder is not None and (
            mlp.input_dim != visual_encoder.output_dim + text_encoder.output_dim
        ):
            raise ValueError("MLP input_dim must match the sum of visual and text output_dim")
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.mlp = mlp

    @property
    def output_dim(self) -> int:
        return self.mlp.output_dim

    def forward(
        self, images: Tensor | None = None, descriptions: Sequence[str] | None = None,
        *, precomputed: Tensor | None = None,
    ) -> Tensor:
        """Project raw items or cached [visual_L2, text_L2] features through the MLP."""
        if precomputed is not None:
            if images is not None or descriptions is not None:
                raise ValueError("provide raw inputs or precomputed features, not both")
            if precomputed.ndim != 2 or precomputed.size(0) == 0 or precomputed.size(1) != self.mlp.input_dim:
                raise ValueError(f"precomputed must have shape [items, {self.mlp.input_dim}]")
            if not torch.is_floating_point(precomputed) or not bool(torch.isfinite(precomputed).all()):
                raise ValueError("precomputed features must be finite floating-point values")
            weights = next(self.mlp.parameters())
            return self.mlp(precomputed.to(device=weights.device, dtype=weights.dtype))
        if images is None or descriptions is None:
            raise ValueError("provide images and descriptions or precomputed features")
        if self.visual_encoder is None or self.text_encoder is None:
            raise ValueError("raw inputs require visual and text encoders")
        if images.ndim != 4:
            raise ValueError("images must have shape [items, channels, height, width]")
        if images.size(0) != len(descriptions):
            raise ValueError("images and descriptions must contain the same number of items")
        visual_embeddings = self.visual_encoder(images)
        text_embeddings = self.text_encoder(descriptions)
        return self.mlp(torch.cat((visual_embeddings, text_embeddings), dim=-1))
