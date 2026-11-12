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
        self, visual_encoder: VisualEncoder, text_encoder: TextEncoder, mlp: MLP
    ) -> None:
        super().__init__()
        if mlp.input_dim != visual_encoder.output_dim + text_encoder.output_dim:
            raise ValueError("MLP input_dim must match the sum of visual and text output_dim")
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.mlp = mlp

    @property
    def output_dim(self) -> int:
        return self.mlp.output_dim

    def forward(self, images: Tensor, descriptions: Sequence[str]) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [items, channels, height, width]")
        if images.size(0) != len(descriptions):
            raise ValueError("images and descriptions must contain the same number of items")
        visual_embeddings = self.visual_encoder(images)
        text_embeddings = self.text_encoder(descriptions)
        return self.mlp(torch.cat((visual_embeddings, text_embeddings), dim=-1))
