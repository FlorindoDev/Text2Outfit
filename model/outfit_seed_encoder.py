"""Encode an outfit's seed image with the visual encoder and an MLP."""

from torch import Tensor, nn

from .mlp import MLP
from .multimodal_representation import VisualEncoder


class OutfitSeedEncoder(nn.Module):
    """Project a seed's visual embedding through a configurable MLP."""

    def __init__(self, visual_encoder: VisualEncoder, mlp: MLP) -> None:
        super().__init__()
        if visual_encoder.output_dim != mlp.input_dim:
            raise ValueError("MLP input_dim must match visual encoder output_dim")
        self.visual_encoder = visual_encoder
        self.mlp = mlp

    @property
    def output_dim(self) -> int:
        return self.mlp.output_dim

    def forward(self, images: Tensor) -> Tensor:
        """Encode preprocessed ``[3, H, W]`` or ``[N, 3, H, W]`` as ``[N, D]``."""
        if images.ndim == 3:
            images = images.unsqueeze(0)
        return self.mlp(self.visual_encoder(images))
