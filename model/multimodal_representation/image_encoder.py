"""FashionCLIP and Marqo FashionSigLIP encoders for clothing images."""

from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn
from torch.nn import functional as F

from ._marqo import DEFAULT_MODEL_NAME, _MarqoFeatures, _load_backbone, _positive_dimension


class VisualEncoder(nn.Module, ABC):
    """Contract required by the multimodal outfit encoder."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of features returned for each image."""

    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        """Return L2-normalized embeddings for images shaped ``[N, C, H, W]``."""


class FashionCLIPVisualEncoder(VisualEncoder):
    """FashionCLIP ViT image tower returning L2-normalized CLIP embeddings."""

    DEFAULT_MODEL_NAME = "patrickjohncyh/fashion-clip"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
        except ImportError as error:
            raise ImportError(
                "FashionCLIPVisualEncoder requires the 'transformers' package"
            ) from error

        self.backbone = CLIPVisionModelWithProjection.from_pretrained(model_name)
        projection_dim = self.backbone.config.projection_dim
        if projection_dim is None or projection_dim <= 0:
            raise ValueError(
                "FashionCLIP visual config must define a positive projection_dim"
            )
        self._output_dim = projection_dim
        self._trainable = trainable

        if not trainable:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> "FashionCLIPVisualEncoder":
        super().train(mode)
        if not self._trainable:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "images must have shape [items, channels, height, width]"
            )
        if images.size(0) == 0:
            raise ValueError("images cannot be empty")
        if images.size(1) != 3:
            raise ValueError("FashionCLIP expects three-channel images")
        embeddings = self.backbone(pixel_values=images).image_embeds
        return F.normalize(embeddings, p=2, dim=-1)


class MarqoFashionSigLIPVisualEncoder(VisualEncoder):
    """Encode preprocessed RGB images with the frozen Marqo image tower."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        super().__init__()
        self.backbone = _load_backbone(model_name)
        features = cast(_MarqoFeatures, self.backbone)
        # Both towers share this embedding space; the timm visual wrapper has no output_dim.
        self._output_dim = _positive_dimension(
            features.model.text.output_dim, "output_dim"
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> "MarqoFashionSigLIPVisualEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError("images must have shape [items, 3, height, width]")
        if images.size(0) == 0:
            raise ValueError("images cannot be empty")
        features = cast(_MarqoFeatures, self.backbone)
        embeddings = features.get_image_features(images, normalize=False)
        return F.normalize(embeddings, p=2, dim=-1)
