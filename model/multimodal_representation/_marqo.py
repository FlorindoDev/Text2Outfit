"""Shared Marqo FashionSigLIP loading and feature contracts."""

from typing import Protocol

from torch import Tensor, nn
from ..config import SIGLIP_MODEL_NAME

DEFAULT_MODEL_NAME = SIGLIP_MODEL_NAME


class _TextTower(Protocol):
    output_dim: int
    context_length: int


class _MultimodalModel(Protocol):
    text: _TextTower


class _MarqoFeatures(Protocol):
    """Feature API provided by Marqo's dynamically loaded remote model."""

    model: _MultimodalModel

    def get_image_features(
        self, pixel_values: Tensor, *, normalize: bool = False
    ) -> Tensor: ...

    def get_text_features(
        self, input_ids: Tensor, *, normalize: bool = False
    ) -> Tensor: ...


def _load_backbone(model_name: str) -> nn.Module:
    from transformers.models.auto.modeling_auto import AutoModel

    # Marqo initializes OpenCLIP weights inside __init__, which cannot run on meta.
    return AutoModel.from_pretrained(
        model_name, trust_remote_code=True, low_cpu_mem_usage=False
    )


def _positive_dimension(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Marqo FashionSigLIP {name} must be a positive integer")
    return value
