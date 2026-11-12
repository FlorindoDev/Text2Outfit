"""Image preprocessing factories for the supported visual encoders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from PIL import Image
from torch import Tensor

ImageTransform = Callable[[Image.Image], Tensor]


class _ImageProcessor(Protocol):
    def __call__(
        self,
        *,
        images: Image.Image,
        return_tensors: Literal["pt"],
    ) -> Mapping[str, Tensor]: ...


@dataclass(frozen=True)
class _ProcessedImageTransform:
    processor: _ImageProcessor

    def __call__(self, image: Image.Image) -> Tensor:
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        if pixel_values.ndim != 4 or pixel_values.size(0) != 1:
            raise ValueError(
                "image processor must return pixel_values shaped [1, C, H, W]"
            )
        return pixel_values.squeeze(0)


def build_fashion_clip_transform(
    model_name: str = "patrickjohncyh/fashion-clip",
) -> ImageTransform:
    """Build preprocessing from the selected FashionCLIP checkpoint."""
    if not model_name.strip():
        raise ValueError("model_name cannot be empty")
    try:
        from transformers.models.auto.image_processing_auto import AutoImageProcessor
    except ImportError as error:
        raise ImportError(
            "build_fashion_clip_transform requires the 'transformers' package"
        ) from error

    processor = cast(_ImageProcessor, AutoImageProcessor.from_pretrained(model_name))
    return _ProcessedImageTransform(processor)


def build_marqo_fashion_siglip_transform(
    model_name: str = "Marqo/marqo-fashionSigLIP",
) -> ImageTransform:
    """Build Marqo's RGB preprocessing using its official processor."""
    if not model_name.strip():
        raise ValueError("model_name cannot be empty")
    from transformers.models.auto.processing_auto import AutoProcessor

    processor = cast(
        _ImageProcessor,
        AutoProcessor.from_pretrained(model_name, trust_remote_code=True),
    )
    return _ProcessedImageTransform(processor)
