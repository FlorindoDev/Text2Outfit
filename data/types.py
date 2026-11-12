"""Shared samples and batches for fashion data."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor


class OutfitItem(BaseModel):
    """One raw item or one precomputed multimodal item embedding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Tensor | None = None
    text: str | None = None
    embedding: Tensor | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "OutfitItem":
        has_image = self.image is not None
        has_text = self.text is not None
        has_embedding = self.embedding is not None

        if has_image != has_text:
            raise ValueError("image and text must be provided together")
        if has_embedding == has_image:
            raise ValueError(
                "provide either image and text, or one precomputed embedding"
            )
        if self.image is not None and self.image.ndim != 3:
            raise ValueError("image must have shape [channels, height, width]")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.embedding is not None and self.embedding.ndim != 1:
            raise ValueError("embedding must be one-dimensional")
        return self


@dataclass(frozen=True)
class FashionItem:
    """One catalog item before conversion to the model input API."""

    item_id: str
    image: Tensor
    description: str
    category: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id cannot be empty")
        if self.image.ndim != 3 or self.image.size(0) != 3:
            raise ValueError("image must have shape [3, height, width]")
        if not torch.is_floating_point(self.image):
            raise TypeError("image must be a floating-point tensor")
        if not bool(torch.isfinite(self.image).all()):
            raise ValueError("image must contain only finite values")
        if not self.description.strip():
            raise ValueError("description cannot be empty")
        if not self.category.strip():
            raise ValueError("category cannot be empty")


@dataclass(frozen=True)
class ItemBatch:
    """Catalog items ready for multimodal processing."""

    item_ids: tuple[str, ...]
    categories: tuple[str, ...]
    model_items: tuple[OutfitItem, ...]

    def as_single_item_outfits(self) -> tuple[tuple[OutfitItem, ...], ...]:
        """Group each item into its own single-item outfit."""
        return tuple((item,) for item in self.model_items)

    def pin_memory(self) -> ItemBatch:
        """Let PyTorch pin tensors stored inside custom batch objects."""
        return ItemBatch(
            item_ids=self.item_ids,
            categories=self.categories,
            model_items=tuple(_pin_model_item(item) for item in self.model_items),
        )


def _pin_model_item(item: OutfitItem) -> OutfitItem:
    return OutfitItem(
        image=item.image.pin_memory() if item.image is not None else None,
        text=item.text,
        embedding=(
            item.embedding.pin_memory() if item.embedding is not None else None
        ),
    )
