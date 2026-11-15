"""Shared samples and batches for fashion data."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor

from .categories import OUTFIT_CATEGORIES


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
    image: Tensor | None
    description: str
    category: str
    subcategory: str | None = None
    embedding: Tensor | None = None

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id cannot be empty")
        if self.image is None and self.embedding is None:
            raise ValueError("provide an image or precomputed item features")
        if self.image is not None:
            if self.image.ndim != 3 or self.image.size(0) != 3:
                raise ValueError("image must have shape [3, height, width]")
            if not torch.is_floating_point(self.image):
                raise TypeError("image must be a floating-point tensor")
            if not bool(torch.isfinite(self.image).all()):
                raise ValueError("image must contain only finite values")
        if self.embedding is not None:
            if self.embedding.ndim != 1 or self.embedding.numel() == 0:
                raise ValueError("embedding must be a non-empty feature vector")
            if not torch.is_floating_point(self.embedding) or not bool(torch.isfinite(self.embedding).all()):
                raise ValueError("embedding must contain finite floating-point values")
        if not self.description.strip():
            raise ValueError("description cannot be empty")
        if not self.category.strip():
            raise ValueError("category cannot be empty")
        if self.subcategory is not None and not self.subcategory.strip():
            raise ValueError("subcategory cannot be empty")


@dataclass(frozen=True)
class OutfitExample:
    """One labelled outfit, with retrieval alternatives aligned to its items."""

    outfit_id: str
    items: tuple[FashionItem, ...]
    label: int
    negatives: tuple[tuple[FashionItem, ...], ...]

    def __post_init__(self) -> None:
        if not self.outfit_id.strip():
            raise ValueError("outfit_id cannot be empty")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("label must be the integer 0 or 1")
        if not self.items:
            raise ValueError("outfit items cannot be empty")
        if len(self.negatives) != len(self.items):
            raise ValueError("provide one negative group per outfit item")
        outfit_ids = {item.item_id for item in self.items}
        for item, negatives in zip(self.items, self.negatives):
            if item.category not in OUTFIT_CATEGORIES:
                raise ValueError(f"unknown outfit category: {item.category}")
            if item.subcategory is None:
                raise ValueError(f"item {item.item_id} requires a subcategory")
            if not negatives:
                raise ValueError("each outfit item requires at least one negative")
            negative_ids = [negative.item_id for negative in negatives]
            if len(set(negative_ids)) != len(negative_ids):
                raise ValueError("negative items must be distinct within each group")
            for negative in negatives:
                if negative.category != item.category:
                    raise ValueError("negatives must share their target item's category")
                if negative.item_id in outfit_ids:
                    raise ValueError("negative items cannot belong to the outfit")
                if negative.subcategory is None:
                    raise ValueError(f"negative {negative.item_id} requires a subcategory")

    @property
    def category_mask(self) -> Tensor:
        """Boolean presence vector in the shared 11-category order."""
        present = {item.category for item in self.items}
        return torch.tensor([category in present for category in OUTFIT_CATEGORIES])


@dataclass(frozen=True)
class OutfitBatch:
    """Ragged outfits and negatives, with labels [B] and presence masks [B, 11]."""

    outfit_ids: tuple[str, ...]
    outfits: tuple[tuple[FashionItem, ...], ...]
    labels: Tensor
    negatives: tuple[tuple[tuple[FashionItem, ...], ...], ...]
    category_mask: Tensor

    def pin_memory(self) -> OutfitBatch:
        return OutfitBatch(
            outfit_ids=self.outfit_ids,
            outfits=tuple(
                tuple(_pin_fashion_item(item) for item in outfit)
                for outfit in self.outfits
            ),
            labels=self.labels.pin_memory(),
            negatives=tuple(
                tuple(
                    tuple(_pin_fashion_item(item) for item in group)
                    for group in outfit_negatives
                )
                for outfit_negatives in self.negatives
            ),
            category_mask=self.category_mask.pin_memory(),
        )


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


def _pin_fashion_item(item: FashionItem) -> FashionItem:
    return replace(
        item, image=item.image.pin_memory() if item.image is not None else None,
        embedding=item.embedding.pin_memory() if item.embedding is not None else None,
    )


@dataclass(frozen=True)
class SeedExample:
    outfit: OutfitExample
    prompt: str
    seed: FashionItem


@dataclass(frozen=True)
class SeedTrainingBatch:
    outfits: OutfitBatch
    prompts: tuple[str, ...]
    seeds: tuple[FashionItem, ...]
