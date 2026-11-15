"""Collate functions for catalog items and labelled outfits."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from data.types import FashionItem, ItemBatch, OutfitBatch, OutfitExample, OutfitItem


def collate_items(examples: Sequence[FashionItem]) -> ItemBatch:
    """Create a batch for item embedding precomputation."""
    _require_non_empty(examples)
    return ItemBatch(
        item_ids=tuple(example.item_id for example in examples),
        categories=tuple(example.category for example in examples),
        model_items=tuple(_to_model_item(example) for example in examples),
    )


def collate_outfits(examples: Sequence[OutfitExample]) -> OutfitBatch:
    """Preserve every item, including multiple items of the same category."""
    _require_non_empty(examples)
    return OutfitBatch(
        outfit_ids=tuple(example.outfit_id for example in examples),
        outfits=tuple(example.items for example in examples),
        labels=torch.tensor([example.label for example in examples], dtype=torch.float32),
        negatives=tuple(example.negatives for example in examples),
        category_mask=torch.stack([example.category_mask for example in examples]),
    )


def _require_non_empty(examples: Sequence[object]) -> None:
    if not examples:
        raise ValueError("cannot collate an empty batch")


def _to_model_item(item: FashionItem) -> OutfitItem:
    return OutfitItem(image=item.image, text=item.description)
