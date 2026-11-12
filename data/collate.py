"""Collate functions for fashion items."""

from __future__ import annotations

from collections.abc import Sequence

from data.types import FashionItem, ItemBatch, OutfitItem


def collate_items(examples: Sequence[FashionItem]) -> ItemBatch:
    """Create a batch for item embedding precomputation."""
    _require_non_empty(examples)
    return ItemBatch(
        item_ids=tuple(example.item_id for example in examples),
        categories=tuple(example.category for example in examples),
        model_items=tuple(_to_model_item(example) for example in examples),
    )


def _require_non_empty(examples: Sequence[object]) -> None:
    if not examples:
        raise ValueError("cannot collate an empty batch")


def _to_model_item(item: FashionItem) -> OutfitItem:
    return OutfitItem(image=item.image, text=item.description)
