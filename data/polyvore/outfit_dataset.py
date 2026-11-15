"""Official Polyvore compatibility outfits with same-category retrieval negatives."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from torch.utils.data import Dataset
from torch import Tensor

from data.categories import OUTFIT_CATEGORIES
from data.types import OutfitExample

from .catalog import PolyvoreCatalog, load_outfit_token_index


class PolyvoreOutfitDataset(Dataset[OutfitExample]):
    """Keep official 0/1 labels and sample alternatives from the catalog's split."""

    def __init__(
        self,
        catalog: PolyvoreCatalog,
        outfits_path: str | Path,
        compatibility_path: str | Path,
        *,
        negatives_per_item: int = 4,
        precomputed_embeddings: Mapping[str, Tensor] | None = None,
        positive_only: bool = False,
        seed: int | None = None,
    ) -> None:
        if type(negatives_per_item) is not int or negatives_per_item <= 0:
            raise ValueError("negatives_per_item must be a positive integer")
        self.catalog = catalog
        self.negatives_per_item = negatives_per_item
        self.precomputed_embeddings = precomputed_embeddings
        self.seed = seed
        self._categories = {
            item_id: catalog.category_for(item_id) for item_id in catalog.item_ids
        }
        grouped: dict[str, list[str]] = defaultdict(list)
        for item_id, category in self._categories.items():
            if category not in OUTFIT_CATEGORIES:
                raise ValueError(f"unknown category {category!r} for item {item_id}")
            grouped[category].append(item_id)
        self._items_by_category = {
            category: tuple(item_ids) for category, item_ids in grouped.items()
        }
        token_index = load_outfit_token_index(outfits_path)
        self._outfits = _load_compatibility(compatibility_path, token_index, catalog)
        if positive_only:
            self._outfits = tuple(outfit for outfit in self._outfits if outfit[1] == 1)
            if not self._outfits:
                raise ValueError("no positive outfits available for generation training")

    def __len__(self) -> int:
        return len(self._outfits)

    def __getitem__(self, index: int) -> OutfitExample:
        outfit_id, label, item_ids = self._outfits[index]
        rng = random if self.seed is None else random.Random(self.seed + index)
        excluded: dict[str, set[str]] = defaultdict(set)
        for item_id in item_ids:
            excluded[self._categories[item_id]].add(item_id)

        negative_ids = []
        for item_id in item_ids:
            category = self._categories[item_id]
            pool = self._items_by_category[category]
            count = self.negatives_per_item + len(excluded[category])
            if count > len(pool):
                raise ValueError(
                    f"outfit {outfit_id}: category {category!r} has fewer than "
                    f"{self.negatives_per_item} distinct negatives outside the outfit"
                )
            # Overdraw by the excluded count so filtering still leaves N candidates.
            candidates = rng.sample(pool, count)
            negative_ids.append(tuple(
                candidate for candidate in candidates if candidate not in excluded[category]
            )[:self.negatives_per_item])

        # Repeated candidates within this example share the same decoded image.
        needed = dict.fromkeys(item_ids)
        for group in negative_ids:
            needed.update(dict.fromkeys(group))
        items = {
            item_id: self.catalog.get(
                item_id, embedding=(self.precomputed_embeddings[item_id]
                                   if self.precomputed_embeddings is not None else None),
            ) for item_id in needed
        }
        return OutfitExample(
            outfit_id=outfit_id,
            items=tuple(items[item_id] for item_id in item_ids),
            label=label,
            negatives=tuple(
                tuple(items[item_id] for item_id in group) for group in negative_ids
            ),
        )


def _load_compatibility(
    path: str | Path,
    token_index: dict[str, str],
    catalog: PolyvoreCatalog,
) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    selected_path = Path(path)
    outfits = []
    with selected_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields:
                continue
            if fields[0] not in ("0", "1") or len(fields) < 2:
                raise ValueError(f"invalid compatibility row at {selected_path}:{line_number}")
            item_ids = []
            for token in fields[1:]:
                if token not in token_index:
                    raise ValueError(f"unknown outfit token {token!r} at line {line_number}")
                item_id = token_index[token]
                if item_id not in catalog:
                    raise ValueError(
                        f"item {item_id} at line {line_number} is outside the catalog split"
                    )
                item_ids.append(item_id)
            outfits.append((
                f"{selected_path.stem}:{line_number}", int(fields[0]), tuple(item_ids),
            ))
    if not outfits:
        raise ValueError("Polyvore compatibility file cannot be empty")
    return tuple(outfits)
