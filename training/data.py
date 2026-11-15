"""Positive outfit batches and the existing per-item feature cache format."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data import OUTFIT_CATEGORIES, collate_outfits
from data.polyvore.catalog import PolyvoreCatalog
from data.source import IndexedDataset
from data.types import OutfitExample, SeedExample, SeedTrainingBatch
from .config import DEFAULT_TRAINING_CONFIG


class PrecomputedItemFeatures(Mapping[str, Tensor]):
    """Index IDs in manifest shards; map at most two shards into memory on demand."""

    def __init__(self, manifest: str | Path) -> None:
        self.path = Path(manifest).resolve()
        self.metadata = json.loads(self.path.read_text(encoding="utf-8"))
        self.embedding_dim = self.metadata["embedding_dim"]
        if type(self.embedding_dim) is not int or self.embedding_dim <= 0 or self.embedding_dim % 2:
            raise ValueError("cache embedding_dim must be a positive even integer")
        encoder = self.metadata.get("encoder", {})
        if encoder.get("aggregation") != "concat_visual_then_text" or encoder.get("normalization") != "l2_per_modality":
            raise ValueError("cache must contain concatenated visual/text features before the item MLP")
        if self.embedding_dim != 2 * encoder.get("modality_dim", 0):
            raise ValueError("cache modality_dim and embedding_dim disagree")
        self._index: dict[str, tuple[Path, int]] = {}
        for shard in self.metadata["shards"]:
            name = Path(shard["file"])
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("cache shards must be relative to the manifest directory")
            path = self.path.parent / name
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            ids, features = payload["item_ids"], payload["embeddings"]
            if features.shape != (len(ids), self.embedding_dim) or not torch.is_floating_point(features):
                raise ValueError(f"invalid cache shard dimensions: {path}")
            for index, item_id in enumerate(ids):
                item_id = str(item_id).strip()
                if not item_id:
                    raise ValueError("cached item IDs cannot be empty")
                if item_id in self._index:
                    raise ValueError(f"duplicate cached item_id: {item_id}")
                self._index[item_id] = (path, index)
        if not self._index or len(self._index) != self.metadata["count"]:
            raise ValueError("cache count does not match its shards")

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)

    def __getitem__(self, key: str) -> Tensor:
        path, index = self._index[key]
        return self._shard(path)[index]

    @staticmethod
    @lru_cache(maxsize=2)
    def _shard(path: Path) -> Tensor:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)["embeddings"]


class SeedOutfitDataset(Dataset[SeedExample]):
    """Add a prompt and seed; validation seed selection stays fixed across epochs."""

    def __init__(
        self, outfits: IndexedDataset[OutfitExample], *, training: bool,
        prompt: str = DEFAULT_TRAINING_CONFIG.prompt,
        prompts: Mapping[str, str] | None = None,
    ) -> None:
        if not prompt.strip():
            raise ValueError("default prompt cannot be empty")
        self.outfits, self.training, self.prompt = outfits, training, prompt
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.outfits)

    def __getitem__(self, index: int) -> SeedExample:
        outfit = self.outfits[index]
        if outfit.label != 1:
            raise ValueError("generation targets must be positive outfits (label=1)")
        seed = random.choice(outfit.items) if self.training else outfit.items[index % len(outfit.items)]
        prompt = self.prompt if self.prompts is None else self.prompts[outfit.outfit_id]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"invalid prompt for {outfit.outfit_id}")
        return SeedExample(outfit, prompt, seed)


def collate_seed_outfits(examples: Sequence[SeedExample]) -> SeedTrainingBatch:
    return SeedTrainingBatch(
        collate_outfits([example.outfit for example in examples]),
        tuple(example.prompt for example in examples), tuple(example.seed for example in examples),
    )


def subcategory_vocabulary(*catalogs: PolyvoreCatalog) -> tuple[tuple[str, ...], ...]:
    """Use one fixed taxonomy for train and validation, without loading images."""
    values: dict[str, set[str]] = {category: set() for category in OUTFIT_CATEGORIES}
    for catalog in catalogs:
        for item_id in catalog.item_ids:
            label = catalog.subcategory_for(item_id)
            category = catalog.category_for(item_id)
            if category not in values or label is None:
                raise ValueError(f"invalid category/subcategory for item {item_id}")
            values[category].add(label)
    return tuple(tuple(sorted(values[category])) for category in OUTFIT_CATEGORIES)
