"""Data ingestion, preprocessing and batching for fashion items and outfits."""

from .categories import OUTFIT_CATEGORIES
from .collate import collate_items, collate_outfits
from .loaders import (
    LoaderConfig,
    create_item_loader,
    create_outfit_loader,
)
from .source import (
    DEFAULT_DATASET_NAME,
    DataSplit,
    DatasetDescriptor,
    DatasetDownloadRequest,
    DatasetRequest,
    DatasetSource,
    IndexedDataset,
    available_dataset_names,
    get_dataset_source,
)
from .transforms import (
    ImageTransform,
    build_fashion_clip_transform,
    build_marqo_fashion_siglip_transform,
)
from .types import (
    FashionItem,
    ItemBatch,
    OutfitBatch,
    OutfitExample,
    OutfitItem,
    SeedExample,
    SeedTrainingBatch,
)

__all__ = [
    "DEFAULT_DATASET_NAME",
    "DataSplit",
    "DatasetDescriptor",
    "DatasetDownloadRequest",
    "DatasetRequest",
    "DatasetSource",
    "FashionItem",
    "ImageTransform",
    "IndexedDataset",
    "ItemBatch",
    "LoaderConfig",
    "OUTFIT_CATEGORIES",
    "OutfitBatch",
    "OutfitExample",
    "OutfitItem",
    "SeedExample",
    "SeedTrainingBatch",
    "build_fashion_clip_transform",
    "build_marqo_fashion_siglip_transform",
    "collate_items",
    "collate_outfits",
    "create_item_loader",
    "create_outfit_loader",
    "available_dataset_names",
    "get_dataset_source",
]
