"""Polyvore implementation of the public dataset-source contract."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from ..source import (
    DatasetDescriptor,
    DatasetDownloadRequest,
    DatasetRequest,
)
from ..transforms import ImageTransform
from ..types import FashionItem, OutfitExample

from .catalog import PolyvoreCatalog
from .download import (
    DEFAULT_DATASET_ROOT,
    POLYVORE_DATASET_ID,
    PolyvoreResources,
    PolyvoreSplit,
    PolyvoreTask,
    PolyvoreVariant,
    download_polyvore_resources,
)
from .item_dataset import PolyvoreItemDataset
from .outfit_dataset import PolyvoreOutfitDataset


class PolyvoreSource:
    """Adapt Polyvore files to project-neutral datasets."""

    descriptor = DatasetDescriptor(
        name="polyvore",
        dataset_id=POLYVORE_DATASET_ID,
        default_root=DEFAULT_DATASET_ROOT,
        subsets=tuple(variant.value for variant in PolyvoreVariant),
        default_subset=PolyvoreVariant.NONDISJOINT.value,
    )

    def item_dataset(
        self,
        request: DatasetRequest,
        image_transform: ImageTransform,
    ) -> Dataset[FashionItem]:
        resources = self._resources(request, PolyvoreTask.ITEMS)
        return PolyvoreItemDataset(_build_catalog(resources, image_transform))

    def outfit_dataset(
        self,
        request: DatasetRequest,
        image_transform: ImageTransform,
        *,
        negatives_per_item: int = 4,
    ) -> Dataset[OutfitExample]:
        resources = self._resources(request, PolyvoreTask.OUTFITS)
        if resources.outfits_path is None or resources.compatibility_path is None:
            raise RuntimeError("Polyvore outfit resources are incomplete")
        return PolyvoreOutfitDataset(
            _build_catalog(resources, image_transform),
            resources.outfits_path,
            resources.compatibility_path,
            negatives_per_item=negatives_per_item,
        )

    def download(self, request: DatasetDownloadRequest) -> Path:
        """Download complete Hugging Face repository to requested directory."""
        request.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ImportError(
                "Polyvore download requires the 'huggingface-hub' package"
            ) from error

        downloaded_path = snapshot_download(
            repo_id=self.descriptor.dataset_id,
            repo_type="dataset",
            revision=request.revision,
            cache_dir=request.cache_dir,
            local_dir=request.output_dir,
            token=request.token,
            max_workers=request.max_workers,
            force_download=request.force_download,
        )
        destination = Path(downloaded_path)
        if not destination.is_dir():
            raise FileNotFoundError(
                f"downloaded dataset directory does not exist: {destination}"
            )
        return destination

    def _resources(
        self,
        request: DatasetRequest,
        task: PolyvoreTask,
        *,
        include_items: bool = True,
        include_metadata: bool | None = None,
    ) -> PolyvoreResources:
        variant = PolyvoreVariant(
            self.descriptor.validate_subset(request.subset)
        )
        split = PolyvoreSplit(request.split.value)
        return download_polyvore_resources(
            task=task,
            variant=variant,
            split=split,
            token=request.token,
            cache_dir=request.cache_dir,
            dataset_root=request.root,
            include_items=include_items,
            include_metadata=include_metadata,
        )


def _build_catalog(
    resources: PolyvoreResources,
    image_transform: ImageTransform,
) -> PolyvoreCatalog:
    if resources.item_rows is None or resources.metadata_path is None:
        raise RuntimeError("Polyvore catalog resources are incomplete")
    return PolyvoreCatalog(
        resources.item_rows,
        resources.metadata_path,
        image_transform,
    )
