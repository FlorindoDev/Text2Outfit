"""Exercise the extracted data pipeline without network or model dependencies."""

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import datasets
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torchvision import transforms

from data import (
    DataSplit,
    DatasetRequest,
    FashionItem,
    LoaderConfig,
    OutfitItem,
    collate_items,
    create_item_loader,
    get_dataset_source,
)
from data.loaders import build_polyvore_item_loader
from data.polyvore import PolyvoreCatalog, load_outfit_token_index
from data.polyvore.download import download_polyvore_resources
from scripts.download_polyvore import download_dataset, parse_args


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache_patch = patch.object(datasets.config, "HF_DATASETS_CACHE", self.cache)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        image_bytes = BytesIO()
        Image.new("RGBA", (12, 8), (200, 100, 50, 255)).save(image_bytes, "PNG")
        self.image_bytes = image_bytes.getvalue()
        self.rows = [
            {"item_id": "1", "image": {"bytes": self.image_bytes, "path": None}},
            {"item_id": "2", "image": {"bytes": self.image_bytes, "path": None}},
        ]
        self.metadata = self.root / "polyvore_item_metadata.json"
        self.metadata.write_text(json.dumps({
            "1": {
                "title": "  Red   shirt ",
                "description": "Cotton shirt",
                "semantic_category": "tops",
            },
        }), encoding="utf-8")
        self.transform = transforms.Compose([
            transforms.Resize((16, 16)), transforms.ToTensor(),
        ])
        self.source = get_dataset_source(" POLYVORE ")
        self.request = DatasetRequest(
            subset="disjoint", split=DataSplit.VALIDATION,
            root=self.root, cache_dir=self.cache, token=False,
        )

    def write_parquet(self):
        path = self.root / "data/disjoint/validation.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(self.rows), path)

    def assert_batch(self, loader):
        batch = next(iter(loader))
        self.assertEqual(batch.item_ids, ("1", "2"))
        self.assertEqual(batch.categories, ("tops", "unknown"))
        self.assertEqual(batch.model_items[0].text, "Red shirt. Cotton shirt")
        self.assertEqual(batch.model_items[1].text, "fashion item")
        self.assertEqual(batch.model_items[0].image.shape, (3, 16, 16))
        self.assertEqual(batch.as_single_item_outfits()[0], (batch.model_items[0],))

    @patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("unexpected network"))
    @patch("datasets.load_dataset", side_effect=AssertionError("unexpected network"))
    def test_local_parquet_to_batch_through_both_public_entry_points(self, *_):
        self.write_parquet()
        dataset = self.source.item_dataset(self.request, self.transform)
        self.assert_batch(create_item_loader(dataset, config=LoaderConfig(batch_size=2)))
        self.assert_batch(build_polyvore_item_loader(
            image_transform=self.transform, variant="disjoint", split="validation",
            dataset_root=self.root, token=False, config=LoaderConfig(batch_size=2),
        ))

    @patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("unexpected network"))
    @patch("datasets.load_dataset", side_effect=AssertionError("unexpected network"))
    def test_arrow_cache_is_used_when_local_parquet_is_missing(self, *_):
        arrow = self.cache / "mvasil___polyvore-outfits/disjoint/1.0/hash/polyvore-outfits-validation.arrow"
        arrow.parent.mkdir(parents=True)
        table = pa.Table.from_pylist(self.rows)
        with pa.OSFile(str(arrow), "wb") as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
        dataset = self.source.item_dataset(self.request, self.transform)
        self.assert_batch(create_item_loader(dataset, config=LoaderConfig(batch_size=2)))

    def test_missing_resources_use_remote_fallback_and_preserve_authentication(self):
        remote_metadata = self.root / "remote_metadata.json"
        self.metadata.rename(remote_metadata)
        rows = datasets.Dataset.from_list(self.rows)
        with (
            patch("datasets.load_dataset", return_value=rows) as remote_rows,
            patch("huggingface_hub.hf_hub_download", side_effect=[
                LocalEntryNotFoundError("not cached"), str(remote_metadata),
            ]) as remote_file,
        ):
            dataset = self.source.item_dataset(self.request, self.transform)
        self.assert_batch(create_item_loader(dataset, config=LoaderConfig(batch_size=2)))
        remote_rows.assert_called_once_with(
            "mvasil/polyvore-outfits", "disjoint", split="validation",
            token=False, cache_dir=str(self.cache),
        )
        self.assertEqual(remote_file.call_count, 2)
        self.assertTrue(remote_file.call_args_list[0].kwargs["local_files_only"])
        self.assertNotIn("local_files_only", remote_file.call_args_list[1].kwargs)
        self.assertIs(remote_file.call_args.kwargs["token"], False)

    def test_metadata_already_in_hub_cache_does_not_trigger_remote_download(self):
        rows = datasets.Dataset.from_list(self.rows)
        remote_metadata = self.root / "cached_metadata.json"
        self.metadata.rename(remote_metadata)
        with (
            patch("datasets.load_dataset", return_value=rows),
            patch("huggingface_hub.hf_hub_download", return_value=str(remote_metadata)) as cached_file,
        ):
            dataset = self.source.item_dataset(self.request, self.transform)
        self.assertEqual(dataset[0].category, "tops")
        cached_file.assert_called_once()
        self.assertTrue(cached_file.call_args.kwargs["local_files_only"])

    def test_all_supported_image_representations_are_decoded_as_rgb(self):
        image_path = self.root / "item.png"
        image_path.write_bytes(self.image_bytes)
        images = [
            Image.new("RGBA", (12, 8)), self.image_bytes, image_path,
            {"bytes": self.image_bytes, "path": None},
            {"bytes": None, "path": str(image_path)},
        ]
        for image in images:
            with self.subTest(image_type=type(image).__name__):
                catalog = PolyvoreCatalog(
                    [{"item_id": "1", "image": image}], self.metadata, self.transform,
                )
                self.assertEqual(catalog.get("1").image.shape, (3, 16, 16))

    def test_invalid_inputs_are_rejected(self):
        invalid_calls = [
            lambda: collate_items([]),
            lambda: get_dataset_source("unknown"),
            lambda: LoaderConfig(batch_size=0).validate(),
            lambda: LoaderConfig(persistent_workers=True).validate(),
            lambda: self.source.item_dataset(
                DatasetRequest("invalid", DataSplit.TRAIN, self.root), self.transform,
            ),
            lambda: OutfitItem(),
            lambda: OutfitItem(image=torch.zeros(3, 2, 2)),
            lambda: FashionItem("1", torch.full((3, 2, 2), float("nan")), "shirt", "tops"),
            lambda: PolyvoreCatalog([self.rows[0], self.rows[0]], self.metadata, self.transform),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()
        catalog = PolyvoreCatalog(self.rows, self.metadata, self.transform)
        with self.assertRaises(KeyError):
            catalog.get("missing")

    def test_outfit_mapping_and_precomputed_item_interface_are_preserved(self):
        path = self.root / "outfits.json"
        path.write_text(json.dumps([
            {"set_id": "outfit", "items": [{"item_id": "1", "index": 2}]},
        ]), encoding="utf-8")
        self.assertEqual(load_outfit_token_index(path), {"outfit_2": "1"})
        item = OutfitItem(embedding=torch.ones(8))
        self.assertEqual(item.embedding.shape, (8,))

    @patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("unexpected network"))
    @patch("datasets.load_dataset", side_effect=AssertionError("unexpected network"))
    def test_metadata_only_request_does_not_load_images(self, *_):
        resources = download_polyvore_resources(
            task="items", dataset_root=self.root, include_items=False,
            include_metadata=True, token=False,
        )
        self.assertIsNone(resources.item_rows)
        self.assertEqual(resources.metadata_path, self.metadata)

    def test_download_cli_preserves_request_options(self):
        output = self.root / "download"
        config = parse_args([
            "--output-dir", str(output), "--cache-dir", str(self.cache),
            "--revision", "example", "--no-token", "--max-workers", "2",
            "--force-download",
        ])
        with patch("huggingface_hub.snapshot_download", return_value=str(output)) as download:
            self.assertEqual(download_dataset(config), output)
        download.assert_called_once_with(
            repo_id="mvasil/polyvore-outfits", repo_type="dataset", revision="example",
            cache_dir=self.cache, local_dir=output, token=False, max_workers=2,
            force_download=True,
        )


if __name__ == "__main__":
    unittest.main()
