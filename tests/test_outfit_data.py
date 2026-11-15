"""Exercise labelled outfit loading and batching without models or network access."""

import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from data import (
    OUTFIT_CATEGORIES,
    DataSplit,
    DatasetRequest,
    LoaderConfig,
    collate_items,
    collate_outfits,
    create_outfit_loader,
    get_dataset_source,
)
from data.polyvore import PolyvoreCatalog, PolyvoreOutfitDataset, load_item_categories


def image_transform(image):
    return torch.full((3, 2, 2), image.getpixel((0, 0))[0] / 255)


class OutfitDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.split_root = self.root / "nondisjoint"
        self.split_root.mkdir()
        self.metadata = {}
        for prefix, category, subcategory, count in (
            ("t", "tops", "21", 3),
            ("s", "shoes", "49", 4),
            ("j", "jewellery", 65, 4),
        ):
            for index in range(1, count + 1):
                self.metadata[f"{prefix}{index}"] = {
                    "title": f"{category} {index}",
                    "semantic_category": category,
                    "category_id": subcategory,
                }
        self.metadata_path = self.root / "polyvore_item_metadata.json"
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")
        original_outfits = [
            {"set_id": "a", "items": self._items("t1", "s1", "j1", "j2")},
            {"set_id": "b", "items": self._items("t2", "s2", "j3", "j4")},
            {"set_id": "c", "items": self._items("t3", "s3")},
            {"set_id": "d", "items": self._items("s4", "outside")},
        ]
        self.outfits_path = self.split_root / "train.json"
        self.outfits_path.write_text(json.dumps(original_outfits), encoding="utf-8")
        self.compatibility_path = self.split_root / "compatibility_train.txt"
        self.compatibility_path.write_text("1 a_1 a_2 a_3 a_4\n0 a_1 b_2\n", encoding="utf-8")
        self.rows = [
            {"item_id": item_id, "image": Image.new("RGB", (2, 2), (128, 0, 0))}
            for item_id in self.metadata
        ]
        self.catalog = PolyvoreCatalog(self.rows, self.metadata_path, image_transform)

    @staticmethod
    def _items(*item_ids):
        return [
            {"index": index, "item_id": item_id}
            for index, item_id in enumerate(item_ids, start=1)
        ]

    def dataset(self, **kwargs):
        return PolyvoreOutfitDataset(
            self.catalog, self.outfits_path, self.compatibility_path,
            negatives_per_item=2, **kwargs,
        )

    def test_labels_items_and_negatives_come_from_the_requested_split(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset), 2)
        for index, expected_label in enumerate((1, 0)):
            example = dataset[index]
            self.assertEqual(example.label, expected_label)
            self.assertEqual(example.outfit_id, f"compatibility_train:{index + 1}")
            self.assertEqual(len(example.items), (4, 2)[index])
            outfit_ids = {item.item_id for item in example.items}
            for item, negatives in zip(example.items, example.negatives):
                self.assertIsInstance(item.subcategory, str)
                self.assertEqual(len(negatives), 2)
                self.assertEqual(len({negative.item_id for negative in negatives}), 2)
                for negative in negatives:
                    self.assertEqual(negative.category, item.category)
                    self.assertNotIn(negative.item_id, outfit_ids)
                    self.assertIn(negative.item_id, self.catalog)
                    self.assertIsNotNone(negative.subcategory)
        self.assertEqual([item.item_id for item in dataset[1].items], ["t1", "s2"])

    def test_precomputed_positive_outfits_skip_image_decoding_and_keep_fixed_negatives(self):
        cache = {item_id: torch.randn(8) for item_id in self.metadata}
        dataset = self.dataset(precomputed_embeddings=cache, positive_only=True, seed=17)
        self.assertEqual(len(dataset), 1)
        with patch("data.polyvore.catalog._decode_image", side_effect=AssertionError("unexpected image decode")):
            first, second = dataset[0], dataset[0]
        self.assertEqual(first.label, 1)
        self.assertTrue(all(item.image is None and item.embedding is not None for item in first.items))
        self.assertEqual([[item.item_id for item in group] for group in first.negatives],
                         [[item.item_id for item in group] for group in second.negatives])
        batch = collate_items(first.items)
        self.assertTrue(all(item.embedding is not None and item.image is None for item in batch.model_items))

    def test_batch_keeps_multiple_items_per_category_and_boolean_masks(self):
        dataset = self.dataset()
        batch = next(iter(create_outfit_loader(dataset, config=LoaderConfig(batch_size=2))))
        self.assertEqual([len(outfit) for outfit in batch.outfits], [4, 2])
        self.assertEqual([item.item_id for item in batch.outfits[0]], ["t1", "s1", "j1", "j2"])
        self.assertEqual(batch.labels.dtype, torch.float32)
        self.assertEqual(batch.labels.tolist(), [1.0, 0.0])
        self.assertEqual(tuple(batch.category_mask.shape), (2, 11))
        self.assertEqual(batch.category_mask.dtype, torch.bool)
        for row, present in zip(batch.category_mask, ({"tops", "shoes", "jewelry"}, {"tops", "shoes"})):
            self.assertEqual(row.tolist(), [category in present for category in OUTFIT_CATEGORIES])
        self.assertEqual(batch.outfits[0][2].subcategory, "65")
        self.assertEqual([len(groups) for groups in batch.negatives], [4, 2])

    def test_initialization_does_not_decode_images(self):
        with patch.object(self.catalog, "get", side_effect=AssertionError("image decoded")):
            self.dataset()

    def test_jewellery_alias_and_existing_item_batch(self):
        self.assertEqual(load_item_categories(self.metadata_path)["j1"], "jewelry")
        item = self.catalog.get("j1")
        batch = collate_items([item])
        self.assertEqual(batch.categories, ("jewelry",))
        self.assertEqual(batch.item_ids, ("j1",))
        self.assertEqual(batch.as_single_item_outfits()[0][0].text, item.description)
        from model.outfit.category_query_embeddings import OUTFIT_CATEGORIES as query_categories
        self.assertIs(query_categories, OUTFIT_CATEGORIES)

    def test_invalid_compatibility_data_fails_clearly(self):
        for row, message in (
            ("2 a_1", "invalid compatibility row"),
            ("1", "invalid compatibility row"),
            ("0 unknown_1", "unknown outfit token"),
            ("1 d_2", "outside the catalog split"),
            ("\n", "cannot be empty"),
        ):
            with self.subTest(row=row):
                self.compatibility_path.write_text(row, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.dataset()

    def test_insufficient_negatives_and_invalid_counts_are_rejected(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive integer"):
                PolyvoreOutfitDataset(
                    self.catalog, self.outfits_path, self.compatibility_path,
                    negatives_per_item=count,
                )
        dataset = PolyvoreOutfitDataset(
            self.catalog, self.outfits_path, self.compatibility_path, negatives_per_item=3,
        )
        with self.assertRaisesRegex(ValueError, "distinct negatives outside the outfit"):
            dataset[0]

    def test_sample_contract_rejects_invalid_labels_categories_and_negatives(self):
        example = self.dataset()[0]
        for changes, message in (
            ({"label": 2}, "label"),
            ({"label": True}, "label"),
            ({"items": ()}, "items cannot be empty"),
            ({"negatives": ()}, "one negative group"),
            ({"items": (replace(example.items[0], subcategory=None), *example.items[1:])}, "requires a subcategory"),
            ({"items": (replace(example.items[0], category="unknown"), *example.items[1:])}, "unknown outfit category"),
            ({"negatives": ((example.items[0],), *example.negatives[1:])}, "cannot belong to the outfit"),
            ({"negatives": ((example.items[1],), *example.negatives[1:])}, "share their target"),
            ({"negatives": ((example.negatives[0][0],) * 2, *example.negatives[1:])}, "must be distinct"),
        ):
            with self.subTest(changes=changes.keys()), self.assertRaisesRegex(ValueError, message):
                replace(example, **changes)
        with self.assertRaisesRegex(ValueError, "empty batch"):
            collate_outfits([])

    def test_pin_memory_preserves_nested_items_and_targets(self):
        batch = collate_outfits([self.dataset()[0]])
        # Check traversal on CPU-only installations without allocating CUDA memory.
        with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor.clone()):
            pinned = batch.pin_memory()
        self.assertTrue(torch.equal(pinned.labels, batch.labels))
        self.assertTrue(torch.equal(pinned.category_mask, batch.category_mask))
        self.assertEqual(pinned.outfit_ids, batch.outfit_ids)
        for before, after in zip(batch.outfits[0], pinned.outfits[0]):
            self.assertEqual(before.item_id, after.item_id)
            self.assertEqual(before.subcategory, after.subcategory)
            self.assertTrue(torch.equal(before.image, after.image))
            self.assertNotEqual(before.image.data_ptr(), after.image.data_ptr())
        for before, after in zip(batch.negatives[0], pinned.negatives[0]):
            for old, new in zip(before, after):
                self.assertEqual(old.item_id, new.item_id)
                self.assertNotEqual(old.image.data_ptr(), new.image.data_ptr())

    def test_source_to_batch_from_local_parquet_without_downloads(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        image_bytes = io.BytesIO()
        Image.new("RGB", (2, 2), (128, 0, 0)).save(image_bytes, format="PNG")
        parquet_root = self.root / "data" / "nondisjoint"
        parquet_root.mkdir(parents=True)
        table = pa.table({
            "item_id": list(self.metadata),
            "image": [image_bytes.getvalue()] * len(self.metadata),
        })
        for split, stem, rows, expected_labels in (
            (DataSplit.TRAIN, "train", "1 a_1 a_2 a_3 a_4\n0 a_1 b_2\n", [1.0, 0.0]),
            (DataSplit.VALIDATION, "valid", "0 a_1 b_2\n", [0.0]),
            (DataSplit.TEST, "test", "1 a_1 a_2\n", [1.0]),
        ):
            pq.write_table(table, parquet_root / f"{split.value}.parquet")
            (self.split_root / f"{stem}.json").write_text(
                self.outfits_path.read_text(encoding="utf-8"), encoding="utf-8",
            )
            (self.split_root / f"compatibility_{stem}.txt").write_text(rows, encoding="utf-8")
            request = DatasetRequest(
                subset="nondisjoint", split=split, root=self.root,
                cache_dir=self.root / "cache", token=False,
            )
            with self.subTest(split=split), patch(
                "huggingface_hub.hf_hub_download", side_effect=AssertionError("unexpected download"),
            ):
                dataset = get_dataset_source("polyvore").outfit_dataset(
                    request, image_transform, negatives_per_item=2,
                )
                batch = next(iter(create_outfit_loader(dataset, config=LoaderConfig(batch_size=2))))
                item_dataset = get_dataset_source("polyvore").item_dataset(request, image_transform)
                self.assertEqual(batch.labels.tolist(), expected_labels)
                self.assertEqual(tuple(batch.category_mask.shape), (len(expected_labels), 11))
                self.assertEqual(len(item_dataset), len(self.metadata))
                self.assertEqual(item_dataset[0].subcategory, "21")


if __name__ == "__main__":
    unittest.main()
