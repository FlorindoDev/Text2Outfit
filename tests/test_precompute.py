"""Exercise embedding shards and the local precomputation pipeline."""

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import datasets
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torchvision import transforms

from data import ItemBatch, OutfitItem
from scripts.precompute_embeddings import (
    DEFAULT_MARQO_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    EmbeddingShardWriter,
    _build_encoders,
    _build_image_transform,
    encode_item_batch,
    parse_args,
    resolve_device,
    run,
)


class FixedEncoder(nn.Module):
    """Small feature source with the same interface as the frozen encoders."""

    output_dim = 3

    def __init__(self, values):
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.register_buffer("features", torch.tensor(values, dtype=torch.float32))

    def forward(self, values):
        return self.features.expand(len(values), -1)


class PrecomputeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        cache_patch = patch.object(datasets.config, "HF_DATASETS_CACHE", self.root / "default-hf")
        cache_patch.start()
        self.addCleanup(cache_patch.stop)

    def writer(self, **kwargs):
        options = {"shard_size": 2, "output_dtype": "float32"}
        options.update(kwargs)
        return EmbeddingShardWriter(self.root / "output", **options)

    def test_shards_round_trip_in_order_and_preserve_metadata(self):
        writer = self.writer(output_dtype="float16")
        features = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        writer.add(["a", "b", "c"], features[:3])
        writer.add(["d", "e"], features[3:])
        manifest_path = writer.finalize({"model_fingerprint": "example"})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["count"], 5)
        self.assertEqual(manifest["embedding_dim"], 3)
        self.assertEqual(manifest["model_fingerprint"], "example")
        self.assertEqual([(s["start"], s["end"]) for s in manifest["shards"]],
                         [(0, 2), (2, 4), (4, 5)])
        shards = [torch.load(manifest_path.parent / s["file"], weights_only=True)
                  for s in manifest["shards"]]
        self.assertEqual(tuple(i for s in shards for i in s["item_ids"]), tuple("abcde"))
        torch.testing.assert_close(torch.cat([s["embeddings"] for s in shards]), features.half())
        self.assertFalse(list(manifest_path.parent.glob("*.tmp")))
        with self.assertRaises(RuntimeError):
            writer.add(["f"], features[:1])
        with self.assertRaises(RuntimeError):
            writer.finalize({})

    def test_invalid_batches_do_not_enter_the_cache(self):
        writer = self.writer(output_dtype="float16")
        invalid = [
            ([], torch.ones(0, 3)),
            ([" "], torch.ones(1, 3)),
            (["a", "a"], torch.ones(2, 3)),
            (["a"], torch.ones(2, 3)),
            (["a"], torch.ones(3)),
            (["a"], torch.ones(1, 0)),
            (["a"], torch.ones(1, 3, dtype=torch.long)),
            (["a"], torch.full((1, 3), float("nan"))),
            (["a"], torch.full((1, 3), 100_000.)),
        ]
        for ids, values in invalid:
            with self.subTest(ids=ids, shape=values.shape), self.assertRaises((ValueError, TypeError)):
                writer.add(ids, values)
            self.assertEqual(writer.count, 0)
        writer.add(["a"], torch.ones(1, 4))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            writer.add(["a"], torch.ones(1, 4))
        with self.assertRaisesRegex(ValueError, "dimension changed"):
            writer.add(["b"], torch.ones(1, 3))
        self.assertEqual(writer.count, 1)

    def test_overwrite_checks_every_file_before_deleting_anything(self):
        writer = self.writer(shard_size=1)
        writer.add(["a"], torch.ones(1, 3))
        manifest = writer.finalize({})
        shard = manifest.parent / "shard-00000.pt"
        original = shard.read_bytes()
        unmanaged = manifest.parent / "notes.txt"
        unmanaged.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.writer()
        with self.assertRaisesRegex(ValueError, "unmanaged"):
            self.writer(overwrite=True)
        self.assertTrue(manifest.exists())
        self.assertEqual(shard.read_bytes(), original)
        self.assertEqual(unmanaged.read_text(encoding="utf-8"), "keep")
        unmanaged.unlink()
        self.writer(overwrite=True)
        self.assertEqual(list(manifest.parent.iterdir()), [])

    def test_empty_cache_cannot_be_finalized(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.writer().finalize({})

    def test_encode_normalizes_each_modality_and_rejects_bad_batches(self):
        visual = FixedEncoder([3, 4, 0])
        text = FixedEncoder([0, 0, 2])
        item = OutfitItem(image=torch.ones(3, 4, 4), text="shirt")
        batch = ItemBatch(("a", "b"), ("tops", "tops"), (item, item))
        features = encode_item_batch(batch, visual, text, torch.device("cpu"))
        torch.testing.assert_close(features, torch.tensor([[.6, .8, 0., 0., 0., 1.]]).expand(2, -1))
        self.assertFalse(features.requires_grad)
        bad_batches = [
            ItemBatch((), (), ()),
            ItemBatch(("a",), (), (item,)),
            ItemBatch(("a",), ("tops",), (OutfitItem(embedding=torch.ones(6)),)),
        ]
        for batch in bad_batches:
            with self.assertRaises(ValueError):
                encode_item_batch(batch, visual, text, torch.device("cpu"))

    def test_invalid_encoder_outputs_are_rejected(self):
        batch = ItemBatch(("a",), ("tops",), (
            OutfitItem(image=torch.ones(3, 4, 4), text="shirt"),
        ))
        for values in ([0, 0, 0], [float("nan"), 1, 2], [1, 2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                encode_item_batch(batch, FixedEncoder(values), FixedEncoder([1, 2, 3]),
                                  torch.device("cpu"))

    def test_cli_defaults_and_backend_selection(self):
        config = parse_args([])
        self.assertEqual(config.subset, "nondisjoint")
        self.assertEqual(config.model_name, DEFAULT_MODEL_NAME)
        self.assertIs(config.token, True)
        config = parse_args(["--marqo-fashion-siglip", "--no-token"])
        self.assertEqual(config.model_name, DEFAULT_MARQO_MODEL_NAME)
        self.assertIs(config.token, False)
        visual, text = FixedEncoder([1, 2, 3]), FixedEncoder([3, 2, 1])
        with (
            patch("scripts.precompute_embeddings.MarqoFashionSigLIPVisualEncoder", return_value=visual),
            patch("scripts.precompute_embeddings.MarqoFashionSigLIPTextEncoder", return_value=text) as make_text,
            patch("scripts.precompute_embeddings.build_marqo_fashion_siglip_transform") as make_transform,
        ):
            self.assertEqual(_build_encoders(config, torch.device("cpu")), (visual, text))
            make_text.assert_called_once_with(DEFAULT_MARQO_MODEL_NAME, backbone=visual.backbone)
            _build_image_transform(config)
            make_transform.assert_called_once_with(DEFAULT_MARQO_MODEL_NAME)
        inferred = parse_args(["--model-name", DEFAULT_MARQO_MODEL_NAME])
        self.assertTrue(inferred.uses_marqo_fashion_siglip)

    def test_invalid_config_and_devices_fail_before_processing(self):
        config = parse_args([])
        for changes in ({"batch_size": 0}, {"num_workers": -1}, {"shard_size": 0},
                        {"limit": 0}, {"log_every": 0}, {"model_name": " "},
                        {"output_dtype": "float64"}, {"subset": "missing"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes).validate()
        for name in ("..", "/"):
            with self.assertRaises(ValueError):
                replace(config, model_name=name).target_dir
        with self.assertRaises(ValueError):
            resolve_device("meta")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                resolve_device("cuda")
        with (patch("torch.cuda.is_available", return_value=True),
              patch("torch.cuda.device_count", return_value=1)):
            with self.assertRaisesRegex(ValueError, "index"):
                resolve_device("cuda:1")
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    @patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("unexpected network"))
    @patch("datasets.load_dataset", side_effect=AssertionError("unexpected network"))
    def test_local_parquet_to_manifest_respects_limit_and_uses_frozen_encoders(self, *_):
        image = BytesIO()
        Image.new("RGB", (8, 8), (120, 100, 80)).save(image, "PNG")
        rows = [{"item_id": str(i), "image": {"bytes": image.getvalue(), "path": None}}
                for i in range(5)]
        dataset_root = self.root / "dataset"
        parquet = dataset_root / "data/nondisjoint/train.parquet"
        parquet.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), parquet)
        (dataset_root / "polyvore_item_metadata.json").write_text(
            json.dumps({str(i): {"title": "shirt", "semantic_category": "tops"}
                        for i in range(5)}), encoding="utf-8",
        )
        config = parse_args([
            "--dataset-root", str(dataset_root), "--output-dir", str(self.root / "cache"),
            "--cache-dir", str(self.root / "hf"), "--batch-size", "2", "--shard-size", "2",
            "--limit", "3", "--device", "cpu", "--no-token",
        ])
        visual, text = FixedEncoder([3, 4, 0]), FixedEncoder([0, 0, 2])
        with (
            patch("scripts.precompute_embeddings.build_fashion_clip_transform",
                  return_value=transforms.ToTensor()) as make_transform,
            patch("scripts.precompute_embeddings.FashionCLIPVisualEncoder", return_value=visual) as make_visual,
            patch("scripts.precompute_embeddings.FashionCLIPTextEncoder", return_value=text) as make_text,
        ):
            manifest_path = run(config)
        make_transform.assert_called_once_with(DEFAULT_MODEL_NAME)
        make_visual.assert_called_once_with(DEFAULT_MODEL_NAME, trainable=False)
        make_text.assert_called_once_with(DEFAULT_MODEL_NAME, trainable=False)
        self.assertFalse(visual.training)
        self.assertFalse(text.training)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["count"], 3)
        self.assertEqual(manifest["embedding_dim"], 6)
        self.assertTrue(list((self.root / "hf").rglob("*.arrow")))
        self.assertFalse((self.root / "default-hf").exists())
        self.assertEqual(manifest["encoder"]["aggregation"], "concat_visual_then_text")
        self.assertEqual(len(manifest["model_fingerprint"]), 64)
        shards = [torch.load(manifest_path.parent / s["file"], weights_only=True)
                  for s in manifest["shards"]]
        self.assertEqual(tuple(i for s in shards for i in s["item_ids"]), ("0", "1", "2"))
        torch.testing.assert_close(torch.cat([s["embeddings"] for s in shards]),
                                   torch.tensor([[.6, .8, 0., 0., 0., 1.]]).expand(3, -1))


if __name__ == "__main__":
    unittest.main()
