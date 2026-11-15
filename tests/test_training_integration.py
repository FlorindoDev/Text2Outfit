"""Integration with the user's OutfitTransformer source, without downloading weights."""

import json
import hashlib
import tempfile
import unittest
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from model.compatibility import OutfitCompatibility, load_outfit_transformer

CP_REPO = Path(__file__).resolve().parents[2] / "OutfitTransformer"


@unittest.skipUnless((CP_REPO / "model/cp/transformer.py").is_file(), "OutfitTransformer source is not installed next to this project")
class OutfitTransformerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = load_outfit_transformer(CP_REPO)
        self.config = self.package.TransformerConfig(
            modality_embedding_dim=4, layers=1, attention_heads=2,
            feedforward_dim=16, dropout=0, max_items=16,
        )

    def checkpoint(self, legacy=False):
        cp = self.package.CompatibilityTransformer(self.config)
        batcher = self.package.OutfitEmbeddingBatcher(self.config)
        context = None
        if legacy:
            builder = import_module(f"{self.package.__name__}.common.transformer_encoder")
            context = builder.build_transformer_encoder(self.config)
            context.norm = nn.LayerNorm(8)
            cp.encoder.norm = nn.LayerNorm(8)
        reference = OutfitCompatibility(cp, batcher, context_encoder=context, legacy_tokens=legacy)
        state = {f"cp.{key}": value for key, value in cp.state_dict().items()}
        state["common.padding_embedding"] = batcher.padding_embedding
        if context is not None:
            state.update({f"common.encoder.{key}": value for key, value in context.state_dict().items()})
        path = self.root / ("legacy.pt" if legacy else "current.pt")
        torch.save(dict(run_config=dict(model=asdict(self.config)), model_state_dict=state), path)
        return path, reference

    def test_current_and_legacy_cp_preserve_weights_scores_and_gradients(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                path, reference = self.checkpoint(legacy)
                scorer = OutfitCompatibility.from_checkpoint(CP_REPO, path, 8)
                self.assertEqual(scorer.source["checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                scorer.train()
                self.assertFalse(scorer.cp.training)
                x = torch.randn(2, 11, 8, requires_grad=True)
                mask = torch.zeros(2,11,dtype=torch.bool)
                mask[0,[0,1,4]], mask[1,[0,4]] = True, True
                torch.testing.assert_close(scorer(x, mask), reference(x, mask))
                (-scorer(x, mask).mean()).backward()
                self.assertGreater(x.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None for p in scorer.cp.parameters()))
                self.assertFalse(any(p.requires_grad for p in scorer.parameters()))
                if legacy:
                    self.assertTrue(all(p.grad is None for p in scorer.context_encoder.parameters()))

    def test_unrecognized_cp_weights_are_rejected(self):
        path, _ = self.checkpoint()
        payload = torch.load(path, weights_only=True)
        payload["model_state_dict"]["unknown.weights"] = torch.zeros(1)
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "refusing to discard"):
            OutfitCompatibility.from_checkpoint(CP_REPO, path, 8)

    def test_incompatible_cp_input_width_is_rejected(self):
        path, _ = self.checkpoint()
        with self.assertRaisesRegex(ValueError, r"retrieval_dim \(6\).*width \(8\)"):
            OutfitCompatibility.from_checkpoint(CP_REPO, path, 6)
        scorer = OutfitCompatibility.from_checkpoint(CP_REPO, path, 8)
        with self.assertRaisesRegex(ValueError, "width 8"):
            scorer(torch.randn(1, 11, 6), torch.ones(1, 11, dtype=torch.bool))

    def test_cli_from_local_parquet_cache_and_tiny_checkpoints(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from transformers import Qwen3Model
        from test_outfit_data import OutfitDataTests
        from test_training import make_llm
        from training.train_seed_to_outfit import main

        fixture = OutfitDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        parquet = fixture.root / "data/nondisjoint"
        parquet.mkdir(parents=True)
        table = pa.table({"item_id": list(fixture.metadata), "image": [b"not decoded"] * len(fixture.metadata)})
        cache_root = self.root / "features"
        features = torch.randn(len(fixture.metadata), 8)
        for split, stem in (("train", "train"), ("validation", "valid")):
            pq.write_table(table, parquet / f"{split}.parquet")
            (fixture.split_root / f"{stem}.json").write_text(fixture.outfits_path.read_text())
            (fixture.split_root / f"compatibility_{stem}.txt").write_text("1 a_1 a_2 a_3 a_4\n0 a_1 b_2\n")
            directory = cache_root / "nondisjoint" / split
            directory.mkdir(parents=True)
            torch.save(dict(item_ids=list(fixture.metadata), embeddings=features), directory / "shard-00000.pt")
            manifest = dict(
                embedding_dim=8, count=len(fixture.metadata), subset="nondisjoint", split=split,
                model_fingerprint="tiny-test-encoder", shards=[dict(file="shard-00000.pt")],
                encoder=dict(model_name="patrickjohncyh/fashion-clip", modality_dim=4,
                             aggregation="concat_visual_then_text", normalization="l2_per_modality"),
            )
            (directory / "manifest.json").write_text(json.dumps(manifest))
        tiny = make_llm()
        llm_path = self.root / "qwen"
        Qwen3Model(tiny.backbone.config).save_pretrained(llm_path)
        tiny.tokenizer.save_pretrained(llm_path)
        cp_path, _ = self.checkpoint()
        output = self.root / "output"
        with (
            patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("unexpected download")),
            patch("model.llm.download.snapshot_download", side_effect=AssertionError("unexpected download")),
            patch("data.polyvore.catalog._decode_image", side_effect=AssertionError("unexpected image decode")),
            patch("transformers.CLIPVisionModelWithProjection.from_pretrained", side_effect=AssertionError("unexpected vision load")),
        ):
            main([
                "--dataset-root", str(fixture.root), "--embedding-root", str(cache_root),
                "--dataset-cache", str(self.root / "datasets-cache"),
                "--llm", str(llm_path), "--cp-repo", str(CP_REPO), "--cp-checkpoint", str(cp_path),
                "--category-embedding-dim", "16",
                "--output-dir", str(output), "--epochs", "1", "--device", "cpu",
                "--retrieval-dim", "8", "--hidden-dim", "8", "--lora-rank", "2", "--lora-alpha", "4",
                "--negatives-per-item", "2",
            ])
        self.assertTrue((output / "best.pt").is_file())
        self.assertEqual(len(list((output / "plots").glob("*.png"))), 7)
        state = torch.load(output / "best.pt", weights_only=True)
        self.assertEqual(state["training_history"]["epochs"], [1])
        self.assertEqual(state["model_config"]["lora"]["rank"], 2)


if __name__ == "__main__":
    unittest.main()
