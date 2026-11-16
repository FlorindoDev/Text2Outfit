"""The public model builder owns dimensions and obeys each MLP configuration."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from model import DEFAULT_MODEL_CONFIG, OutfitCompatibility, build_model
from test_training import Batcher, Scorer, Visual, Textual, make_llm, make_example
from training import collate_seed_outfits
from training.config import TrainingConfig
from training.train_seed_to_outfit import build_parser


class ModelAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "cp.pt"
        self.checkpoint.write_bytes(b"mocked CP loader")
        self.config = replace(
            DEFAULT_MODEL_CONFIG, cp_repo=self.root, cp_checkpoint=self.checkpoint,
            category_embedding_dim=16, seed_hidden_dims=(7, 5), item_hidden_dims=(10,),
            retrieval_hidden_dims=(9,), category_hidden_dims=(),
            fine_grained_hidden_dims=(8, 4),
            lora_rank=2, lora_alpha=4, lora_dropout=0,
        )

    @staticmethod
    def cp_loader(repo, checkpoint, width):
        return OutfitCompatibility(Scorer(width), Batcher(width))

    def build(self, cached, *, encoder="fashionclip", retrieval_dim=None):
        llm = make_llm(enable_lora=False)
        siglip_visual = Visual()
        siglip_visual.backbone = None
        with (
            patch("model.outfit.build.resolve_cp_assets", return_value=(self.root, self.checkpoint)),
            patch("model.outfit.build.Qwen3LLM", return_value=llm),
            patch("model.outfit.build.OutfitCompatibility.from_checkpoint", side_effect=self.cp_loader),
            patch("model.outfit.build.FashionCLIPVisualEncoder", return_value=Visual()) as visual,
            patch("model.outfit.build.FashionCLIPTextEncoder", return_value=Textual()) as textual,
            patch("model.outfit.build.MarqoFashionSigLIPVisualEncoder", return_value=siglip_visual) as siglip,
            patch("model.outfit.build.MarqoFashionSigLIPTextEncoder", return_value=Textual()),
        ):
            config = replace(self.config, encoder=encoder, retrieval_dim=retrieval_dim)
            model = build_model(tuple(("a", "b") for _ in range(11)), config,
                                precomputed_modality_dim=4 if cached else None)
        self.assertEqual(visual.call_count, int(not cached and encoder == "fashionclip"))
        self.assertEqual(textual.call_count, visual.call_count)
        self.assertEqual(siglip.call_count, int(not cached and encoder == "siglip"))
        return model

    def test_raw_and_cached_models_have_independently_configured_mlps(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                model = self.build(cached)
                self.assertEqual(model.generator.category_queries.output_dim, self.config.category_embedding_dim)
                architecture = model.architecture()
                self.assertEqual(len(architecture["mlps"]), 35)
                self.assertEqual(model.config.retrieval_dim, 1024)
                self.assertEqual(model.compatibility.input_dim, 1024)
                self.assertFalse(any(p.requires_grad for p in model.compatibility.parameters()))
                expected = {
                    "seed": [4, 7, 5, 16], "item": [8, 10, 1024],
                    "retrieval.shoes": [16, 9, 1024], "presence.shoes": [16, 1],
                    "subcategory.shoes": [16, 8, 4, 2],
                }
                for name, dimensions in expected.items():
                    self.assertEqual(architecture["mlps"][name]["dimensions"], dimensions)
                    parameters = sum((left + 1) * right for left, right in zip(dimensions, dimensions[1:]))
                    self.assertEqual(architecture["mlps"][name]["parameters"], parameters)
                self.assertIn("4 -> 7 -> 5 -> 16", model.architecture_summary())
                self.assertEqual(model.generator.llm.lora_settings["rank"], 2)
                result = model("outfit", seed_features=torch.randn(1, 4))
                self.assertEqual(tuple(result.retrieval_embeddings.shape), (1, 11, 1024))
                losses = model.compute_losses(collate_seed_outfits([make_example()]))
                losses["total"].backward()
                self.assertTrue(all(p.grad is not None for p in model.item_encoder.mlp.parameters()))

    def test_siglip_defaults_to_1536_for_predictions_items_and_cp(self):
        args = build_parser().parse_args(["--encoder", "siglip"])
        for cached in (False, True):
            with self.subTest(cached=cached):
                model = self.build(cached, encoder=args.encoder, retrieval_dim=args.retrieval_dim)
                self.assertEqual(model.config.retrieval_dim, 1536)
                self.assertEqual(model.item_encoder.output_dim, 1536)
                self.assertEqual(model.compatibility.input_dim, 1536)
                output = model("outfit", seed_features=torch.randn(1, 4))
                self.assertEqual(tuple(output.retrieval_embeddings.shape), (1, 11, 1536))
                losses = model.compute_losses(collate_seed_outfits([make_example(cached=cached)]))
                losses["total"].backward()
                self.assertTrue(torch.isfinite(losses["total"]))
                self.assertTrue(all(p.grad is not None for p in model.item_encoder.mlp.parameters()))

    def test_explicit_retrieval_width_overrides_encoder_default(self):
        for encoder in ("fashionclip", "siglip"):
            with self.subTest(encoder=encoder):
                args = build_parser().parse_args(["--encoder", encoder, "--retrieval-dim", "8"])
                model = self.build(True, encoder=args.encoder, retrieval_dim=args.retrieval_dim)
                output = model("outfit", seed_features=torch.randn(1, 4))
                self.assertEqual(tuple(output.retrieval_embeddings.shape), (1, 11, 8))
                self.assertEqual(model.item_encoder.output_dim, 8)
                self.assertEqual(model.compatibility.input_dim, 8)

    def test_invalid_configuration_fails_before_loading_any_backbone(self):
        invalid = (
            replace(self.config, category_embedding_dim=0),
            replace(self.config, category_embedding_dim=-1),
            replace(self.config, category_embedding_dim=True),
            replace(self.config, retrieval_dim=0),
            replace(self.config, retrieval_dim=-1),
            replace(self.config, retrieval_dim=True),
            replace(self.config, seed_hidden_dims=(0,)),
            replace(self.config, loss_weights=(0., 0., 0.)),
            replace(self.config, lora_target_modules=()),
            replace(self.config, llm_dtype="invalid"),
            replace(self.config, llm_trainable=True, lora_enabled=True),
        )
        for config in invalid:
            with self.subTest(config=config), patch("model.outfit.build.Qwen3LLM") as loader:
                with self.assertRaises(ValueError):
                    build_model(tuple(("a", "b") for _ in range(11)), config, precomputed_modality_dim=4)
                loader.assert_not_called()
        self.assertEqual(TrainingConfig.__module__, "training.config")

    def test_category_query_width_must_match_llm_before_enabling_lora(self):
        llm = make_llm(enable_lora=False)
        with (
            patch("model.outfit.build.resolve_cp_assets", return_value=(self.root, self.checkpoint)),
            patch("model.outfit.build.Qwen3LLM", return_value=llm),
            patch.object(llm, "enable_lora") as enable_lora,
            patch("model.outfit.build.OutfitCompatibility.from_checkpoint") as cp_loader,
        ):
            with self.assertRaisesRegex(ValueError, r"category_embedding_dim \(8\).*hidden size \(16\)"):
                build_model(tuple(("a", "b") for _ in range(11)),
                            replace(self.config, category_embedding_dim=8), precomputed_modality_dim=4)
            enable_lora.assert_not_called()
            cp_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
