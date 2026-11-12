"""Check encoder behavior using small local models without downloading weights."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from transformers import (
    BatchEncoding,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from model import (
    FashionCLIPTextEncoder,
    FashionCLIPVisualEncoder,
    MarqoFashionSigLIPTextEncoder,
    MarqoFashionSigLIPVisualEncoder,
    MLP,
    MultimodalEncoder,
    OutfitSeedEncoder,
)


class MarqoBackbone(nn.Module):
    """Small stand-in for the remote Marqo feature API."""

    def __init__(self, output_dim=8, context_length=6):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(1, 9, dtype=torch.float32))
        self.model = SimpleNamespace(text=SimpleNamespace(
            output_dim=output_dim, context_length=context_length,
        ))

    def get_image_features(self, pixel_values, *, normalize=False):
        assert normalize is False
        return pixel_values.mean(dim=(1, 2, 3)).unsqueeze(1) * self.weight

    def get_text_features(self, input_ids, *, normalize=False):
        assert normalize is False
        return input_ids.float().sum(dim=1).unsqueeze(1) * self.weight


class EncoderTests(unittest.TestCase):
    def setUp(self):
        vision = CLIPVisionModelWithProjection(CLIPVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, image_size=8, patch_size=4, projection_dim=8,
        ))
        text = CLIPTextModelWithProjection(CLIPTextConfig(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            max_position_embeddings=6, projection_dim=8,
        ))
        self.tokenizer = Mock(return_value=BatchEncoding({
            "input_ids": torch.ones(2, 6, dtype=torch.long),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
        }))
        self.processor = Mock(return_value={
            "input_ids": torch.ones(2, 6, dtype=torch.long),
        })
        self.marqo = MarqoBackbone()
        patches = [
            patch("transformers.CLIPVisionModelWithProjection.from_pretrained", return_value=vision),
            patch("transformers.CLIPTextModelWithProjection.from_pretrained", return_value=text),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tokenizer),
            patch("transformers.models.auto.modeling_auto.AutoModel.from_pretrained", return_value=self.marqo),
            patch("transformers.models.auto.processing_auto.AutoProcessor.from_pretrained", return_value=self.processor),
        ]
        self.loads = [item.start() for item in patches]
        for item in patches:
            self.addCleanup(item.stop)

    def test_fashion_clip_encodes_both_modalities_and_allows_gradients(self):
        visual = FashionCLIPVisualEncoder()
        text = FashionCLIPTextEncoder()
        image_features = visual(torch.ones(2, 3, 8, 8))
        text_features = text(["red shirt", "blue trousers"])
        self.assertEqual(image_features.shape, (2, visual.output_dim))
        self.assertEqual(text_features.shape, (2, text.output_dim))
        self.assertEqual(visual.output_dim, 8)
        self.assertEqual(text.output_dim, 8)
        torch.testing.assert_close(image_features.norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(text_features.norm(dim=-1), torch.ones(2))
        mlp = MLP(16, 3, hidden_dims=[12, 6])
        encoder = MultimodalEncoder(visual, text, mlp)
        embeddings = encoder(torch.ones(2, 3, 8, 8), ["red shirt", "blue trousers"])
        self.assertIs(encoder.mlp, mlp)
        self.assertEqual(encoder.output_dim, 3)
        self.assertEqual(embeddings.shape, (2, 3))
        torch.testing.assert_close(embeddings, mlp(torch.cat((image_features, text_features), dim=-1)))
        embeddings.sum().backward()
        self.assertIsNotNone(visual.backbone.visual_projection.weight.grad)
        self.assertIsNotNone(text.backbone.text_projection.weight.grad)
        self.assertTrue(all(p.grad is not None for p in mlp.parameters()))
        self.tokenizer.assert_called_with(
            ["red shirt", "blue trousers"], padding=True,
            truncation=True, max_length=6, return_tensors="pt",
        )
        self.loads[0].assert_called_once_with("patrickjohncyh/fashion-clip")
        self.loads[1].assert_called_once_with("patrickjohncyh/fashion-clip")

    def test_frozen_fashion_clip_stays_in_evaluation_mode(self):
        for encoder in (
            FashionCLIPVisualEncoder(trainable=False),
            FashionCLIPTextEncoder(trainable=False),
        ):
            with self.subTest(encoder=type(encoder).__name__):
                self.assertIs(encoder.train(), encoder)
                self.assertFalse(encoder.backbone.training)
                self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))

    def test_marqo_encoders_share_frozen_backbone_and_return_normalized_features(self):
        visual = MarqoFashionSigLIPVisualEncoder()
        text = MarqoFashionSigLIPTextEncoder(backbone=visual.backbone)
        self.assertIs(text.backbone, visual.backbone)
        self.assertEqual(visual.output_dim, 8)
        self.assertEqual(text.output_dim, 8)
        expected = (self.marqo.weight / self.marqo.weight.norm()).expand(2, -1)
        torch.testing.assert_close(
            visual(torch.ones(2, 3, 8, 8)), expected,
        )
        torch.testing.assert_close(
            text(["shirt", "trousers"]), expected,
        )
        self.loads[3].assert_called_once_with(
            "Marqo/marqo-fashionSigLIP", trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        self.loads[4].assert_called_once_with(
            "Marqo/marqo-fashionSigLIP", trust_remote_code=True,
        )
        self.processor.assert_called_once_with(
            text=["shirt", "trousers"], padding="max_length",
            truncation=True, max_length=6, return_tensors="pt",
        )
        for encoder in (visual, text):
            encoder.train()
            self.assertFalse(encoder.backbone.training)
            self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))

    def test_marqo_multimodal_head_trains_on_concatenated_frozen_features(self):
        visual = MarqoFashionSigLIPVisualEncoder()
        mlp = MLP(16, 4, hidden_dims=[6])
        encoder = MultimodalEncoder(
            visual, MarqoFashionSigLIPTextEncoder(backbone=visual.backbone), mlp,
        ).train()
        embeddings = encoder(torch.ones(2, 3, 8, 8), ["shirt", "trousers"])
        expected = (self.marqo.weight / self.marqo.weight.norm()).repeat(2, 2)
        self.assertEqual(encoder.output_dim, 4)
        torch.testing.assert_close(embeddings, mlp(expected))
        embeddings.square().sum().backward()
        self.assertTrue(all(p.grad is not None for p in mlp.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in mlp.parameters()))
        self.assertTrue(all(p.grad is None for p in visual.parameters()))
        self.assertFalse(visual.backbone.training)

    def test_multimodal_head_trains_with_frozen_fashion_clip(self):
        visual = FashionCLIPVisualEncoder(trainable=False)
        text = FashionCLIPTextEncoder(trainable=False)
        mlp = MLP(16, 4, hidden_dims=[6])
        encoder = MultimodalEncoder(visual, text, mlp).train()
        encoder(torch.ones(2, 3, 8, 8), ["shirt", "trousers"]).sum().backward()
        self.assertTrue(all(p.grad is not None for p in mlp.parameters()))
        self.assertTrue(all(p.grad is None for p in visual.parameters()))
        self.assertTrue(all(p.grad is None for p in text.parameters()))
        self.assertFalse(visual.backbone.training)
        self.assertFalse(text.backbone.training)

    def test_multimodal_encoder_rejects_incompatible_mlp(self):
        with self.assertRaisesRegex(ValueError, "input_dim"):
            MultimodalEncoder(FashionCLIPVisualEncoder(), FashionCLIPTextEncoder(), MLP(8, 3))

    def test_seed_reuses_the_multimodal_visual_encoder_and_applies_mlp(self):
        multimodal = MultimodalEncoder(
            FashionCLIPVisualEncoder(trainable=False), FashionCLIPTextEncoder(trainable=False),
            MLP(16, 4),
        ).eval()
        mlp = MLP(multimodal.visual_encoder.output_dim, 3, hidden_dims=[6, 4])
        seed = OutfitSeedEncoder(multimodal.visual_encoder, mlp).eval()
        images = torch.ones(2, 3, 8, 8)
        self.assertIs(seed.visual_encoder, multimodal.visual_encoder)
        self.assertIs(seed.mlp, mlp)
        self.assertIsNot(seed.mlp, multimodal.mlp)
        self.assertEqual(seed.output_dim, 3)
        self.assertEqual(seed(images).shape, (2, 3))
        torch.testing.assert_close(seed(images), mlp(multimodal.visual_encoder(images)))

    def test_single_seed_image_trains_mlp_with_frozen_visual_encoders(self):
        for visual in (FashionCLIPVisualEncoder(trainable=False), MarqoFashionSigLIPVisualEncoder()):
            with self.subTest(encoder=type(visual).__name__):
                seed = OutfitSeedEncoder(visual, MLP(visual.output_dim, 3, hidden_dims=[5])).train()
                output = seed(torch.ones(3, 8, 8))
                self.assertEqual(output.shape, (1, 3))
                output.square().sum().backward()
                self.assertTrue(all(p.grad is not None for p in seed.mlp.parameters()))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in seed.mlp.parameters()))
                self.assertTrue(all(p.grad is None for p in visual.parameters()))
                self.assertFalse(visual.backbone.training)

    def test_seed_preserves_gradients_for_trainable_visual_encoder(self):
        visual = FashionCLIPVisualEncoder(trainable=True)
        seed = OutfitSeedEncoder(visual, MLP(visual.output_dim, 3))
        seed(torch.ones(1, 3, 8, 8)).sum().backward()
        self.assertIsNotNone(visual.backbone.visual_projection.weight.grad)
        self.assertTrue(torch.isfinite(visual.backbone.visual_projection.weight.grad).all())

    def test_seed_rejects_incompatible_mlp(self):
        visual = FashionCLIPVisualEncoder()
        with self.assertRaisesRegex(ValueError, "input_dim"):
            OutfitSeedEncoder(visual, MLP(visual.output_dim * 2, 3))

    def test_seed_rejects_invalid_images(self):
        visual = FashionCLIPVisualEncoder()
        seed = OutfitSeedEncoder(visual, MLP(visual.output_dim, 3))
        for images in (torch.ones(8, 8), torch.ones(1, 8, 8), torch.ones(0, 3, 8, 8)):
            with self.subTest(shape=images.shape), self.assertRaises(ValueError):
                seed(images)

    def test_zero_visual_features_remain_finite_after_normalization(self):
        visual = MarqoFashionSigLIPVisualEncoder()
        encoder = MultimodalEncoder(
            visual, MarqoFashionSigLIPTextEncoder(backbone=visual.backbone),
            MLP(16, 4),
        )
        images = torch.zeros(2, 3, 8, 8)
        descriptions = ["shirt", "trousers"]
        visual_features = visual(images)
        text_features = encoder.text_encoder(descriptions)
        embeddings = encoder(images, descriptions)
        self.assertTrue(torch.isfinite(embeddings).all())
        torch.testing.assert_close(visual_features, torch.zeros(2, 8))
        torch.testing.assert_close(text_features.norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(embeddings, encoder.mlp(torch.cat((visual_features, text_features), dim=-1)))

    def test_multimodal_encoder_rejects_unpaired_items(self):
        encoder = MultimodalEncoder(FashionCLIPVisualEncoder(), FashionCLIPTextEncoder(), MLP(16, 4))
        for descriptions in ([], ["shirt"], ["shirt", "trousers", "shoes"]):
            with self.subTest(descriptions=descriptions):
                with self.assertRaisesRegex(ValueError, "same number of items"):
                    encoder(torch.ones(2, 3, 8, 8), descriptions)
        with self.assertRaisesRegex(ValueError, "shape"):
            encoder(torch.ones(3, 8, 8), ["shirt"])
        self.tokenizer.assert_not_called()

    def test_marqo_text_can_load_its_own_backbone(self):
        text = MarqoFashionSigLIPTextEncoder()
        self.assertIs(text.backbone, self.marqo)
        self.loads[3].assert_called_once()

    def test_invalid_image_batches_are_rejected(self):
        invalid = [torch.ones(3, 8, 8), torch.ones(2, 1, 8, 8), torch.ones(0, 3, 8, 8)]
        for encoder in (FashionCLIPVisualEncoder(), MarqoFashionSigLIPVisualEncoder()):
            for images in invalid:
                with self.subTest(encoder=type(encoder).__name__, shape=images.shape):
                    with self.assertRaises(ValueError):
                        encoder(images)

    def test_invalid_descriptions_are_rejected(self):
        for encoder in (FashionCLIPTextEncoder(), MarqoFashionSigLIPTextEncoder()):
            for descriptions in ([], [" "], ["shirt", None]):
                with self.subTest(encoder=type(encoder).__name__, descriptions=descriptions):
                    with self.assertRaises(ValueError):
                        encoder(descriptions)
        self.tokenizer.assert_not_called()
        self.processor.assert_not_called()

    def test_invalid_fashion_clip_configuration_is_rejected(self):
        self.loads[0].return_value.config.projection_dim = 0
        with self.assertRaisesRegex(ValueError, "projection_dim"):
            FashionCLIPVisualEncoder()
        self.loads[1].return_value.config = SimpleNamespace(
            projection_dim=8, max_position_embeddings=False,
        )
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            FashionCLIPTextEncoder()

    def test_invalid_marqo_configuration_is_rejected(self):
        self.marqo.model.text.output_dim = False
        with self.assertRaisesRegex(ValueError, "output_dim"):
            MarqoFashionSigLIPVisualEncoder()
        self.marqo.model.text.output_dim = 8
        self.marqo.model.text.context_length = 0
        with self.assertRaisesRegex(ValueError, "context_length"):
            MarqoFashionSigLIPTextEncoder()


if __name__ == "__main__":
    unittest.main()
