"""Check composition and gradients using small, local stand-ins for the encoders."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from model import (
    CategoryQueryEmbeddings, LLM, MLP, OUTFIT_CATEGORIES, OutfitSeedEncoder,
    SeedToOutfitModel, SeedToOutfitOutput, VisualEncoder,
)


class TinyLLM(LLM):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8).requires_grad_(False)
        self.received = None
        self.states = None

    @property
    def output_dim(self):
        return 8

    def forward(self, prompts, category_embeddings, *, seed_embeddings=None):
        self.received = (prompts, seed_embeddings)
        weights = self.projection.weight
        inputs = category_embeddings.to(weights)
        if seed_embeddings is not None:
            inputs = inputs + seed_embeddings.to(weights).unsqueeze(1)
        self.states = self.projection(inputs)
        return self.states


class TinyVisualEncoder(VisualEncoder):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4).requires_grad_(False)

    @property
    def output_dim(self):
        return 4

    def forward(self, images):
        return F.normalize(self.projection(images.mean(dim=(-2, -1))), dim=-1)


class SeedToOutfitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.llm = TinyLLM()
        self.queries = CategoryQueryEmbeddings(8)
        self.seed_encoder = OutfitSeedEncoder(TinyVisualEncoder(), MLP(4, 8))
        self.options = dict(
            llm=self.llm, category_queries=self.queries, seed_encoder=self.seed_encoder,
            retrieval_heads=[MLP(8, 6) for _ in OUTFIT_CATEGORIES],
            category_heads=[MLP(8, 1) for _ in OUTFIT_CATEGORIES],
            fine_grained_heads=[MLP(8, 1 + i % 3) for i in range(11)],
        )
        self.model = SeedToOutfitModel(**self.options)

    def test_image_forward_preserves_category_order_and_raw_head_outputs(self):
        images = torch.randn(2, 3, 4, 4)
        output = self.model(["casual", "formal"], seed_images=images)
        self.assertIsInstance(output, SeedToOutfitOutput)
        self.assertEqual(self.model.categories, OUTFIT_CATEGORIES)
        self.assertEqual(tuple(output.retrieval_embeddings.shape), (2, 11, 6))
        self.assertEqual(tuple(output.category_logits.shape), (2, 11))
        torch.testing.assert_close(self.llm.received[1], self.seed_encoder(images))
        self.assertEqual(len(output.fine_grained_logits), 11)
        for i, (retrieval, category, fine) in enumerate(zip(
            self.model.retrieval_heads, self.model.category_heads, self.model.fine_grained_heads,
        )):
            torch.testing.assert_close(output.retrieval_embeddings[:, i], retrieval(self.llm.states[:, i]))
            torch.testing.assert_close(output.category_logits[:, i], category(self.llm.states[:, i]).squeeze(-1))
            torch.testing.assert_close(output.fine_grained_logits[i], fine(self.llm.states[:, i]))
            self.assertEqual(tuple(output.fine_grained_logits[i].shape), (2, 1 + i % 3))

    def test_each_category_uses_its_own_presence_head(self):
        prompts = ["casual", "formal"]
        seeds = ["shirt", "shoes"]
        before = self.model(prompts, seed_texts=seeds)
        with torch.no_grad():
            self.model.category_heads[4].layers[-1].bias.add_(5)
        after = self.model(prompts, seed_texts=seeds)
        difference = after.category_logits - before.category_logits
        expected = torch.zeros(2, 11)
        expected[:, 4] = 5
        torch.testing.assert_close(difference, expected)
        torch.testing.assert_close(after.retrieval_embeddings, before.retrieval_embeddings)
        for actual, original in zip(after.fine_grained_logits, before.fine_grained_logits):
            torch.testing.assert_close(actual, original)

    def test_single_image_and_batch_have_the_same_result(self):
        image = torch.randn(3, 4, 4)
        single = self.model("casual", seed_images=image)
        batched = self.model(["casual"], seed_images=image.unsqueeze(0))
        self.assertEqual(tuple(single.category_logits.shape), (1, 11))
        torch.testing.assert_close(single.retrieval_embeddings, batched.retrieval_embeddings)

    def test_text_seed_needs_no_visual_encoder(self):
        self.options["seed_encoder"] = None
        model = SeedToOutfitModel(**self.options)
        with patch.object(self.seed_encoder, "forward", side_effect=AssertionError("unexpected image")):
            output = model(["casual", "formal"], seed_texts=["red shirt", "black shoes"])
        self.assertEqual(self.llm.received, ([
            "casual\nSeed item: red shirt", "formal\nSeed item: black shoes",
        ], None))
        self.assertEqual(tuple(output.category_logits.shape), (2, 11))
        with self.assertRaisesRegex(ValueError, "seed_encoder"):
            model("casual", seed_images=torch.randn(3, 4, 4))

    def test_gradients_reach_all_trainable_parts_through_frozen_backbones(self):
        self.llm.double()  # Heads and seed encoder stay FP32, as with a BF16 LLM.
        self.model.train()
        output = self.model("casual", seed_images=torch.randn(3, 4, 4))
        self.assertEqual(output.retrieval_embeddings.dtype, torch.float32)
        self.assertEqual(output.category_logits.dtype, torch.float32)
        scalar = (
            output.retrieval_embeddings.square().sum() + output.category_logits.sum()
            + sum(logits.square().sum() for logits in output.fine_grained_logits)
        )
        scalar.backward()
        for module in (
            self.queries, self.seed_encoder.mlp, *self.model.category_heads,
            *self.model.retrieval_heads, *self.model.fine_grained_heads,
        ):
            with self.subTest(module=module):
                self.assertTrue(all(parameter.grad is not None for parameter in module.parameters()))
                self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in module.parameters()), 0)
        for module in (self.llm, self.seed_encoder.visual_encoder):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_requires_one_seed_per_prompt_and_exactly_one_modality(self):
        cases = (
            {},
            dict(seed_texts="shirt", seed_images=torch.zeros(3, 4, 4)),
            dict(seed_texts=["shirt", "shoes"]),
            dict(seed_texts=""),
            dict(seed_images=torch.zeros(2, 3, 4, 4)),
            dict(seed_images=torch.zeros(4, 4)),
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model("casual", **options)

    def test_rejects_inconsistent_component_dimensions(self):
        cases = (
            dict(category_queries=CategoryQueryEmbeddings(7)),
            dict(seed_encoder=OutfitSeedEncoder(TinyVisualEncoder(), MLP(4, 7))),
            dict(retrieval_heads=[MLP(8, 6)] * 10),
            dict(retrieval_heads=[MLP(7, 6)] * 11),
            dict(retrieval_heads=[MLP(8, 6)] * 10 + [MLP(8, 5)]),
            dict(fine_grained_heads=[MLP(8, 3)] * 10),
            dict(fine_grained_heads=[MLP(7, 3)] * 11),
            dict(category_heads=[MLP(8, 1)] * 10),
            dict(category_heads=[MLP(8, 1)] * 12),
            dict(category_heads=[MLP(8, 2)] * 11),
            dict(category_heads=[MLP(7, 1)] * 11),
        )
        for replacement in cases:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                SeedToOutfitModel(**(self.options | replacement))
        self.llm.categories = tuple(reversed(OUTFIT_CATEGORIES))
        with self.assertRaisesRegex(ValueError, "category order"):
            SeedToOutfitModel(**self.options)


if __name__ == "__main__":
    unittest.main()
