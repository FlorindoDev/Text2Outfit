"""Test the real Qwen3 forward on a tiny random backbone, without downloads."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from model import (
    CategoryQueryEmbeddings, LLM, MLP, OUTFIT_CATEGORIES, Qwen3LLM, SeedToOutfitModel,
)


class Qwen3LLMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        vocabulary = {
            "[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
            "red": 3, "blue": 4, "shirt": 5, "shoes": 6, "casual": 7,
        }
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            unk_token="[UNK]",
            eos_token="[EOS]",
            padding_side="left",
        )
        self.backbone = Qwen3Model(Qwen3Config(
            vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
            pad_token_id=0, eos_token_id=2, sliding_window=None,
        ))
        self.tokenizer_loader = patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=self.tokenizer,
        ).start()
        self.backbone_loader = patch(
            "transformers.Qwen3Model.from_pretrained", return_value=self.backbone,
        ).start()
        self.path_resolver = patch(
            "model.llm.download.resolve_model_path", return_value=Path("local-qwen3"),
        ).start()
        self.addCleanup(patch.stopall)
        self.llm = Qwen3LLM()
        self.queries = CategoryQueryEmbeddings(embedding_dim=self.llm.output_dim)

    def test_returns_only_raw_category_states_in_shared_order(self):
        states = self.llm("red shirt", self.queries())
        self.assertIsInstance(self.llm, LLM)
        self.assertIsInstance(states, torch.Tensor)
        self.assertEqual(tuple(states.shape), (1, 11, 16))
        self.assertEqual(self.llm.categories, OUTFIT_CATEGORIES)
        self.assertTrue(bool(torch.isfinite(states).all()))
        self.assertTrue(bool((states.norm(dim=-1) > 2).all()))
        self.assertEqual(states.untyped_storage().nbytes(), states.numel() * states.element_size())
        self.backbone_loader.assert_called_once_with(
            "local-qwen3", torch_dtype="auto", attn_implementation="sdpa", local_files_only=True,
        )
        self.tokenizer_loader.assert_called_once_with(
            "local-qwen3", padding_side="left", local_files_only=True,
        )
        self.path_resolver.assert_called_once_with("Qwen/Qwen3-Embedding-8B", cache_dir=None)

    def test_unbatched_queries_keep_a_batch_dimension(self):
        queries = self.queries()
        self.assertTrue(torch.allclose(
            self.llm("red shirt", queries[0]), self.llm(["red shirt"], queries),
        ))

    def test_padded_batches_match_individual_prompts(self):
        texts = ["red shirt shoes", "blue"]
        queries = self.queries(batch_size=2)
        for padding_side in ("left", "right"):
            with self.subTest(padding_side=padding_side):
                self.tokenizer.padding_side = padding_side
                batch = self.llm(texts, queries)
                for index, text in enumerate(texts):
                    individual = self.llm(text, queries[index:index + 1])
                    self.assertTrue(torch.allclose(batch[index:index + 1], individual, atol=1e-5))

    def test_attention_keeps_causal_text_and_mutually_visible_queries(self):
        received = {}

        def capture(module, args, kwargs):
            received.update(kwargs)

        handle = self.backbone.register_forward_pre_hook(capture, with_kwargs=True)
        self.addCleanup(handle.remove)
        self.llm(["red shirt shoes", "blue"], self.queries(batch_size=2))
        mask = received["attention_mask"]
        blocked = torch.finfo(mask.dtype).min
        self.assertEqual(tuple(mask.shape), (2, 1, 14, 14))
        self.assertEqual(mask[0, 0, 0, 1].item(), blocked)
        self.assertTrue(bool((mask[0, 0, :3, 3:] == blocked).all()))
        self.assertTrue(bool((mask[0, 0, 3:, :] == 0).all()))
        self.assertTrue(bool((mask[1, 0, :, :2] == blocked).all()))
        self.assertEqual(received["position_ids"][1].tolist(), [0, 0, *range(12)])
        self.assertFalse(received["use_cache"])
        self.assertFalse(received["output_hidden_states"])

    def test_a_later_category_changes_an_earlier_category_state(self):
        queries = self.queries().detach()
        changed = queries.clone()
        changed[0, -1, 0] += 10
        original = self.llm("red shirt", queries)
        modified = self.llm("red shirt", changed)
        self.assertFalse(torch.allclose(original[:, 0], modified[:, 0], atol=1e-7, rtol=1e-7))

    def test_prompt_changes_category_states(self):
        queries = self.queries()
        red = self.llm("red shirt", queries)
        blue = self.llm("blue shoes", queries)
        self.assertFalse(torch.allclose(red, blue, atol=1e-6))

    def test_frozen_backbone_preserves_query_gradients_and_dtype_conversion(self):
        self.llm.train()
        self.queries.double()
        states = self.llm("red shirt", self.queries())
        self.assertFalse(self.backbone.training)
        self.assertEqual(states.dtype, torch.float32)
        states[..., 0].sum().backward()
        gradient = self.queries.embeddings.weight.grad
        self.assertIsNotNone(gradient)
        self.assertEqual(gradient.dtype, torch.float64)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.backbone.parameters()))

    def test_trainable_backbone_receives_gradients(self):
        llm = Qwen3LLM(trainable=True).train()
        llm("red shirt", self.queries())[..., 0].sum().backward()
        projection = self.backbone.layers[0].self_attn.q_proj
        self.assertTrue(self.backbone.training)
        self.assertIsNotNone(projection.weight.grad)
        self.assertGreater(projection.weight.grad.abs().sum().item(), 0)

    def test_image_seed_is_a_prefix_token_and_receives_gradients(self):
        received = {}

        def capture(module, args, kwargs):
            received.update(kwargs)

        handle = self.backbone.register_forward_pre_hook(capture, with_kwargs=True)
        self.addCleanup(handle.remove)
        seed = torch.randn(16, dtype=torch.float64, requires_grad=True)
        states = self.llm("red shirt shoes", self.queries(), seed_embeddings=seed)
        self.assertEqual(tuple(states.shape), (1, 11, 16))
        self.assertTrue(torch.equal(received["inputs_embeds"][0, 3], seed.float()))
        mask = received["attention_mask"]
        blocked = torch.finfo(mask.dtype).min
        self.assertEqual(tuple(mask.shape), (1, 1, 15, 15))
        self.assertTrue(bool((mask[0, 0, :4, 4:] == blocked).all()))
        self.assertTrue(bool((mask[0, 0, 4:, :] == 0).all()))
        changed = self.llm("red shirt shoes", self.queries(), seed_embeddings=-seed)
        self.assertFalse(torch.allclose(states, changed, atol=1e-6))
        states[..., 0].sum().backward()
        self.assertGreater(seed.grad.abs().sum().item(), 0)
        self.assertEqual(seed.grad.dtype, torch.float64)
        self.assertGreater(self.queries.embeddings.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.backbone.parameters()))

    def test_padded_seed_batches_match_individual_examples(self):
        texts = ["red shirt shoes", "blue"]
        queries = self.queries(batch_size=2)
        seeds = torch.randn(2, 16)
        for padding_side in ("left", "right"):
            with self.subTest(padding_side=padding_side):
                self.tokenizer.padding_side = padding_side
                batch = self.llm(texts, queries, seed_embeddings=seeds)
                for index, text in enumerate(texts):
                    individual = self.llm(text, queries[index], seed_embeddings=seeds[index])
                    torch.testing.assert_close(batch[index], individual[0], atol=1e-5, rtol=1e-5)

    def test_seed_validation_and_context_limit(self):
        for seed in (torch.zeros(15), torch.zeros(2, 16), torch.zeros(1, 1, 16)):
            with self.subTest(shape=seed.shape), self.assertRaisesRegex(ValueError, "shape"):
                self.llm("red", self.queries(), seed_embeddings=seed)
        with self.assertRaises(TypeError):
            self.llm("red", self.queries(), seed_embeddings=torch.zeros(16, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.llm("red", self.queries(), seed_embeddings=torch.full((16,), float("nan")))
        self.llm(" ".join(["red"] * 52), self.queries(), seed_embeddings=torch.zeros(16))
        with self.assertRaisesRegex(ValueError, "exceeds Qwen3 context length 64"):
            self.llm(" ".join(["red"] * 53), self.queries(), seed_embeddings=torch.zeros(16))

    def test_complete_text_seed_forward_with_real_qwen_and_heads(self):
        model = SeedToOutfitModel(
            self.llm, self.queries,
            retrieval_heads=[MLP(16, 6) for _ in range(11)],
            category_heads=[MLP(16, 1) for _ in range(11)],
            fine_grained_heads=[MLP(16, 3) for _ in range(11)],
        )
        output = model(["casual", "blue"], seed_texts=["red shirt", "blue shoes"])
        self.assertEqual(tuple(output.retrieval_embeddings.shape), (2, 11, 6))
        self.assertEqual(tuple(output.category_logits.shape), (2, 11))
        self.assertEqual([tuple(tensor.shape) for tensor in output.fine_grained_logits], [(2, 3)] * 11)
        output.category_logits.sum().backward()
        self.assertGreater(self.queries.embeddings.weight.grad.abs().sum().item(), 0)

    def test_invalid_prompts_and_queries_are_rejected(self):
        for prompts in ("", "  ", [], ["red", ""], [123], None):
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                self.llm(prompts, self.queries())
        for shape in ((10, 16), (11, 15), (2, 11, 16), (16,), (1, 11, 16, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.llm("red", torch.zeros(shape))
        for queries in ([1, 2], torch.zeros(1, 11, 16, dtype=torch.long)):
            with self.assertRaises(TypeError):
                self.llm("red", queries)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                self.llm("red", torch.full((1, 11, 16), value))

    def test_context_limit_includes_all_eleven_queries(self):
        states = self.llm(" ".join(["red"] * 53), self.queries())
        self.assertEqual(tuple(states.shape), (1, 11, 16))
        with self.assertRaisesRegex(ValueError, "exceeds Qwen3 context length 64"):
            self.llm(" ".join(["red"] * 54), self.queries())

    def test_tokenizer_uses_eos_when_padding_is_missing(self):
        self.tokenizer.pad_token = None
        llm = Qwen3LLM()
        self.assertEqual(llm.tokenizer.pad_token_id, llm.tokenizer.eos_token_id)
        states = llm(["red shirt", "blue"], self.queries(batch_size=2))
        self.assertEqual(tuple(states.shape), (2, 11, 16))


class Qwen3CheckpointTests(unittest.TestCase):
    def test_loads_a_local_causal_checkpoint_as_a_backbone_without_an_lm_head(self):
        vocabulary = {"[PAD]": 0, "[UNK]": 1, "red": 2, "shirt": 3}
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        )
        source = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=4, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=64, sliding_window=None,
        ))
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            with patch(
                "model.llm.download.snapshot_download", side_effect=AssertionError("unexpected download"),
            ):
                llm = Qwen3LLM(directory, dtype=torch.float32)
        self.assertIsInstance(llm.backbone, Qwen3Model)
        self.assertFalse(hasattr(llm.backbone, "lm_head"))
        self.assertTrue(torch.equal(
            llm.backbone.get_input_embeddings().weight,
            source.get_input_embeddings().weight,
        ))
        queries = torch.randn(11, 16, requires_grad=True)
        states = llm("red shirt", queries)
        self.assertEqual(tuple(states.shape), (1, 11, 16))
        states[..., 0].sum().backward()
        self.assertGreater(queries.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
