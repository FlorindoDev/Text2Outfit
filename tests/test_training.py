"""Offline gradient, cache, checkpoint and plot checks on a real tiny Qwen3."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from data import OUTFIT_CATEGORIES
from data.types import FashionItem, OutfitExample
from model import (
    CategoryQueryEmbeddings, MLP, MultimodalEncoder, OutfitSeedEncoder,
    Qwen3LLM, SeedToOutfitModel, TextEncoder, VisualEncoder, Text2OutfitModel,
)
from training import (
    OutfitCompatibility, SeedExample, TrainingConfig,
    collate_seed_outfits, train,
)
from training.data import PrecomputedItemFeatures, SeedOutfitDataset
from model.outfit.text2outfit import LOSS_NAMES


class Visual(VisualEncoder):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4).requires_grad_(False)

    @property
    def output_dim(self):
        return 4

    def forward(self, images):
        return F.normalize(self.projection(images.mean((-2, -1))), dim=-1)


class Textual(TextEncoder):
    @property
    def output_dim(self):
        return 4

    def forward(self, descriptions):
        return F.normalize(torch.tensor([[len(text), 1., 2., 3.] for text in descriptions]), dim=-1)


class Batcher(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.padding_embedding = nn.Parameter(torch.ones(input_dim), requires_grad=False)
        self.config = SimpleNamespace(max_items=16, model_dim=input_dim)

    def forward(self, outfits):
        padded = nn.utils.rnn.pad_sequence(outfits, batch_first=True)
        mask = torch.arange(padded.size(1))[None, :] >= torch.tensor([len(outfit) for outfit in outfits])[:, None]
        return SimpleNamespace(item_embeddings=F.normalize(padded, dim=-1), padding_mask=mask)


class Scorer(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(self, batch):
        present = (~batch.padding_mask).unsqueeze(-1)
        mean = (batch.item_embeddings * present).sum(1) / present.sum(1)
        return self.head(mean).sigmoid()


def make_llm(enable_lora=True):
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "outfit": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", padding_side="left")
    backbone = Qwen3Model(Qwen3Config(
        vocab_size=3, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, attention_dropout=0.0, sliding_window=None,
    ))
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.Qwen3Model.from_pretrained", return_value=backbone),
        patch("model.llm.download.resolve_model_path", return_value=Path("tiny-llm")),
    ):
        llm = Qwen3LLM()
    if enable_lora:
        llm.enable_lora(rank=2, alpha=4, dropout=0, gradient_checkpointing=True)
    return llm


def make_model():
    torch.manual_seed(17)
    visual, textual = Visual(), Textual()
    llm = make_llm()
    generator = SeedToOutfitModel(
        llm, CategoryQueryEmbeddings(16), seed_encoder=OutfitSeedEncoder(visual, MLP(4, 16)),
        retrieval_heads=[MLP(16, 8) for _ in range(11)],
        category_heads=[MLP(16, 1) for _ in range(11)],
        fine_grained_heads=[MLP(16, 2) for _ in range(11)],
    )
    return Text2OutfitModel(
        generator, MultimodalEncoder(visual, textual, MLP(8, 8)),
        OutfitCompatibility(Scorer(), Batcher()),
        tuple(("a", "b") for _ in range(11)),
    )


def make_example(cached=True, duplicate=False):
    visual, textual = Visual(), Textual()

    def item(name, category, label):
        image = torch.randn(3, 2, 2)
        features = torch.cat((visual(image[None])[0], textual([name])[0])) if cached else None
        return FashionItem(name, None if cached else image, name, category, label, features)

    positives, negatives = [], []
    for i, category in enumerate(OUTFIT_CATEGORIES):
        positives.append(item(f"item{i}", category, "a"))
        negatives.append(tuple(item(f"neg{i}_{j}", category, "b") for j in range(2)))
    if duplicate:
        positives.append(item("another_top", "tops", "b"))
        negatives.append(negatives[0])
    outfit = OutfitExample("outfit1", tuple(positives), 1, tuple(negatives))
    return SeedExample(outfit, "outfit", positives[0])


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.example = make_example()

    def test_lora_and_every_external_mlp_update_with_frozen_backbones(self):
        self.model.train()
        batch = collate_seed_outfits([self.example])
        before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            losses = self.model.compute_losses(batch)
            losses["total"].backward()
            optimizer.step()
        trainable_llm = [name for name, p in self.model.generator.llm.backbone.named_parameters() if p.requires_grad]
        self.assertTrue(trainable_llm)
        self.assertTrue(all("lora_" in name for name in trainable_llm))
        self.assertTrue(all(any(target in name for target in ("q_proj", "k_proj", "v_proj")) for name in trainable_llm))
        self.assertTrue(self.model.generator.llm.backbone.is_gradient_checkpointing)
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        modules = ["generator.seed_encoder.mlp", "generator.category_queries", "item_encoder.mlp"]
        modules.extend(f"generator.{head}.{i}" for head in ("retrieval_heads", "category_heads", "fine_grained_heads") for i in range(11))
        for prefix in modules:
            with self.subTest(module=prefix):
                self.assertTrue(any(not torch.equal(parameter, before[name]) for name, parameter in self.model.named_parameters() if name.startswith(prefix + ".")))
        for suffix in ("lora_A.default.weight", "lora_B.default.weight"):
            self.assertTrue(any(not torch.equal(parameter, before[name]) for name, parameter in self.model.named_parameters() if name.endswith(suffix)))
        self.assertFalse(self.model.compatibility.cp.training)
        with self.assertRaisesRegex(ValueError, "already"):
            self.model.generator.llm.enable_lora()

    def test_cp_alone_backpropagates_through_frozen_scorer(self):
        losses = self.model.compute_losses(collate_seed_outfits([self.example]))
        losses["cp"].backward()
        self.assertGreater(self.model.generator.category_queries.embeddings.weight.grad.abs().sum(), 0)
        for head in self.model.generator.retrieval_heads:
            self.assertGreater(head.layers[0].weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in self.model.compatibility.cp.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.model.compatibility.parameters()))

    def test_raw_and_cached_forward_match_and_cache_keeps_item_mlp_trainable(self):
        raw = make_example(cached=False)
        encoder = self.model.item_encoder
        def cache(item):
            with torch.no_grad():
                visual = encoder.visual_encoder(item.image[None])[0]
                text = encoder.text_encoder([item.description])[0]
            return replace(item, image=None, embedding=torch.cat((visual, text)))
        outfit = replace(raw.outfit, items=tuple(cache(item) for item in raw.outfit.items),
                         negatives=tuple(tuple(cache(item) for item in group) for group in raw.outfit.negatives))
        cached = SeedExample(outfit, raw.prompt, outfit.items[0])
        expected = self.model.compute_losses(collate_seed_outfits([raw]))
        with patch.object(encoder.visual_encoder, "forward", side_effect=AssertionError("unexpected image encoding")):
            actual = self.model.compute_losses(collate_seed_outfits([cached]))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=1e-5, rtol=1e-5)
        actual["retrieval"].backward()
        self.assertTrue(all(p.grad is not None for p in encoder.mlp.parameters()))

    def test_text_seed_and_repeated_categories_remain_trainable(self):
        self.model.seed_mode = "text"
        example = make_example(duplicate=True)
        losses = self.model.compute_losses(collate_seed_outfits([example]))
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        torch.testing.assert_close(losses["composition"], losses["category"] + losses["fine_grained"])
        torch.testing.assert_close(losses["total"], losses["retrieval"] + losses["cp"] + losses["composition"])
        losses["total"].backward()
        with self.assertRaisesRegex(ValueError, "positive"):
            SeedOutfitDataset([replace(example.outfit, label=0)], training=True)[0]

    def test_cache_only_encoder_and_invalid_inputs(self):
        features = torch.randn(2, 8)
        cached = MultimodalEncoder(None, None, MLP(8, 6))
        cached(precomputed=features).sum().backward()
        self.assertIsNotNone(cached.mlp.layers[0].weight.grad)
        for values in (torch.zeros(2, 7), torch.full((2, 8), float("nan")), torch.ones(2, 8, dtype=torch.long)):
            with self.assertRaises(ValueError):
                cached(precomputed=values)
        with self.assertRaises(ValueError):
            cached(torch.zeros(2,3,2,2), ["a", "b"], precomputed=features)
        with self.assertRaisesRegex(ValueError, "encoders"):
            cached(torch.zeros(2,3,2,2), ["a", "b"])

    def test_checkpoint_resume_and_plots_for_every_epoch(self):
        from training.plots import save_loss_plots

        def interrupt_after_checkpoint(history, directory, epoch):
            save_loss_plots(history, directory, epoch)
            raise RuntimeError("simulated interruption")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loader = DataLoader([self.example], batch_size=1, collate_fn=collate_seed_outfits,
                                generator=torch.Generator().manual_seed(42))
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"), patch(
                "training.plots.save_loss_plots", side_effect=interrupt_after_checkpoint,
            ):
                train(self.model, loader, loader, TrainingConfig(output_dir=root, epochs=2))
            first = root / "epochs/seed_to_outfit_epoch_001.pt"
            checkpoint = torch.load(first, weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
            self.assertEqual(checkpoint["scheduler_state_dict"]["last_epoch"], 1)
            self.assertEqual(checkpoint["scheduler_state_dict"]["total_steps"], 2)
            self.assertTrue(any("lora_A" in key for key in checkpoint["model_state_dict"]))
            self.assertFalse(any("embed_tokens" in key or "compatibility.cp" in key for key in checkpoint["model_state_dict"]))
            self.assertEqual(len(list((root / "plots").glob("*.png"))), 7)
            with self.assertRaisesRegex(ValueError, "total steps"):
                train(make_model(), loader, loader, TrainingConfig(output_dir=root, epochs=3, resume=first))
            legacy = dict(checkpoint, checkpoint_schema_version=1)
            legacy.pop("scheduler_state_dict")
            torch.save(legacy, root / "legacy.pt")
            with self.assertRaisesRegex(ValueError, "OneCycleLR.*scheduler state"):
                train(make_model(), loader, loader,
                      TrainingConfig(output_dir=root, epochs=2, resume=root / "legacy.pt"))
            restored = make_model()
            history = train(restored, loader, loader, TrainingConfig(output_dir=root, epochs=2, resume=first))
            self.assertEqual(history["epochs"], [1, 2])
            self.assertEqual(len(list((root / "epochs").glob("*.pt"))), 2)
            self.assertEqual(len(list((root / "plots").glob("*.png"))), 14)
            self.assertTrue((root / "best.pt").is_file())
            self.assertEqual(json.loads((root / "history.json").read_text())["epochs"], [1,2])
            resumed = torch.load(root / "epochs/seed_to_outfit_epoch_002.pt", weights_only=True)
            self.assertEqual(resumed["global_step"], 2)
            self.assertEqual(resumed["scheduler_state_dict"]["last_epoch"], 2)
            uninterrupted = make_model()
            with patch("training.plots.save_loss_plots"):
                train(uninterrupted, loader, loader,
                      TrainingConfig(output_dir=root / "uninterrupted", epochs=2))
            for actual, expected in zip(restored.parameters(), uninterrupted.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            reference = torch.load(root / "uninterrupted/epochs/seed_to_outfit_epoch_002.pt", weights_only=True)
            self.assertEqual(resumed["scheduler_state_dict"], reference["scheduler_state_dict"])
            self.assertEqual(resumed["optimizer_state_dict"]["param_groups"], reference["optimizer_state_dict"]["param_groups"])
            with self.assertRaises(FileExistsError):
                train(restored, loader, loader, TrainingConfig(output_dir=root))

    def test_gradient_accumulation_handles_uneven_final_batches(self):
        other = copy.deepcopy(self.model)
        examples = [self.example] * 5
        with tempfile.TemporaryDirectory() as directory, patch("training.plots.save_loss_plots"):
            small = DataLoader(examples, batch_size=2, collate_fn=collate_seed_outfits)
            large = DataLoader(examples, batch_size=5, collate_fn=collate_seed_outfits)
            train(self.model, small, small, TrainingConfig(output_dir=Path(directory)/"small", epochs=1, gradient_accumulation_steps=4))
            train(other, large, large, TrainingConfig(output_dir=Path(directory)/"large", epochs=1, gradient_accumulation_steps=1))
        for actual, expected in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_one_cycle_follows_optimizer_updates_including_partial_groups(self):
        rates = []

        class RecordingAdamW(AdamW):
            def step(self, closure=None):
                rates.append((self.param_groups[0]["lr"], self.param_groups[0]["betas"][0]))
                return super().step(closure)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("training.trainer.AdamW", RecordingAdamW),
            patch("training.plots.save_loss_plots"),
        ):
            root = Path(directory)
            loader = DataLoader([self.example] * 5, batch_size=2, collate_fn=collate_seed_outfits)
            config = TrainingConfig(output_dir=root, epochs=3, gradient_accumulation_steps=2)
            train(self.model, loader, loader, config)
            checkpoint = torch.load(root / "epochs/seed_to_outfit_epoch_003.pt", weights_only=True)
            settings = json.loads((root / "config.json").read_text())["scheduler"]
        self.assertEqual(settings["name"], "OneCycleLR")
        self.assertEqual(settings["total_steps"], 6)
        self.assertEqual(checkpoint["global_step"], 6)
        self.assertEqual(checkpoint["scheduler_state_dict"]["last_epoch"], 6)
        optimizer = AdamW([nn.Parameter(torch.zeros(()))], lr=config.learning_rate)
        scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=6,
                               pct_start=0.3, anneal_strategy="cos", div_factor=25.0,
                               final_div_factor=10_000.0)
        expected = []
        for _ in range(6):
            expected.append((optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["betas"][0]))
            optimizer.step()
            scheduler.step()
        self.assertEqual(rates, expected)

    def test_training_and_validation_log_every_50_batches_and_the_last_batch(self):
        parameter = next(p for p in self.model.parameters() if p.requires_grad)

        def compute_losses(_batch):
            loss = parameter.flatten()[0] * 0 + (2.0 if torch.is_grad_enabled() else 3.0)
            return dict.fromkeys(LOSS_NAMES, loss)

        for batch_count in (49, 50, 51, 100):
            with (
                self.subTest(batch_count=batch_count),
                tempfile.TemporaryDirectory() as directory,
                patch.object(self.model, "compute_losses", side_effect=compute_losses),
                patch("training.plots.save_loss_plots"),
                self.assertLogs("training.seed_to_outfit", level="INFO") as captured,
            ):
                loader = DataLoader([self.example] * batch_count, batch_size=1,
                                    collate_fn=collate_seed_outfits)
                train(self.model, loader, loader, TrainingConfig(output_dir=Path(directory), epochs=1))
            messages = [record.getMessage() for record in captured.records]
            batches = sorted(set(range(50, batch_count + 1, 50)) | {batch_count})
            expected = [
                (phase, f"{batch}/{batch_count}", f"{loss:.6f}")
                for phase, loss in (("train", 2.0), ("validation", 3.0))
                for batch in batches
            ]
            progress = [dict(field.split("=", 1) for field in message.split())
                        for message in messages if "microbatch=" in message]
            self.assertEqual([(entry["phase"], entry["microbatch"], entry["loss"]) for entry in progress], expected)
            for entry in progress:
                self.assertEqual(set(entry), {"epoch", "microbatch", "loss", "lr", "phase"})
                self.assertEqual(entry["epoch"], "1")
                self.assertGreater(float(entry["lr"]), 0)
                self.assertLessEqual(float(entry["lr"]), 1e-4)
            self.assertTrue(all(entry["lr"] == progress[len(batches) - 1]["lr"]
                                for entry in progress if entry["phase"] == "validation"))
            self.assertEqual(messages[0], f"epoch=1/1 phase=train batches={batch_count}")
            self.assertEqual(messages[1 + len(batches)], f"epoch=1/1 phase=validation batches={batch_count}")
            self.assertEqual(messages[-1], "epoch=1/1 train_loss=2.000000 val_loss=3.000000 best_val_loss=3.000000")


class CacheTests(unittest.TestCase):
    def test_existing_manifest_format_is_indexed_by_item_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = torch.randn(2, 8)
            torch.save(dict(item_ids=["a", "b"], embeddings=features), root / "shard-00000.pt")
            manifest = dict(embedding_dim=8, count=2, shards=[dict(file="shard-00000.pt")],
                            encoder=dict(modality_dim=4, aggregation="concat_visual_then_text", normalization="l2_per_modality"))
            (root/"manifest.json").write_text(json.dumps(manifest))
            cache = PrecomputedItemFeatures(root/"manifest.json")
            torch.testing.assert_close(cache["b"], features[1])
            self.assertEqual(list(cache), ["a", "b"])
            with self.assertRaises(KeyError):
                cache["missing"]


if __name__ == "__main__":
    unittest.main()
