"""Complete outfit model: generation, item representation and compatibility."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data.types import FashionItem, SeedTrainingBatch
from ..config import DEFAULT_MODEL_CONFIG, ModelConfig
from ..mlp import MLP
from ..multimodal_representation import MultimodalEncoder
from .seed_to_outfit import SeedToOutfitModel, SeedToOutfitOutput
from ..losses import (
    category_classification_loss, compatibility_loss,
    fine_grained_category_loss, outfit_retrieval_loss,
)

from ..compatibility import OutfitCompatibility

LOSS_NAMES = ("total", "retrieval", "cp", "composition", "category", "fine_grained")


class Text2OutfitModel(nn.Module):
    def __init__(
        self, generator: SeedToOutfitModel, item_encoder: MultimodalEncoder,
        compatibility: OutfitCompatibility, subcategories: tuple[tuple[str, ...], ...],
        *, config: ModelConfig = DEFAULT_MODEL_CONFIG,
    ) -> None:
        super().__init__()
        config.validate()
        if len(subcategories) != len(generator.categories):
            raise ValueError("provide a subcategory vocabulary per category")
        if item_encoder.output_dim != compatibility.input_dim:
            raise ValueError("item embeddings must match the CP input width; no CP adapter is used")
        if any(head.output_dim != item_encoder.output_dim for head in generator.retrieval_heads):
            raise ValueError("retrieval heads must match item_encoder.output_dim")
        if any(head.output_dim != max(1, len(labels)) for head, labels in zip(generator.fine_grained_heads, subcategories)):
            raise ValueError("fine-grained head widths must match the taxonomy")
        seed_visual = generator.seed_encoder.visual_encoder if generator.seed_encoder is not None else None
        for encoder in (item_encoder.visual_encoder, item_encoder.text_encoder, seed_visual):
            if encoder is not None and any(parameter.requires_grad for parameter in encoder.parameters()):
                raise ValueError("visual and text encoders must be frozen")
        self.generator, self.item_encoder, self.compatibility = generator, item_encoder, compatibility
        self.config = config
        self.subcategories, self.seed_mode = subcategories, config.seed_mode
        self.loss_weights, self.temperature = config.loss_weights, config.temperature
        self._labels = tuple({label: i for i, label in enumerate(labels)} for labels in subcategories)

    def forward(
        self, prompts: str | Sequence[str], *, seed_images: Tensor | None = None,
        seed_texts: str | Sequence[str] | None = None, seed_features: Tensor | None = None,
    ) -> SeedToOutfitOutput:
        """Run generation; losses are computed explicitly with compute_losses()."""
        return self.generator(prompts, seed_images=seed_images, seed_texts=seed_texts, seed_features=seed_features)

    def compute_losses(self, batch: SeedTrainingBatch) -> dict[str, Tensor]:
        if not bool((batch.outfits.labels == 1).all()):
            raise ValueError("generation training requires positive outfits")
        if len(batch.prompts) != len(batch.outfits.outfits) or len(batch.seeds) != len(batch.prompts):
            raise ValueError("provide one seed and prompt per outfit")
        device = next(self.item_encoder.mlp.parameters()).device
        if self.seed_mode == "text":
            output = self.generator(batch.prompts, seed_texts=[item.description for item in batch.seeds])
        elif all(item.embedding is not None for item in batch.seeds):
            features = torch.stack([item.embedding for item in batch.seeds if item.embedding is not None])
            output = self.generator(batch.prompts, seed_features=features[:, :features.size(1) // 2])
        else:
            if any(item.image is None for item in batch.seeds):
                raise ValueError("all image seeds must have images or all must have cached features")
            images = torch.stack([item.image for item in batch.seeds if item.image is not None]).to(device)
            output = self.generator(batch.prompts, seed_images=images)

        positives, negative_groups, owners, categories, labels = [], [], [], [], []
        for owner, (outfit, negatives) in enumerate(zip(batch.outfits.outfits, batch.outfits.negatives)):
            for item, group in zip(outfit, negatives):
                category = self.generator.categories.index(item.category)
                positives.append(item)
                negative_groups.append(group)
                owners.append(owner)
                categories.append(category)
                if item.subcategory is None or item.subcategory not in self._labels[category]:
                    raise ValueError(f"unknown subcategory {item.subcategory!r} for {item.category}")
                labels.append(self._labels[category][item.subcategory])
        all_items = {item.item_id: item for item in positives}
        all_items.update({item.item_id: item for group in negative_groups for item in group})
        ids = list(all_items)
        features = self.encode_items(list(all_items.values()))
        positions = {item_id: i for i, item_id in enumerate(ids)}
        positive = features[[positions[item.item_id] for item in positives]]
        negatives = torch.stack([features[[positions[item.item_id] for item in group]] for group in negative_groups])
        owners_t = torch.tensor(owners, device=device)
        categories_t = torch.tensor(categories, device=device)
        target_mask = batch.outfits.category_mask.to(device)
        batch_size, count = len(batch.prompts), len(positives)
        # Preserve all items when several belong to the same category.
        retrieval = outfit_retrieval_loss(
            output.retrieval_embeddings[owners_t, categories_t].unsqueeze(1),
            positive.unsqueeze(1), negatives.unsqueeze(1),
            torch.ones((count, 1), device=device, dtype=torch.bool), temperature=self.temperature,
        ) * count / batch_size
        category = category_classification_loss(output.category_logits, target_mask)
        fine_mask = F.one_hot(categories_t, num_classes=len(self.generator.categories)).bool()
        fine_targets = torch.full_like(fine_mask, -100, dtype=torch.long)
        fine_targets[torch.arange(count, device=device), categories_t] = torch.tensor(labels, device=device)
        fine = fine_grained_category_loss(
            tuple(logits[owners_t] for logits in output.fine_grained_logits), fine_targets, fine_mask,
        ) * count / batch_size
        composition = category + fine
        cp = compatibility_loss(self.compatibility(output.retrieval_embeddings, target_mask))
        ret_weight, cp_weight, comp_weight = self.loss_weights
        total = ret_weight * retrieval + cp_weight * cp + comp_weight * composition
        return dict(total=total, retrieval=retrieval, cp=cp, composition=composition, category=category, fine_grained=fine)

    def encode_items(self, items: list[FashionItem]) -> Tensor:
        """Encode cached and raw partitions once each and restore their order."""
        outputs: dict[int, Tensor] = {}
        for cached in (False, True):
            positions = [i for i, item in enumerate(items) if (item.embedding is not None) == cached]
            if not positions:
                continue
            selected = [items[i] for i in positions]
            if cached:
                result = self.item_encoder(precomputed=torch.stack([item.embedding for item in selected if item.embedding is not None]))
            else:
                weights = next(self.item_encoder.mlp.parameters())
                images = torch.stack([item.image for item in selected if item.image is not None]).to(weights.device)
                result = self.item_encoder(images, [item.description for item in selected])
            outputs.update(zip(positions, result))
        return torch.stack([outputs[i] for i in range(len(items))])

    def architecture(self) -> dict[str, Any]:
        """Read actual layer widths and parameter counts from the constructed model."""
        mlps: list[tuple[str, MLP]] = []
        if self.generator.seed_encoder is not None:
            mlps.append(("seed", self.generator.seed_encoder.mlp))
        mlps.append(("item", self.item_encoder.mlp))
        for name, heads in (
            ("retrieval", self.generator.retrieval_heads),
            ("presence", self.generator.category_heads),
            ("subcategory", self.generator.fine_grained_heads),
        ):
            mlps.extend((f"{name}.{category}", cast(MLP, head)) for category, head in zip(self.generator.categories, heads))
        return dict(
            llm_width=self.generator.llm.output_dim, categories=list(self.generator.categories),
            mlps={name: dict(dimensions=list(mlp.dimensions), parameters=sum(p.numel() for p in mlp.parameters()))
                  for name, mlp in mlps},
            parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def architecture_summary(self) -> str:
        architecture = self.architecture()
        lines = [
            f"Text2OutfitModel | LLM width: {architecture['llm_width']} | Categories: {len(self.generator.categories)}",
            f"{'MLP':<29} {'Layer widths (Linear/ReLU)':<40} Parameters",
        ]
        for name, info in architecture["mlps"].items():
            dimensions = " -> ".join(map(str, info["dimensions"]))
            lines.append(f"{name:<29} {dimensions:<40} {info['parameters']:,}")
        lines.append(f"Total parameters: {architecture['parameters']:,} | Trainable: {architecture['trainable_parameters']:,}")
        return "\n".join(lines)
