"""Compose the seed-to-outfit forward, leaving losses and catalog encoding outside."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .category_query_embeddings import CategoryQueryEmbeddings
from .llm import LLM
from .llm.base import _validate_prompts
from .mlp import MLP
from .outfit_seed_encoder import OutfitSeedEncoder


@dataclass(frozen=True)
class SeedToOutfitOutput:
    """Continuous predictions in the shared category order, without decoding or loss."""

    retrieval_embeddings: Tensor  # [B, 11, D]
    category_logits: Tensor  # [B, 11]
    fine_grained_logits: tuple[Tensor, ...]  # 11 tensors [B, C_i]


class SeedToOutfitModel(nn.Module):
    """Prompt and seed -> category queries -> LLM -> retrieval/composition heads."""

    def __init__(
        self,
        llm: LLM,
        category_queries: CategoryQueryEmbeddings,
        *,
        retrieval_heads: Sequence[MLP],
        category_heads: Sequence[MLP],
        fine_grained_heads: Sequence[MLP],
        seed_encoder: OutfitSeedEncoder | None = None,
    ) -> None:
        super().__init__()
        if llm.categories != category_queries.categories:
            raise ValueError("LLM and queries must use the same category order")
        if category_queries.output_dim != llm.output_dim:
            raise ValueError("query width must match the LLM hidden width")
        if seed_encoder is not None and seed_encoder.output_dim != llm.output_dim:
            raise ValueError("seed encoder output_dim must match the LLM hidden width")
        num_categories = len(category_queries.categories)
        for name, heads in (
            ("retrieval", retrieval_heads),
            ("category", category_heads),
            ("fine-grained", fine_grained_heads),
        ):
            if len(heads) != num_categories:
                raise ValueError(f"provide one {name} head per category")
            if any(head.input_dim != llm.output_dim for head in heads):
                raise ValueError(f"{name} head input_dim must match the LLM hidden width")
        if len({head.output_dim for head in retrieval_heads}) != 1:
            raise ValueError("all retrieval heads must have the same output_dim")
        if any(head.output_dim != 1 for head in category_heads):
            raise ValueError("each category head must produce one logit")
        self.llm = llm
        self.category_queries = category_queries
        self.seed_encoder = seed_encoder
        self.retrieval_heads = nn.ModuleList(retrieval_heads)
        self.category_heads = nn.ModuleList(category_heads)
        self.fine_grained_heads = nn.ModuleList(fine_grained_heads)

    @property
    def categories(self) -> tuple[str, ...]:
        return self.category_queries.categories

    def forward(
        self,
        prompts: str | Sequence[str],
        *,
        seed_images: Tensor | None = None,
        seed_texts: str | Sequence[str] | None = None,
    ) -> SeedToOutfitOutput:
        """Use exactly one seed modality, with one seed for each prompt."""
        texts = _validate_prompts(prompts)
        if (seed_images is None) == (seed_texts is None):
            raise ValueError("provide exactly one of seed_images or seed_texts")
        seed_embeddings = None
        if seed_images is not None:
            if self.seed_encoder is None:
                raise ValueError("seed_encoder is required for seed_images")
            if not isinstance(seed_images, Tensor) or seed_images.ndim not in (3, 4):
                raise ValueError("seed_images must have shape [3, H, W] or [B, 3, H, W]")
            seed_batch_size = 1 if seed_images.ndim == 3 else seed_images.size(0)
            if seed_batch_size != len(texts):
                raise ValueError("provide one seed image per prompt")
            seed_embeddings = self.seed_encoder(seed_images)
        elif seed_texts is not None:
            descriptions = _validate_prompts(seed_texts)
            if len(descriptions) != len(texts):
                raise ValueError("provide one seed text per prompt")
            texts = [
                f"{prompt}\nSeed item: {description}"
                for prompt, description in zip(texts, descriptions)
            ]
        queries = self.category_queries(batch_size=len(texts))
        hidden_states = self.llm(texts, queries, seed_embeddings=seed_embeddings)
        return SeedToOutfitOutput(
            retrieval_embeddings=torch.stack([
                _apply_head(head, hidden_states[:, index, :])
                for index, head in enumerate(self.retrieval_heads)
            ], dim=1),
            category_logits=torch.cat([
                _apply_head(head, hidden_states[:, index, :])
                for index, head in enumerate(self.category_heads)
            ], dim=1),
            fine_grained_logits=tuple(
                _apply_head(head, hidden_states[:, index, :])
                for index, head in enumerate(self.fine_grained_heads)
            ),
        )


def _apply_head(head: nn.Module, hidden_states: Tensor) -> Tensor:
    # The LLM may return BF16 states while the small prediction heads use FP32.
    weights = next(head.parameters())
    return head(hidden_states.to(device=weights.device, dtype=weights.dtype))
