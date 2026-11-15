"""Qwen3 backbone with external, mutually visible outfit category queries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from ..config import DEFAULT_MODEL_CONFIG
from .base import LLM, _validate_prompts


class Qwen3LLM(LLM):
    """Append eleven queries to each prompt and return their last-layer states."""

    DEFAULT_MODEL_NAME = DEFAULT_MODEL_CONFIG.llm_model_name

    def __init__(
        self,
        model_name: str | Path = DEFAULT_MODEL_NAME,
        *,
        trainable: bool = DEFAULT_MODEL_CONFIG.llm_trainable,
        dtype: torch.dtype | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        if dtype is not None and (
            not isinstance(dtype, torch.dtype) or not dtype.is_floating_point
        ):
            raise ValueError("dtype must be a floating-point torch.dtype or None")
        try:
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
        except ImportError as error:
            raise ImportError("Qwen3LLM requires transformers>=4.51.3,<4.52") from error

        from .download import resolve_model_path

        self.model_path = resolve_model_path(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), padding_side="left", local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Qwen3 tokenizer requires a padding or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = Qwen3Model.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype if dtype is not None else "auto",
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self._output_dim = self.backbone.config.hidden_size
        self._max_length = min(
            self.backbone.config.max_position_embeddings, self.tokenizer.model_max_length,
        )
        if self._max_length <= len(self.categories):
            raise ValueError("Qwen3 context must leave room for the prompt and 11 queries")
        self._trainable = trainable
        self.lora_settings: dict[str, int | float | bool] | None = None
        self.backbone.requires_grad_(trainable)
        if not trainable:
            self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> Qwen3LLM:
        super().train(mode)
        if not self._trainable:
            self.backbone.eval()
        return self

    def forward(
        self,
        prompts: str | Sequence[str],
        category_embeddings: Tensor,
        *,
        seed_embeddings: Tensor | None = None,
    ) -> Tensor:
        texts = _validate_prompts(prompts)
        queries = self._validate_queries(category_embeddings, len(texts))
        if seed_embeddings is not None:
            seed_embeddings = self._validate_seed(seed_embeddings, len(texts))
        tokens = self.tokenizer(
            texts, padding=True, truncation=False, return_tensors="pt",
        )
        num_queries = len(self.categories)
        num_seed_tokens = int(seed_embeddings is not None)
        if tokens["input_ids"].size(1) + num_seed_tokens + num_queries > self._max_length:
            raise ValueError(
                f"prompt plus seed and {num_queries} category queries exceeds "
                f"Qwen3 context length {self._max_length}"
            )
        word_embeddings = cast(nn.Embedding, self.backbone.get_input_embeddings())
        device = word_embeddings.weight.device
        text_embeddings = word_embeddings(tokens["input_ids"].to(device))
        queries = queries.to(device=device, dtype=text_embeddings.dtype)
        if seed_embeddings is not None:
            seed = seed_embeddings.to(device=device, dtype=text_embeddings.dtype).unsqueeze(1)
            text_embeddings = torch.cat((text_embeddings, seed), dim=1)
        inputs = torch.cat((text_embeddings, queries), dim=1)
        present = torch.cat((
            tokens["attention_mask"].to(device),
            torch.ones((len(texts), num_seed_tokens + num_queries), device=device, dtype=torch.long),
        ), dim=1)
        position_ids = present.cumsum(dim=1) - 1
        position_ids = position_ids.masked_fill(present == 0, 0)
        outputs = self.backbone(
            inputs_embeds=inputs,
            attention_mask=_category_attention_mask(present, num_queries, inputs.dtype),
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        # In inference, keep only the query storage instead of retaining prompt states.
        return outputs.last_hidden_state[:, -num_queries:, :].clone()

    def enable_lora(
        self, *, rank: int = DEFAULT_MODEL_CONFIG.lora_rank,
        alpha: int = DEFAULT_MODEL_CONFIG.lora_alpha,
        dropout: float = DEFAULT_MODEL_CONFIG.lora_dropout,
        gradient_checkpointing: bool = DEFAULT_MODEL_CONFIG.gradient_checkpointing,
        target_modules: tuple[str, ...] = DEFAULT_MODEL_CONFIG.lora_target_modules,
    ) -> None:
        """Freeze base weights and train Q/K/V adapters without changing the forward."""
        if self.lora_settings is not None:
            raise ValueError("LoRA is already enabled")
        if type(rank) is not int or rank <= 0 or type(alpha) is not int or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive integers")
        if not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not target_modules or any(not name.strip() for name in target_modules):
            raise ValueError("provide non-empty LoRA target module names")
        from peft.mapping import inject_adapter_in_model
        from peft.tuners.lora.config import LoraConfig

        self.backbone.requires_grad_(False)
        inject_adapter_in_model(LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=list(target_modules), bias="none",
        ), self.backbone)
        # Keep optimizer updates in FP32 even when the frozen backbone is BF16.
        for parameter in self.backbone.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        self.lora_settings = dict(
            rank=rank, alpha=alpha, dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._trainable = True
        self.train(self.training)

    def _validate_queries(self, queries: Tensor, batch_size: int) -> Tensor:
        if not isinstance(queries, Tensor) or not torch.is_floating_point(queries):
            raise TypeError("category_embeddings must be a floating-point tensor")
        if queries.ndim == 2:
            queries = queries.unsqueeze(0)
        if queries.shape != (batch_size, len(self.categories), self.output_dim):
            raise ValueError(
                f"category_embeddings must have shape [{batch_size}, "
                f"{len(self.categories)}, {self.output_dim}]"
            )
        if not bool(torch.isfinite(queries).all()):
            raise ValueError("category_embeddings must contain only finite values")
        return queries

    def _validate_seed(self, seed: Tensor, batch_size: int) -> Tensor:
        if not isinstance(seed, Tensor) or not torch.is_floating_point(seed):
            raise TypeError("seed_embeddings must be a floating-point tensor")
        if seed.ndim == 1:
            seed = seed.unsqueeze(0)
        if seed.shape != (batch_size, self.output_dim):
            raise ValueError(f"seed_embeddings must have shape [{batch_size}, {self.output_dim}]")
        if not bool(torch.isfinite(seed).all()):
            raise ValueError("seed_embeddings must contain only finite values")
        return seed


def _category_attention_mask(present: Tensor, num_queries: int, dtype: torch.dtype) -> Tensor:
    """Causal text; queries see the whole prompt and every other query; ignore padding."""
    # shortcut: dense mask for outfit prompts; use block attention for long contexts.
    length = present.size(1)
    blocked = torch.finfo(dtype).min
    mask = torch.full((length, length), blocked, dtype=dtype, device=present.device)
    mask = mask.triu(diagonal=1)
    mask[-num_queries:, -num_queries:] = 0
    return mask[None, None].expand(present.size(0), 1, length, length).masked_fill(
        present[:, None, None, :] == 0, blocked,
    )
