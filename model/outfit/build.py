"""Construct the complete architecture; all defaults come from model/config.py."""

from __future__ import annotations

from dataclasses import replace

import torch

from .category_query_embeddings import CategoryQueryEmbeddings, OUTFIT_CATEGORIES
from ..compatibility import OutfitCompatibility, resolve_cp_assets
from ..config import DEFAULT_MODEL_CONFIG, ModelConfig
from ..llm import Qwen3LLM
from ..mlp import MLP
from ..multimodal_representation import (
    FashionCLIPTextEncoder, FashionCLIPVisualEncoder,
    MarqoFashionSigLIPTextEncoder, MarqoFashionSigLIPVisualEncoder, MultimodalEncoder,
)
from .outfit_seed_encoder import OutfitSeedEncoder
from .seed_to_outfit import SeedToOutfitModel
from .text2outfit import Text2OutfitModel


def build_model(
    subcategories: tuple[tuple[str, ...], ...], config: ModelConfig = DEFAULT_MODEL_CONFIG,
    *, precomputed_modality_dim: int | None = None, device: str | torch.device = "cpu",
) -> Text2OutfitModel:
    """Read backbone/taxonomy widths, then explicitly build each MLP family."""
    config.validate()
    retrieval_dim = config.resolved_retrieval_dim
    config = replace(config, retrieval_dim=retrieval_dim)
    if len(subcategories) != len(OUTFIT_CATEGORIES):
        raise ValueError("provide one subcategory vocabulary per outfit category")
    cp_repo, cp_checkpoint = resolve_cp_assets(config)
    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if str(device) == "auto" else torch.device(device)
    if config.llm_dtype == "auto":
        llm_dtype = torch.bfloat16 if selected_device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    else:
        llm_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[config.llm_dtype]

    if precomputed_modality_dim is not None:
        if type(precomputed_modality_dim) is not int or precomputed_modality_dim <= 0:
            raise ValueError("precomputed_modality_dim must be a positive integer")
        visual, textual = None, None
        visual_dim = text_dim = precomputed_modality_dim
    elif config.encoder == "fashionclip":
        visual = FashionCLIPVisualEncoder(config.resolved_encoder_model_name, trainable=config.encoder_trainable)
        textual = FashionCLIPTextEncoder(config.resolved_encoder_model_name, trainable=config.encoder_trainable)
        visual_dim, text_dim = visual.output_dim, textual.output_dim
    else:
        visual = MarqoFashionSigLIPVisualEncoder(config.resolved_encoder_model_name)
        textual = MarqoFashionSigLIPTextEncoder(config.resolved_encoder_model_name, backbone=visual.backbone)
        visual_dim, text_dim = visual.output_dim, textual.output_dim

    llm = Qwen3LLM(config.llm_model_name, cache_dir=config.llm_cache, dtype=llm_dtype, trainable=config.llm_trainable)
    llm_dim = llm.output_dim
    if config.category_embedding_dim != llm_dim:
        raise ValueError(
            f"category_embedding_dim ({config.category_embedding_dim}) must match "
            f"the LLM hidden size ({llm_dim})"
        )
    if config.lora_enabled:
        llm.enable_lora(
            rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout,
            gradient_checkpointing=config.gradient_checkpointing, target_modules=config.lora_target_modules,
        )

    seed_mlp = MLP(visual_dim, llm_dim, config.seed_hidden_dims)
    item_mlp = MLP(visual_dim + text_dim, retrieval_dim, config.item_hidden_dims)
    retrieval_heads = [MLP(llm_dim, retrieval_dim, config.retrieval_hidden_dims) for _ in OUTFIT_CATEGORIES]
    category_heads = [MLP(llm_dim, 1, config.category_hidden_dims) for _ in OUTFIT_CATEGORIES]
    fine_grained_heads = [
        MLP(llm_dim, max(1, len(labels)), config.fine_grained_hidden_dims)
        for labels in subcategories
    ]
    generator = SeedToOutfitModel(
        llm, CategoryQueryEmbeddings(config.category_embedding_dim),
        seed_encoder=OutfitSeedEncoder(visual, seed_mlp),
        retrieval_heads=retrieval_heads, category_heads=category_heads,
        fine_grained_heads=fine_grained_heads,
    )
    item_encoder = MultimodalEncoder(visual, textual, item_mlp)
    compatibility = OutfitCompatibility.from_checkpoint(
        cp_repo, cp_checkpoint, retrieval_dim,
    )
    return Text2OutfitModel(
        generator, item_encoder, compatibility, subcategories, config=config,
    ).to(selected_device)
