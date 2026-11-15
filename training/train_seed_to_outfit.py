"""Run with python -m training.train_seed_to_outfit --help."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import build_fashion_clip_transform, build_marqo_fashion_siglip_transform
from data.polyvore.catalog import PolyvoreCatalog
from data.polyvore.download import PolyvoreSplit, PolyvoreTask, download_polyvore_resources
from data.polyvore.outfit_dataset import PolyvoreOutfitDataset
from model import DEFAULT_MODEL_CONFIG, build_model

from .config import DEFAULT_TRAINING_CONFIG
from .data import PrecomputedItemFeatures, SeedOutfitDataset, collate_seed_outfits, subcategory_vocabulary
from .trainer import train


def _cached_only_image(_image):
    raise RuntimeError("an image was decoded despite enabling precomputed features")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train seed-to-outfit with LoRA and frozen OutfitTransformer CP.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_TRAINING_CONFIG.dataset_root)
    parser.add_argument("--dataset-cache", type=Path, default=DEFAULT_TRAINING_CONFIG.dataset_cache)
    parser.add_argument("--subset", choices=("nondisjoint", "disjoint"), default=DEFAULT_TRAINING_CONFIG.subset)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_TRAINING_CONFIG.embedding_root, help="cache directory containing SUBSET/train and SUBSET/validation")
    parser.add_argument("--encoder", choices=("fashionclip", "siglip"), default=DEFAULT_MODEL_CONFIG.encoder)
    parser.add_argument("--encoder-model", default=DEFAULT_MODEL_CONFIG.encoder_model_name, help="override the visual/text checkpoint name")
    parser.add_argument("--llm", default=DEFAULT_MODEL_CONFIG.llm_model_name)
    parser.add_argument("--llm-cache", type=Path, default=DEFAULT_MODEL_CONFIG.llm_cache)
    parser.add_argument("--category-embedding-dim", type=int, default=DEFAULT_MODEL_CONFIG.category_embedding_dim,
                        help="category query width; must match the selected LLM hidden size")
    parser.add_argument("--cp-repo", type=Path, default=DEFAULT_MODEL_CONFIG.cp_repo)
    parser.add_argument("--cp-checkpoint", type=Path, default=DEFAULT_MODEL_CONFIG.cp_checkpoint, help="path to the trained CP checkpoint")
    parser.add_argument("--cp-source-url", default=DEFAULT_MODEL_CONFIG.cp_source_url)
    parser.add_argument("--cp-weights-repo", default=DEFAULT_MODEL_CONFIG.cp_weights_repo)
    parser.add_argument("--cp-weights-file", default=DEFAULT_MODEL_CONFIG.cp_weights_file, help="Hub path of the selected best.pt")
    parser.add_argument("--cp-cache", type=Path, default=DEFAULT_MODEL_CONFIG.cp_cache)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINING_CONFIG.output_dir)
    parser.add_argument("--resume", type=Path, default=DEFAULT_TRAINING_CONFIG.resume)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_CONFIG.batch_size)
    parser.add_argument("--accumulation-steps", type=int, default=DEFAULT_TRAINING_CONFIG.gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAINING_CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_TRAINING_CONFIG.weight_decay)
    parser.add_argument("--negatives-per-item", type=int, default=DEFAULT_TRAINING_CONFIG.negatives_per_item)
    parser.add_argument("--retrieval-dim", type=int, default=DEFAULT_MODEL_CONFIG.retrieval_dim)
    parser.add_argument("--hidden-dim", type=int, default=None, help="override every MLP's hidden layers with one layer of this width")
    parser.add_argument("--seed-mode", choices=("image", "text"), default=DEFAULT_MODEL_CONFIG.seed_mode)
    parser.add_argument("--prompt", default=DEFAULT_TRAINING_CONFIG.prompt)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_TRAINING_CONFIG.prompts, help="JSON object mapping outfit IDs to prompts for both splits")
    parser.add_argument("--loss-weights", type=float, nargs=3, default=DEFAULT_MODEL_CONFIG.loss_weights, metavar=("RET", "CP", "COMP"))
    parser.add_argument("--temperature", type=float, default=DEFAULT_MODEL_CONFIG.temperature)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_MODEL_CONFIG.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_MODEL_CONFIG.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_MODEL_CONFIG.lora_dropout)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=DEFAULT_MODEL_CONFIG.gradient_checkpointing)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_CONFIG.seed)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_TRAINING_CONFIG.num_workers)
    parser.add_argument("--device", default=DEFAULT_TRAINING_CONFIG.device)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_config = replace(
        DEFAULT_MODEL_CONFIG, llm_model_name=args.llm, llm_cache=args.llm_cache,
        category_embedding_dim=args.category_embedding_dim,
        encoder=args.encoder, encoder_model_name=args.encoder_model,
        cp_repo=args.cp_repo, cp_checkpoint=args.cp_checkpoint,
        cp_source_url=args.cp_source_url, cp_weights_repo=args.cp_weights_repo,
        cp_weights_file=args.cp_weights_file, cp_cache=args.cp_cache,
        retrieval_dim=args.retrieval_dim, seed_mode=args.seed_mode,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        loss_weights=tuple(args.loss_weights), temperature=args.temperature,
    ).with_hidden_dim(args.hidden_dim)
    training_config = replace(
        DEFAULT_TRAINING_CONFIG, output_dir=args.output_dir, epochs=args.epochs, resume=args.resume,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.accumulation_steps, batch_size=args.batch_size,
        negatives_per_item=args.negatives_per_item, seed=args.seed, num_workers=args.num_workers,
        device=args.device, dataset_root=args.dataset_root, dataset_cache=args.dataset_cache,
        subset=args.subset, embedding_root=args.embedding_root, prompt=args.prompt, prompts=args.prompts,
    )
    model_config.validate()
    training_config.validate()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_name = model_config.resolved_encoder_model_name
    precomputed_modality_dim = None
    caches = {}
    if args.embedding_root is not None:
        for split in ("train", "validation"):
            cache = PrecomputedItemFeatures(args.embedding_root / args.subset / split / "manifest.json")
            if cache.metadata.get("subset") != args.subset or cache.metadata.get("split") != split:
                raise ValueError(f"cache subset/split mismatch: {cache.path}")
            if cache.metadata["encoder"]["model_name"] != model_name:
                raise ValueError("cached features were produced by a different encoder")
            caches[split] = cache
        if caches["train"].metadata.get("model_fingerprint") != caches["validation"].metadata.get("model_fingerprint"):
            raise ValueError("train and validation cache encoder fingerprints differ")
        transform = _cached_only_image
        precomputed_modality_dim = caches["train"].embedding_dim // 2
    elif args.encoder == "fashionclip":
        transform = build_fashion_clip_transform(model_name)
    else:
        transform = build_marqo_fashion_siglip_transform(model_name)

    catalogs, datasets = {}, {}
    prompts = json.loads(args.prompts.read_text(encoding="utf-8")) if args.prompts else None
    for split in (PolyvoreSplit.TRAIN, PolyvoreSplit.VALIDATION):
        resources = download_polyvore_resources(
            task=PolyvoreTask.OUTFITS, variant=args.subset, split=split,
            dataset_root=args.dataset_root, cache_dir=args.dataset_cache,
        )
        if resources.item_rows is None or resources.metadata_path is None or resources.outfits_path is None or resources.compatibility_path is None:
            raise ValueError("incomplete Polyvore outfit resources")
        catalog = PolyvoreCatalog(resources.item_rows, resources.metadata_path, transform)
        cache = caches.get(split.value)
        if cache is not None and set(catalog.item_ids).difference(cache):
            raise ValueError(f"precomputed cache is missing catalog items for {split.value}")
        outfits = PolyvoreOutfitDataset(
            catalog, resources.outfits_path, resources.compatibility_path,
            negatives_per_item=args.negatives_per_item, precomputed_embeddings=cache,
            positive_only=True, seed=args.seed if split == PolyvoreSplit.VALIDATION else None,
        )
        catalogs[split.value] = catalog
        datasets[split.value] = SeedOutfitDataset(outfits, training=split == PolyvoreSplit.TRAIN, prompt=args.prompt, prompts=prompts)
    vocabulary = subcategory_vocabulary(*catalogs.values())
    model = build_model(vocabulary, model_config, precomputed_modality_dim=precomputed_modality_dim, device=args.device)
    logging.getLogger(__name__).info("\n%s", model.architecture_summary())
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_seed_outfits, num_workers=args.num_workers,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size,
                                   collate_fn=collate_seed_outfits, num_workers=args.num_workers)
    metadata = {key: value for key, value in vars(args).items() if key not in ("resume", "output_dir", "epochs")}
    metadata["cache_fingerprints"] = {split: cache.metadata.get("model_fingerprint") for split, cache in caches.items()}
    train(model, train_loader, validation_loader, training_config, run_config=metadata)


if __name__ == "__main__":
    main()
