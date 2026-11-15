"""Load and run the frozen OutfitTransformer scorer without an input adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def load_outfit_transformer(repo: str | Path) -> Any:
    """Import the other repository under its own name, without shadowing model/."""
    directory = Path(repo).resolve() / "model"
    name = "_text2outfit_cp_" + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    if name in sys.modules:
        return sys.modules[name]
    source = directory / "__init__.py"
    if not source.is_file():
        raise FileNotFoundError(f"OutfitTransformer model package not found: {source}")
    spec = importlib.util.spec_from_file_location(name, source, submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import OutfitTransformer from {directory}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(package)
    finally:
        sys.dont_write_bytecode = previous
    return package


class OutfitCompatibility(nn.Module):
    """Score generated items in the GT composition without projecting or detaching inputs."""

    def __init__(
        self, cp: nn.Module, batcher: nn.Module,
        *, context_encoder: nn.Module | None = None, legacy_tokens: bool = False,
        source: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.cp: Any = cp.requires_grad_(False).eval()
        self.batcher: Any = batcher.requires_grad_(False).eval()
        self.context_encoder = context_encoder
        if context_encoder is not None:
            context_encoder.requires_grad_(False).eval()
        self.legacy_tokens = legacy_tokens
        self.source = source or {}

    @property
    def input_dim(self) -> int:
        """Item embedding width required by the pretrained CP checkpoint."""
        return self.batcher.config.model_dim

    @classmethod
    def from_checkpoint(
        cls, repo: str | Path, checkpoint: str | Path, embedding_dim: int,
    ) -> OutfitCompatibility:
        package = load_outfit_transformer(repo)
        path = Path(checkpoint).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, dict) or "run_config" not in payload:
            raise ValueError("CP checkpoint must include model_state_dict and run_config")
        settings = payload["run_config"]["model"]
        if "modality_embedding_dim" not in settings:
            raise ValueError("CP checkpoint uses an unsupported older architecture; use the CP clip best.pt")
        config = package.TransformerConfig(**settings)
        if embedding_dim != config.model_dim:
            raise ValueError(
                f"retrieval_dim ({embedding_dim}) must match the CP checkpoint input "
                f"width ({config.model_dim}); no CP adapter is used"
            )
        cp = package.CompatibilityTransformer(config)
        batcher = package.OutfitEmbeddingBatcher(config)
        state = payload["model_state_dict"]
        if not state or any(key != "common.padding_embedding" and not key.startswith(("cp.", "common.encoder.")) for key in state):
            raise ValueError("unsupported CP checkpoint components; refusing to discard weights")
        legacy = "cp.encoder.norm.weight" in state
        if legacy:
            cp.encoder.norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_epsilon)
        cp.load_state_dict({key.removeprefix("cp."): value for key, value in state.items()
                            if key.startswith("cp.")}, strict=True)
        batcher.load_state_dict({"padding_embedding": state["common.padding_embedding"]}, strict=True)
        context = None
        context_state = {key.removeprefix("common.encoder."): value for key, value in state.items()
                         if key.startswith("common.encoder.")}
        if context_state:
            # Legacy clip checkpoints include the original common Transformer.
            from importlib import import_module
            builder = import_module(f"{package.__name__}.common.transformer_encoder")
            context = builder.build_transformer_encoder(config)
            if "norm.weight" in context_state:
                context.norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_epsilon)
            context.load_state_dict(context_state, strict=True)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return cls(
            cp, batcher,
            context_encoder=context, legacy_tokens=legacy,
            source={"repo": str(Path(repo).resolve()), "checkpoint": str(path),
                    "checkpoint_sha256": digest.hexdigest(), "model": settings},
        )

    def train(self, mode: bool = True) -> OutfitCompatibility:
        super().train(mode)
        self.cp.eval()
        self.batcher.eval()
        if self.context_encoder is not None:
            self.context_encoder.eval()
        return self

    def forward(self, embeddings: Tensor, present: Tensor) -> Tensor:
        if embeddings.ndim != 3 or present.shape != embeddings.shape[:2] or present.dtype != torch.bool:
            raise ValueError("CP expects embeddings [B, K, D] and a boolean presence mask [B, K]")
        if embeddings.size(-1) != self.input_dim:
            raise ValueError(f"CP item embeddings must have width {self.input_dim}")
        if bool((~present.any(dim=1)).any()):
            raise ValueError("CP outfits cannot be empty")
        if bool((present.sum(dim=1) > self.batcher.config.max_items).any()):
            raise ValueError("CP checkpoint max_items is smaller than the outfit composition")
        weights = next(self.cp.parameters())
        embeddings = embeddings.to(device=weights.device, dtype=weights.dtype)
        present = present.to(device=weights.device)
        batch = self.batcher([row[mask] for row, mask in zip(embeddings, present)])
        if self.context_encoder is not None:
            contextual = self.context_encoder(batch.item_embeddings, src_key_padding_mask=batch.padding_mask)
            padding = F.normalize(self.batcher.padding_embedding, dim=-1)
            batch = replace(batch, item_embeddings=torch.where(batch.padding_mask.unsqueeze(-1), padding, contextual))
        if not self.legacy_tokens:
            return self.cp(batch)
        # Old checkpoints used the raw CP token; current OutfitTransformer normalizes it.
        token = torch.cat((self.cp.task_embedding(), self.cp.predict_emb)).view(1, 1, -1)
        tokens = torch.cat((token.expand(embeddings.size(0), -1, -1), batch.item_embeddings), dim=1)
        mask = F.pad(batch.padding_mask, (1, 0), value=False)
        return self.cp.head(self.cp.encoder(tokens, src_key_padding_mask=mask)[:, 0])
