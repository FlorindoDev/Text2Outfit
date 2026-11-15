"""Defaults for the model architecture, backbones, LoRA and losses."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

FASHIONCLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"
SIGLIP_MODEL_NAME = "Marqo/marqo-fashionSigLIP"
CP_CHECKPOINT_RELATIVE_PATH = Path("checkpoints/nondisjoint/cp_precomputed-flashionclip/best.pt")


@dataclass(frozen=True)
class ModelConfig:
    """Backbones, category embeddings, MLP dimensions, LoRA and losses."""

    # --- LLM ---
    llm_model_name: str = "Qwen/Qwen3-Embedding-8B"
    llm_dtype: str = "auto"  # Auto: BF16 on a compatible CUDA device, otherwise FP32.
    llm_trainable: bool = False
    llm_cache: Path | None = Path(".cache/llm")

    # --- Embedding di categoria: 11 vettori apprendibili [11, H] ---
    # H deve coincidere con hidden_size del checkpoint LLM (4096 per Qwen3 Embedding 8B).
    category_embedding_dim: int = 4096

    # --- Encoder immagine/testo: dimensioni V/T lette dal checkpoint o dalla cache ---
    # FashionCLIP: V=T=512. Marqo FashionSigLIP: V=T=768.
    encoder: str = "fashionclip"
    encoder_model_name: str | None = None
    encoder_trainable: bool = False

    # --- MLP: ogni tupla *_hidden_dims contiene SOLO i layer nascosti ---
    # (512,) = un layer nascosto; (1024, 512) = due; () = proiezione lineare.
    # Nei percorsi seguenti: ingresso -> layer nascosti -> uscita.

    # Seed immagine (1 MLP): V -> seed_hidden_dims -> H. Default: 512 -> 512 -> 4096.
    seed_mode: str = "image"  # "text" inserisce la descrizione nel prompt.
    seed_hidden_dims: tuple[int, ...] = (512,)

    # Retrieval (11 MLP): H -> retrieval_hidden_dims -> D. Default: 4096 -> 512 -> 1024.
    # D e' la dimensione condivisa dalle predizioni retrieval e dagli item del catalogo.
    retrieval_dim: int = 1024
    retrieval_hidden_dims: tuple[int, ...] = (512,)

    # Presenza categoria (11 MLP): H -> category_hidden_dims -> 1. Default: 4096 -> 512 -> 1.
    category_hidden_dims: tuple[int, ...] = (512,)

    # Sottocategoria (11 MLP): H -> fine_grained_hidden_dims -> C_i. Default: 4096 -> 512 -> C_i.
    # C_i e' il numero di sottocategorie della categoria i, letto dalla tassonomia.
    fine_grained_hidden_dims: tuple[int, ...] = (512,)

    # Item del catalogo (1 MLP): (V + T) -> item_hidden_dims -> D. Default: 1024 -> 512 -> 1024.
    item_hidden_dims: tuple[int, ...] = (512,)

    # --- CP congelato: riceve direttamente gli embedding retrieval, senza adattatore ---
    # retrieval_dim deve coincidere con model_dim del checkpoint CP (1024 nel default).
    cp_repo: Path = Path(".cache/outfit_transformer")
    cp_checkpoint: Path | None = None
    cp_source_url: str = "https://codeload.github.com/FlorindoDev/OutfitTransformer/zip/refs/heads/main-branch"
    cp_weights_repo: str = "FlorindoDev/OutfitTransformer-weights"
    cp_weights_file: str = CP_CHECKPOINT_RELATIVE_PATH.as_posix()
    cp_cache: Path = Path(".cache/huggingface-cp")

    # --- LoRA ---
    # LoRA updates these projections while keeping original LLM weights frozen.
    lora_enabled: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj")
    gradient_checkpointing: bool = True

    # --- Loss ---
    temperature: float = 0.07
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # Retrieval, CP, composition.

    @property
    def resolved_encoder_model_name(self) -> str:
        if self.encoder_model_name is not None:
            return self.encoder_model_name
        return FASHIONCLIP_MODEL_NAME if self.encoder == "fashionclip" else SIGLIP_MODEL_NAME

    @property
    def cp_checkpoint_path(self) -> Path:
        return self.cp_checkpoint or self.cp_repo / self.cp_weights_file

    def with_hidden_dim(self, value: int | None) -> ModelConfig:
        """Apply the existing --hidden-dim shortcut to all five MLP families."""
        if value is None:
            return self
        return replace(
            self, seed_hidden_dims=(value,), retrieval_hidden_dims=(value,),
            category_hidden_dims=(value,), fine_grained_hidden_dims=(value,),
            item_hidden_dims=(value,),
        )

    def validate(self) -> None:
        if not self.cp_source_url.strip() or not self.cp_weights_repo.strip():
            raise ValueError("CP source URL and weights repository cannot be empty")
        weights_file = PurePosixPath(self.cp_weights_file)
        if weights_file.is_absolute() or Path(self.cp_weights_file).drive or ".." in weights_file.parts or weights_file.name != "best.pt" or "\\" in self.cp_weights_file:
            raise ValueError("cp_weights_file must be a relative Hub path ending in best.pt")
        if self.encoder not in ("fashionclip", "siglip") or self.seed_mode not in ("image", "text"):
            raise ValueError("encoder must be fashionclip/siglip and seed_mode must be image/text")
        if not self.llm_model_name.strip() or not self.resolved_encoder_model_name.strip():
            raise ValueError("backbone model names cannot be empty")
        if self.llm_dtype not in ("auto", "float32", "bfloat16"):
            raise ValueError("llm_dtype must be auto, float32 or bfloat16")
        if self.lora_enabled and self.llm_trainable:
            raise ValueError("choose LoRA or full LLM fine-tuning, not both")
        if self.encoder_trainable:
            raise ValueError("the complete outfit model requires frozen visual/text encoders")
        if type(self.category_embedding_dim) is not int or self.category_embedding_dim <= 0:
            raise ValueError("category_embedding_dim must be a positive integer")
        if any(type(value) is not int or value <= 0 for value in (self.retrieval_dim, self.lora_rank, self.lora_alpha)):
            raise ValueError("retrieval_dim and LoRA rank/alpha must be positive integers")
        for name, dimensions in (
            ("seed", self.seed_hidden_dims), ("retrieval", self.retrieval_hidden_dims),
            ("category", self.category_hidden_dims), ("fine_grained", self.fine_grained_hidden_dims),
            ("item", self.item_hidden_dims),
        ):
            if any(type(width) is not int or width <= 0 for width in dimensions):
                raise ValueError(f"{name} hidden layer widths must be positive integers")
        if not self.lora_target_modules or any(not name.strip() for name in self.lora_target_modules):
            raise ValueError("provide non-empty LoRA target module names")
        if not 0 <= self.lora_dropout < 1 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("invalid LoRA dropout or retrieval temperature")
        if len(self.loss_weights) != 3 or any(not math.isfinite(w) or w < 0 for w in self.loss_weights) or not any(self.loss_weights):
            raise ValueError("provide three finite nonnegative loss weights, with at least one positive")


DEFAULT_MODEL_CONFIG = ModelConfig()
