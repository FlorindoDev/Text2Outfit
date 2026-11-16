"""Defaults for the training loop, data loading and run outputs."""

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer, data loading, reproducibility and output defaults."""

    output_dir: Path = Path("checkpoints/seed_to_outfit")
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    resume: Path | None = None
    batch_size: int = 128
    negatives_per_item: int = 4
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    dataset_root: Path = Path("datasets/polyvore-outfits")
    dataset_cache: Path = Path(".cache/datasets")
    subset: str = "nondisjoint"
    embedding_root: Path | None = None
    prompt: str = "Complete a compatible outfit using this seed item."
    prompts: Path | None = None

    def validate(self) -> None:
        if self.epochs <= 0 or self.gradient_accumulation_steps <= 0 or self.batch_size <= 0 or self.negatives_per_item <= 0:
            raise ValueError("epochs, batch_size, negatives and accumulation steps must be positive")
        if self.num_workers < 0 or self.seed < 0 or self.subset not in ("nondisjoint", "disjoint"):
            raise ValueError("invalid workers, seed or dataset subset")
        if not self.prompt.strip():
            raise ValueError("default prompt cannot be empty")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive and finite")


DEFAULT_TRAINING_CONFIG = TrainingConfig()
