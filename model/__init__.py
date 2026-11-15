"""Image/text representations and trainable embedding networks."""

from .category_query_embeddings import OUTFIT_CATEGORIES, CategoryQueryEmbeddings
from .llm import LLM, Qwen3LLM
from .losses import (
    category_classification_loss,
    compatibility_loss,
    composition_loss,
    fine_grained_category_loss,
    outfit_retrieval_loss,
)
from .mlp import MLP

from .multimodal_representation import (
    FashionCLIPTextEncoder,
    FashionCLIPVisualEncoder,
    MarqoFashionSigLIPTextEncoder,
    MarqoFashionSigLIPVisualEncoder,
    MultimodalEncoder,
    TextEncoder,
    VisualEncoder,
)
from .outfit_seed_encoder import OutfitSeedEncoder
from .seed_to_outfit import SeedToOutfitModel, SeedToOutfitOutput

__all__ = [
    "OUTFIT_CATEGORIES",
    "CategoryQueryEmbeddings",
    "LLM",
    "Qwen3LLM",
    "FashionCLIPTextEncoder",
    "FashionCLIPVisualEncoder",
    "MarqoFashionSigLIPTextEncoder",
    "MarqoFashionSigLIPVisualEncoder",
    "MLP",
    "MultimodalEncoder",
    "OutfitSeedEncoder",
    "SeedToOutfitModel",
    "SeedToOutfitOutput",
    "TextEncoder",
    "VisualEncoder",
    "category_classification_loss",
    "compatibility_loss",
    "composition_loss",
    "fine_grained_category_loss",
    "outfit_retrieval_loss",
]
