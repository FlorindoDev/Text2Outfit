"""Image/text representations and trainable embedding networks."""

from .outfit.category_query_embeddings import OUTFIT_CATEGORIES, CategoryQueryEmbeddings
from .config import DEFAULT_MODEL_CONFIG, ModelConfig
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
from .outfit.outfit_seed_encoder import OutfitSeedEncoder
from .outfit.seed_to_outfit import SeedToOutfitModel, SeedToOutfitOutput
from .compatibility import OutfitCompatibility
from .outfit.text2outfit import Text2OutfitModel
from .outfit.build import build_model

__all__ = [
    "DEFAULT_MODEL_CONFIG",
    "ModelConfig",
    "Text2OutfitModel",
    "OutfitCompatibility",
    "build_model",
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
