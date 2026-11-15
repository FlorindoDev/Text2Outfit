"""Download OutfitTransformer assets and expose its frozen compatibility scorer."""

from .download import resolve_cp_assets
from .scorer import OutfitCompatibility, load_outfit_transformer

__all__ = ["OutfitCompatibility", "load_outfit_transformer", "resolve_cp_assets"]
