"""Learnable external IMG_cat queries for the Text2Outfit categories."""

from torch import Tensor, nn


OUTFIT_CATEGORIES = (
    "tops",
    "bottoms",
    "all-body",
    "outerwear",
    "shoes",
    "bags",
    "jewelry",
    "accessories",
    "sunglasses",
    "hats",
    "scarves",
)


class CategoryQueryEmbeddings(nn.Module):
    """Return one trainable query per category in a fixed, public order."""

    categories = OUTFIT_CATEGORIES

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if (
            not isinstance(embedding_dim, int)
            or isinstance(embedding_dim, bool)
            or embedding_dim <= 0
        ):
            raise ValueError("embedding_dim must be a positive integer")
        self.embeddings = nn.Embedding(
            num_embeddings=len(self.categories),
            embedding_dim=embedding_dim,
        )

    @property
    def output_dim(self) -> int:
        """Number of features in each category query."""
        return self.embeddings.embedding_dim

    def forward(self, batch_size: int = 1) -> Tensor:
        """Share the 11 category queries across a batch: ``[batch_size, 11, D]``."""
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")

        category_embeddings = self.embeddings.weight
        num_categories, embedding_dim = category_embeddings.shape
        queries_with_batch_dim = category_embeddings.unsqueeze(0)

        # Reuse the learned vectors for every batch item without copying their data.
        return queries_with_batch_dim.expand(batch_size, num_categories, embedding_dim)
