"""Contract for contextualizing the eleven external outfit category queries."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch import Tensor, nn

from data.categories import OUTFIT_CATEGORIES


class LLM(nn.Module, ABC):
    """Map prompts and category queries to their final hidden states."""

    categories = OUTFIT_CATEGORIES

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Hidden width, also required for the input category embeddings."""

    @abstractmethod
    def forward(
        self,
        prompts: str | Sequence[str],
        category_embeddings: Tensor,
        *,
        seed_embeddings: Tensor | None = None,
    ) -> Tensor:
        """Return only raw category hidden states, shaped ``[B, 11, output_dim]``.

        A single prompt accepts queries [11, H] or [1, 11, H]. A prompt batch
        accepts queries [B, 11, H]. Category order is ``OUTFIT_CATEGORIES``.
        Implementations preserve gradients, without pooling or L2 normalization.
        Optional image seed embeddings [B, H] (or [H] for one prompt) are
        inserted before the category queries and are not returned.
        """


def _validate_prompts(prompts: str | Sequence[str]) -> list[str]:
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, Sequence) or not prompts:
        raise ValueError("prompts must be a non-empty string or sequence of strings")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain non-empty strings")
    return list(prompts)
