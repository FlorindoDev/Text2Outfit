"""Loss di presenza e sottocategoria per la composizione degli outfit."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from ._validation import floating_tensor, presence_mask


def category_classification_loss(logits: Tensor, category_targets: Tensor) -> Tensor:
    """Cross-entropy binaria sui logits [B, K]: somma sulle categorie, media sul batch."""
    logits = floating_tensor(logits, "category_logits", ndim=2)
    present = presence_mask(category_targets, logits.shape, logits.device)
    losses = F.binary_cross_entropy_with_logits(
        logits, present.to(dtype=logits.dtype), reduction="none",
    )
    return losses.sum(dim=1).mean()


def fine_grained_category_loss(
    logits: Sequence[Tensor],
    fine_grained_targets: Tensor,
    category_targets: Tensor,
) -> Tensor:
    """Cross-entropy su K teste [B, C_i] e target [B, K]; ignora le categorie assenti."""
    if not isinstance(fine_grained_targets, Tensor) or fine_grained_targets.dtype != torch.long:
        raise TypeError("fine_grained_targets must be a torch.long tensor")
    if fine_grained_targets.ndim != 2 or fine_grained_targets.numel() == 0:
        raise ValueError("fine_grained_targets must have non-empty shape [batch, categories]")
    batch_size, num_categories = fine_grained_targets.shape
    if not isinstance(logits, Sequence) or len(logits) != num_categories:
        raise ValueError("provide one fine-grained logits tensor per category")
    present = presence_mask(
        category_targets, fine_grained_targets.shape, fine_grained_targets.device,
    )

    category_losses = []
    for category_index, category_logits in enumerate(logits):
        category_logits = floating_tensor(category_logits, "fine-grained logits", ndim=2)
        if category_logits.size(0) != batch_size:
            raise ValueError("fine-grained logits and targets must have the same batch size")
        if category_logits.device != fine_grained_targets.device:
            raise ValueError("fine-grained logits and targets must be on the same device")

        category_present = present[:, category_index]
        targets = fine_grained_targets[:, category_index]
        active_targets = targets[category_present]
        if bool(((active_targets < 0) | (active_targets >= category_logits.size(1))).any()):
            raise ValueError(f"fine-grained target out of range for category {category_index}")
        targets = targets.masked_fill(~category_present, -100)
        category_losses.append(F.cross_entropy(
            category_logits, targets, ignore_index=-100, reduction="sum",
        ))

    return sum(category_losses[1:], category_losses[0]) / batch_size


def composition_loss(
    category_logits: Tensor,
    fine_grained_logits: Sequence[Tensor],
    category_targets: Tensor,
    fine_grained_targets: Tensor,
) -> Tensor:
    """Somma le loss di presenza e sottocategoria sugli stessi outfit e categorie."""
    return (
        category_classification_loss(category_logits, category_targets)
        + fine_grained_category_loss(fine_grained_logits, fine_grained_targets, category_targets)
    )
