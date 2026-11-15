"""Loss InfoNCE sugli embedding predetti, con confronti nella stessa categoria."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from ._validation import floating_tensor, presence_mask


def outfit_retrieval_loss(
    predicted: Tensor,
    positive: Tensor,
    negatives: Tensor,
    category_targets: Tensor,
    *,
    temperature: float = 0.07,
) -> Tensor:
    """InfoNCE con predizioni e positivi [B, K, D] e negativi [B, K, N, D]."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a positive finite number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    predicted = floating_tensor(predicted, "predicted", ndim=3)
    positive = floating_tensor(positive, "positive", ndim=3)
    negatives = floating_tensor(negatives, "negatives", ndim=4)
    if positive.shape != predicted.shape:
        raise ValueError("predicted and positive must share shape [batch, categories, features]")
    if negatives.shape[:2] != predicted.shape[:2] or negatives.size(-1) != predicted.size(-1):
        raise ValueError("negatives must have shape [batch, categories, negatives, features]")
    if positive.device != predicted.device or negatives.device != predicted.device:
        raise ValueError("all embeddings must be on the same device")
    present = presence_mask(category_targets, predicted.shape[:2], predicted.device)

    if not bool(present.any()):
        return predicted[present].sum() + positive[present].sum() + negatives[present].sum()

    queries = F.normalize(predicted[present], dim=-1)
    positive_items = F.normalize(positive[present], dim=-1)
    negative_items = F.normalize(negatives[present], dim=-1)
    positive_scores = (queries * positive_items).sum(dim=-1, keepdim=True)
    negative_scores = (queries.unsqueeze(1) * negative_items).sum(dim=-1)
    scores = torch.cat((positive_scores, negative_scores), dim=-1) / temperature
    labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, labels, reduction="sum") / predicted.size(0)
