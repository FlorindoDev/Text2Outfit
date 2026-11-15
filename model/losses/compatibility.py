"""Massimizzazione dello score di un modello di compatibilità già addestrato."""

from torch import Tensor

from ._validation import floating_tensor


def compatibility_loss(scores: Tensor) -> Tensor:
    """Restituisce la media negativa degli score in [0, 1], con forma [B] o [B, 1]."""
    if isinstance(scores, Tensor) and scores.ndim == 2 and scores.size(1) == 1:
        scores = scores.squeeze(1)
    scores = floating_tensor(scores, "compatibility scores", ndim=1)
    if bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError("compatibility scores must be in [0, 1], not logits or percentages")
    return -scores.mean()
