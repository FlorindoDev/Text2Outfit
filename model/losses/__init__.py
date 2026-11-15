"""Obiettivi di addestramento differenziabili per Text2Outfit."""

from .compatibility import compatibility_loss
from .composition import (
    category_classification_loss,
    composition_loss,
    fine_grained_category_loss,
)
from .retrieval import outfit_retrieval_loss

__all__ = [
    "category_classification_loss",
    "compatibility_loss",
    "composition_loss",
    "fine_grained_category_loss",
    "outfit_retrieval_loss",
]
