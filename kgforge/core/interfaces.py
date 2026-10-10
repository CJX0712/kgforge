"""Structural-typing interfaces (Protocol) for pluggable components.

Author: 晨星
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import KnowledgeGraph


class KGEmbedder(Protocol):
    """Contract every embedding model implements.

    Score convention is **unified**: ``score`` returns a real number where
    *larger = more plausible*. This lets the fusion layer aggregate models
    with a single, fair comparison.
    """

    name: str

    def fit(self, kg: KnowledgeGraph) -> KGEmbedder: ...
    def score(self, heads: np.ndarray, relations: np.ndarray, tails: np.ndarray) -> np.ndarray: ...
    def predict_scores_head(self, relations: np.ndarray, tails: np.ndarray) -> np.ndarray: ...
    def predict_scores_tail(self, heads: np.ndarray, relations: np.ndarray) -> np.ndarray: ...


class FusionStrategy(Protocol):
    """Contract for the link-prediction fusion flagship."""

    name: str

    def fuse(self, per_model_ranks: list[np.ndarray]) -> np.ndarray: ...
