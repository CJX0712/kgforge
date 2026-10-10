"""Embedding models: TransE / DistMult / ComplEx / RotatE + registry.

Each model reuses a *published* SOTA scoring function (not invented here); the
novel contribution is the KGFuse fusion flagship in ``pipeline``.

Author: 晨星
"""

from .base import BaseKGEModel
from .complex import ComplEx
from .distmult import DistMult
from .rotate import RotatE
from .transe import TransE

MODEL_REGISTRY = {
    "transe": TransE,
    "distmult": DistMult,
    "complex": ComplEx,
    "rotate": RotatE,
}

__all__ = ["MODEL_REGISTRY", "BaseKGEModel", "ComplEx", "DistMult", "RotatE", "TransE"]
