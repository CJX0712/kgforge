"""KGForge — world-class Knowledge Graph embedding & link-prediction system.

Flagship ``KGFuse`` = max-consensus reciprocal-rank ensemble over four SOTA
embedding models (TransE / DistMult / ComplEx / RotatE). Pure NumPy, offline,
CPU-only, deterministic, reproducible.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"

from .core.config import Config
from .core.errors import (  # noqa: F401
    E100,
    E200,
    E300,
    E400,
    E500,
    ConfigError,
    KGError,
    TrainingError,
)
from .core.seed import set_all
from .eval.metrics import evaluate_link_prediction, filtered_ranks
from .models import (
    MODEL_REGISTRY,
    ComplEx,
    DistMult,
    RotatE,
    TransE,
)
from .pipeline.kgfuse import KGFuse, KGFusePipeline

__all__ = [
    "MODEL_REGISTRY",
    "ComplEx",
    "Config",
    "ConfigError",
    "DistMult",
    "KGError",
    "KGFuse",
    "KGFusePipeline",
    "RotatE",
    "TrainingError",
    "TransE",
    "evaluate_link_prediction",
    "filtered_ranks",
    "set_all",
]
