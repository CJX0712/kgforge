"""Error taxonomy with stable E-codes (E100..E500).

Author: 晨星
"""

from __future__ import annotations


class KGError(Exception):
    """Base class for all KGForge errors."""


class ConfigError(KGError):
    """Invalid configuration / schema violation (E100)."""


class DataError(KGError):
    """Data loading / generation problem (E200)."""


class ModelError(KGError):
    """Model construction / forward problem (E300)."""


class TrainingError(KGError):
    """Training divergence / invariant violation (E400)."""


class EvalError(KGError):
    """Evaluation / metric problem (E500)."""


# Stable code registry (kept for tooling / future structured logging).
E100 = "E100:config"
E200 = "E200:data"
E300 = "E300:model"
E400 = "E400:training"
E500 = "E500:eval"
