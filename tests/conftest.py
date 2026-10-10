"""pytest bootstrap: pin threads, fix sys.path, share a fast test config.

Author: 晨星
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kgforge.core.config import Config  # noqa: E402


def small_cfg(**kw) -> Config:
    """Tiny config so the whole suite stays fast (CPU-friendly)."""
    base = dict(
        dim=8,
        epochs=5,
        batch_size=32,
        n_entities=60,
        n_relations=4,
        n_train=200,
        n_valid=40,
        n_test=40,
        difficulty="medium",
    )
    base.update(kw)
    return Config(**base)
