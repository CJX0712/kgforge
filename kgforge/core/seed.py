"""Deterministic global seeding.

A single entry point ``set_all(seed)`` fixes every randomness source used by
KGForge and pins BLAS/OpenMP threads to 1 so that two runs with the same seed
produce **bit-identical** benchmark numbers (a hard DoD gate).

Author: 晨星
"""

from __future__ import annotations

import os
import random
from typing import Final

import numpy as np

# Pin threads BEFORE numpy import time matters; we set env vars defensively and
# also configure the threadpools at runtime so behaviour is stable on re-import.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")


def set_all(seed: int) -> np.random.Generator:
    """Seed numpy + stdlib random and pin thread pools. Returns one Generator."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    try:  # pragma: no cover - best effort, runtime thread pinning
        import threadpoolctl  # type: ignore

        threadpoolctl.threadpool_limits(limits=1)
    except Exception:
        pass

    random.seed(seed)
    np.random.seed(seed)
    try:
        _rng = np.random.default_rng(seed)
    except Exception:  # numpy < 1.17 fallback
        _rng = np.random.RandomState(seed)  # type: ignore
    return _rng


# Deterministic float tolerance used by invariant tests.
DETERMINISM_TOL: Final[float] = 0.0  # bit-identical expected for core metrics
