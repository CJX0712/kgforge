"""TransE (Bordes et al., 2013) — translational embedding.

score(h,r,t) = -||h + r - t||^2  (larger = more plausible).

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from .base import BaseKGEModel


class TransE(BaseKGEModel):
    name = "transe"
    complex_like = False

    def _init_embeddings(self, rng: np.ndarray, n_ent: int, n_rel: int) -> None:  # type: ignore[override]
        s = 1.0 / (self.dim**0.5)
        self.ent_emb = rng.standard_normal((n_ent, self.dim)).astype(np.float64) * s
        self.rel_emb = rng.standard_normal((n_rel, self.dim)).astype(np.float64) * s

    def score_and_grad(self, H, R, T):
        h = self.ent_emb[H]
        r = self.rel_emb[R]
        t = self.ent_emb[T]
        D = h + r - t
        s = -np.sum(D * D, axis=1)
        gH = -2.0 * D
        gR = -2.0 * D
        gT = 2.0 * D
        return s, gH, gR, gT
