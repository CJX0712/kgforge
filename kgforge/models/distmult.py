"""DistMult (Yang et al., 2015) — bilinear diagonal embedding.

score(h,r,t) = <h, r, t> = sum_i h_i r_i t_i  (larger = more plausible).

NOTE: DistMult is symmetric in h and t, so it cannot model antisymmetric
relations; this structural blind-spot is exactly what gives KGFuse its
complementary signal over TransE / RotatE.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from .base import BaseKGEModel


class DistMult(BaseKGEModel):
    name = "distmult"
    complex_like = False

    def _init_embeddings(self, rng: np.ndarray, n_ent: int, n_rel: int) -> None:  # type: ignore[override]
        s = 1.0 / (self.dim**0.5)
        self.ent_emb = rng.standard_normal((n_ent, self.dim)).astype(np.float64) * s
        self.rel_emb = rng.standard_normal((n_rel, self.dim)).astype(np.float64) * s

    def score_and_grad(self, H, R, T):
        h = self.ent_emb[H]
        r = self.rel_emb[R]
        t = self.ent_emb[T]
        s = np.sum(h * r * t, axis=1)
        gH = r * t
        gR = h * t
        gT = h * r
        return s, gH, gR, gT
