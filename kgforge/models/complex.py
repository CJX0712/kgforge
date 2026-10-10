"""ComplEx (Trouillon et al., 2016) — complex bilinear embedding.

score(h,r,t) = Re(<h, r, conj(t)>)  (larger = more plausible).
Implemented with an explicit real (2d)-parameterisation so gradients stay real
and fully deterministic. Strong on symmetric relations.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from .base import BaseKGEModel


class ComplEx(BaseKGEModel):
    name = "complex"
    complex_like = True

    def _init_embeddings(self, rng, n_ent: int, n_rel: int) -> None:
        s = 1.0 / (self.dim**0.5)
        self.ent_emb = rng.standard_normal((n_ent, 2 * self.dim)).astype(np.float64) * s
        self.rel_emb = rng.standard_normal((n_rel, 2 * self.dim)).astype(np.float64) * s

    def score_and_grad(self, H, R, T):
        h = self.ent_emb[H]
        r = self.rel_emb[R]
        t = self.ent_emb[T]
        d = self.dim
        ahr, ahi = h[:, :d], h[:, d:]
        brr, bri = r[:, :d], r[:, d:]
        ctr, cti = t[:, :d], t[:, d:]

        A = ahr * brr - ahi * bri
        B_ = ahr * bri + ahi * brr
        s = np.sum(A * ctr + B_ * cti, axis=1)

        g_ahr = brr * ctr + bri * cti
        g_ahi = -bri * ctr + brr * cti
        g_brr = ahr * ctr + ahi * cti
        g_bri = -ahi * ctr + ahr * cti
        g_ctr = A
        g_cti = B_

        gH = np.concatenate([g_ahr, g_ahi], axis=1)
        gR = np.concatenate([g_brr, g_bri], axis=1)
        gT = np.concatenate([g_ctr, g_cti], axis=1)
        return s, gH, gR, gT
