"""RotatE (Sun et al., 2019) — relation as rotation in complex space.

score(h,r,t) = -||h ∘ r - t||^2  (larger = more plausible), where relations are
unit-modulus complex vectors. RotatE models composition (rotation addition) and
symmetric/antisymmetric/inverse patterns; it is a papers-with-code SOTA for
link prediction. Relations are kept on the unit circle after every step.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from .base import BaseKGEModel


class RotatE(BaseKGEModel):
    name = "rotate"
    complex_like = True

    def _init_embeddings(self, rng, n_ent: int, n_rel: int) -> None:
        s = 1.0 / (self.dim**0.5)
        self.ent_emb = rng.standard_normal((n_ent, 2 * self.dim)).astype(np.float64) * s
        # relations on the unit circle: (cos θ, sin θ) per dimension
        angles = rng.uniform(0.0, 2.0 * np.pi, size=(n_rel, self.dim))
        rel = np.zeros((n_rel, 2 * self.dim), dtype=np.float64)
        # BLOCK layout [real(d), imag(d)] must match score_and_grad / _post_update
        rel[:, : self.dim] = np.cos(angles)
        rel[:, self.dim :] = np.sin(angles)
        self.rel_emb = rel

    def score_and_grad(self, H, R, T):
        h = self.ent_emb[H]
        r = self.rel_emb[R]
        t = self.ent_emb[T]
        d = self.dim
        ahr, ahi = h[:, :d], h[:, d:]
        brr, bri = r[:, :d], r[:, d:]
        ctr, cti = t[:, :d], t[:, d:]

        u = ahr * brr - ahi * bri - ctr
        v = ahr * bri + ahi * brr - cti
        s = -np.sum(u * u + v * v, axis=1)

        g_ahr = -2.0 * (u * brr + v * bri)
        g_ahi = -2.0 * (-u * bri + v * brr)
        g_brr = -2.0 * (u * ahr + v * ahi)
        g_bri = -2.0 * (-u * ahi + v * ahr)
        g_ctr = 2.0 * u
        g_cti = 2.0 * v

        gH = np.concatenate([g_ahr, g_ahi], axis=1)
        gR = np.concatenate([g_brr, g_bri], axis=1)
        gT = np.concatenate([g_ctr, g_cti], axis=1)
        return s, gH, gR, gT

    def _post_update(self) -> None:
        """Keep |r_i| = 1 per complex dimension (same BLOCK layout as scoring)."""
        d = self.dim
        re = self.rel_emb[:, :d]
        im = self.rel_emb[:, d:]
        norm = np.sqrt(re * re + im * im)
        norm[norm == 0] = 1.0
        self.rel_emb[:, :d] = re / norm
        self.rel_emb[:, d:] = im / norm
