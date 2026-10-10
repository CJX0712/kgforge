"""Base class: shared SGD training, deterministic seeding, prediction API.

Score convention (unified across all models): ``score`` returns a real number
where **larger = more plausible**. Training uses margin-ranking loss with
negative sampling + L2 weight decay. All randomness comes from a single
``np.random.Generator`` passed in, so two runs with the same seed are
bit-identical.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.errors import ModelError, TrainingError
from ..core.types import KnowledgeGraph


class BaseKGEModel:
    name = "base"
    complex_like = False  # emb_dim == 2 * dim for ComplEx / RotatE

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dim = cfg.dim
        self.emb_dim = (2 * cfg.dim) if self.complex_like else cfg.dim
        self.ent_emb: np.ndarray | None = None
        self.rel_emb: np.ndarray | None = None

    # ----- to be implemented by subclasses -----------------------------------
    def _init_embeddings(self, rng: np.random.Generator, n_ent: int, n_rel: int) -> None:
        raise NotImplementedError

    def score_and_grad(self, H, R, T):
        """Return (scores (B,), grad_H, grad_R, grad_T) each (B, emb_dim).

        ``scores`` uses the unified convention larger = more plausible.
        ``grad_*`` is the gradient of ``scores`` w.r.t. the corresponding
        embedding rows.
        """
        raise NotImplementedError

    def _post_update(self) -> None:
        """Hook run after each gradient step (e.g. RotatE unit-norm)."""

    # ----- shared machinery ---------------------------------------------------
    def score(self, H, R, T):
        s, _, _, _ = self.score_and_grad(H, R, T)
        return s

    def fit(self, kg: KnowledgeGraph, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng(self.cfg.seed)
        n_ent, n_rel = kg.n_entities, kg.n_relations
        self._init_embeddings(rng, n_ent, n_rel)
        if self.ent_emb is None or self.rel_emb is None:
            raise ModelError(f"{self.name}: embeddings not initialised")
        self._train(kg, rng)
        return self

    def _train(self, kg: KnowledgeGraph, rng: np.random.Generator) -> None:
        cfg = self.cfg
        E = self.ent_emb
        Rl = self.rel_emb
        assert E is not None and Rl is not None
        n_ent = E.shape[0]

        triples = np.asarray([[t.head, t.relation, t.tail] for t in kg.train], dtype=np.int64)
        if triples.shape[0] == 0:
            raise TrainingError(f"{self.name}: empty training set")
        N = triples.shape[0]
        H_all, R_all, T_all = triples[:, 0], triples[:, 1], triples[:, 2]

        lr = float(cfg.learning_rate)
        margin = float(cfg.margin)
        reg = float(cfg.reg)
        bs = int(cfg.batch_size)
        decay = 1.0 - lr * reg

        for _epoch in range(int(cfg.epochs)):
            order = rng.permutation(N)
            for start in range(0, N, bs):
                idx = order[start : start + bs]
                hp = H_all[idx]
                rp = R_all[idx]
                tp = T_all[idx]
                B = hp.shape[0]

                cor = rng.integers(0, 2, size=B)
                ne = rng.integers(0, n_ent, size=B)
                hn = np.where(cor == 0, ne, hp)
                tn = np.where(cor == 0, tp, ne)

                sp, gHp, gRp, gTp = self.score_and_grad(hp, rp, tp)
                sn, gHn, gRn, gTn = self.score_and_grad(hn, rp, tn)

                viol = margin + sn - sp
                np.maximum(0.0, viol, out=viol)
                mask = viol > 0
                if mask.any():
                    # L = max(0, gamma + s_neg - s_pos);  minimise by Gradient Descent:
                    #   dL/d emb_pos = -g_pos  ->  emb_pos += lr * g_pos   (raise s_pos)
                    #   dL/d emb_neg = +g_neg  ->  emb_neg -= lr * g_neg   (lower s_neg)
                    clip = self._clip
                    self._scatter(E, hp[mask], clip(+lr * gHp[mask]))
                    self._scatter(Rl, rp[mask], clip(+lr * gRp[mask]))
                    self._scatter(E, tp[mask], clip(+lr * gTp[mask]))
                    self._scatter(E, hn[mask], clip(-lr * gHn[mask]))
                    self._scatter(Rl, rp[mask], clip(-lr * gRn[mask]))
                    self._scatter(E, tn[mask], clip(-lr * gTn[mask]))
                    # L2 weight decay on involved rows only
                    ie = np.unique(np.concatenate([hp[mask], tp[mask], hn[mask], tn[mask]]))
                    ir = np.unique(rp[mask])
                    E[ie] *= decay
                    Rl[ir] *= decay
                self._post_update()

    def _clip(self, g: np.ndarray) -> np.ndarray:
        """Per-row L2 gradient clipping — stops bilinear models overflowing."""
        mx = float(getattr(self.cfg, "grad_clip", 0.0))
        if mx <= 0.0:
            return g
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return g * np.minimum(1.0, mx / np.maximum(n, 1e-12))

    @staticmethod
    def _scatter(mat: np.ndarray, idx: np.ndarray, grad: np.ndarray) -> None:
        np.add.at(mat, idx, grad)

    # ----- prediction (vectorised scoring of all entities) --------------------
    def predict_scores_head(self, R: np.ndarray, T: np.ndarray) -> np.ndarray:
        """score(e, R_i, T_i) for all e -> shape (B, n_entities)."""
        B = R.shape[0]
        n = self.ent_emb.shape[0]
        H = np.repeat(np.arange(n), B)
        Rr = np.tile(R, n)
        Tt = np.tile(T, n)
        sc = self.score(H, Rr, Tt)
        return sc.reshape(n, B).T

    def predict_scores_tail(self, H: np.ndarray, R: np.ndarray) -> np.ndarray:
        """score(H_i, R_i, e) for all e -> shape (B, n_entities)."""
        B = H.shape[0]
        n = self.ent_emb.shape[0]
        Tt = np.repeat(np.arange(n), B)
        Hh = np.tile(H, n)
        Rr = np.tile(R, n)
        sc = self.score(Hh, Rr, Tt)
        return sc.reshape(n, B).T
