"""Filtered link-prediction metrics (Bordes et al. 2013 protocol).

We follow the *filtered* setting: when ranking candidate answers for a test
triple, every other true fact sharing the same (head, relation) or
(relation, tail) is masked to -inf so it cannot outrank the query. This is the
standard, non-inflated protocol used by papers-with-code link-prediction
leaderboards.

Score convention: larger = more plausible (matches the models).

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import KnowledgeGraph
from ..models.base import BaseKGEModel


def _scores_head(model: BaseKGEModel, kg: KnowledgeGraph) -> np.ndarray:
    # head is the MISSING slot here, so only (relation, tail) is used
    R = np.array([t.relation for t in kg.test], dtype=np.int64)
    T = np.array([t.tail for t in kg.test], dtype=np.int64)
    return model.predict_scores_head(R, T)  # (B, n_ent)


def _scores_tail(model: BaseKGEModel, kg: KnowledgeGraph) -> np.ndarray:
    H = np.array([t.head for t in kg.test], dtype=np.int64)
    R = np.array([t.relation for t in kg.test], dtype=np.int64)
    return model.predict_scores_tail(H, R)  # (B, n_ent)


def _mask_known(S: np.ndarray, kg: KnowledgeGraph, direction: str) -> np.ndarray:
    """Set scores of other true answers to -inf (filtered protocol)."""
    test = kg.test
    if direction == "head":
        known = kg.true_head_per_rel_tail()
        for i, t in enumerate(test):
            for e in known.get((t.relation, t.tail), ()):
                if e != t.head:
                    S[i, e] = -np.inf
    else:
        known = kg.true_tail_per_head_rel()
        for i, t in enumerate(test):
            for e in known.get((t.head, t.relation), ()):
                if e != t.tail:
                    S[i, e] = -np.inf
    return S


def _ranks_descending(S: np.ndarray) -> np.ndarray:
    """Rank positions (1 = best / largest score); ties get adjacent ranks."""
    B, n = S.shape
    order = np.argsort(-S, axis=1, kind="stable")
    pos = np.empty_like(order)
    pos[np.arange(B)[:, None], order] = np.arange(n)[None, :]
    return pos + 1.0


def _reciprocal_rank_of_true(ranks: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    B = ranks.shape[0]
    return 1.0 / ranks[np.arange(B), true_idx]


def evaluate_link_prediction(
    model: BaseKGEModel, kg: KnowledgeGraph, top_k: tuple[int, ...] = (1, 3, 10)
) -> dict[str, object]:
    """Filtered MRR + Hits@k for a single model (head & tail averaged)."""
    if len(kg.test) == 0:
        return {"mrr": float("nan"), "hits": {k: float("nan") for k in top_k}, "n_test": 0}
    Sh = _mask_known(_scores_head(model, kg), kg, "head")
    St = _mask_known(_scores_tail(model, kg), kg, "tail")
    rh = _reciprocal_rank_of_true(
        _ranks_descending(Sh), np.array([t.head for t in kg.test], dtype=np.int64)
    )
    rt = _reciprocal_rank_of_true(
        _ranks_descending(St), np.array([t.tail for t in kg.test], dtype=np.int64)
    )
    all_rr = np.concatenate([rh, rt])
    ranks = 1.0 / all_rr
    out = {
        "mrr": float(np.mean(all_rr)),
        "hits": {int(k): float(np.mean(ranks <= k)) for k in top_k},
        "n_test": len(kg.test),
        "_rr_head": rh,
        "_rr_tail": rt,
    }
    return out


def reciprocal_rank_arrays(
    models: list[BaseKGEModel], kg: KnowledgeGraph, direction: str
) -> np.ndarray:
    """Per-model reciprocal-rank arrays over all candidates: shape (M, B, n_ent)."""
    B = len(kg.test)
    n_ent = models[0].ent_emb.shape[0]
    M = len(models)
    out = np.empty((M, B, n_ent), dtype=np.float64)
    for m, model in enumerate(models):
        if direction == "head":
            S = _mask_known(_scores_head(model, kg), kg, "head")
        else:
            S = _mask_known(_scores_tail(model, kg), kg, "tail")
        ranks = _ranks_descending(S)
        out[m] = 1.0 / ranks
    return out


def fuse_mean_rr(rr_arrays: np.ndarray) -> np.ndarray:
    """Mean reciprocal-rank fusion: variance-reducing average across models."""
    return np.mean(rr_arrays, axis=0)  # (B, n_ent)


def fuse_rrf(rr_arrays: np.ndarray, k: float = 60.0) -> np.ndarray:
    """Reciprocal Rank Fusion (Cormack et al. 2009): sum_m 1/(k + rank_m)."""
    ranks = 1.0 / rr_arrays
    return np.sum(1.0 / (k + ranks), axis=0)  # (B, n_ent)


def fuse_max_consensus(rr_arrays: np.ndarray) -> np.ndarray:
    """Max-consensus fusion: fused reciprocal rank = max over models per entity.

    Rejected variant (kept as an honest negative result): taking the max *per
    candidate* also inflates every false candidate, so the true entity's fused
    rank can get WORSE than every base model.
    """
    return np.max(rr_arrays, axis=0)  # (B, n_ent)


FUSIONS = {
    "mean_rr": fuse_mean_rr,
    "rrf": fuse_rrf,
    "max_consensus": fuse_max_consensus,
}


def filtered_ranks(fused_rr: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """Rank (1=best) of the true entity under a fused reciprocal-rank matrix."""
    B, n = fused_rr.shape
    order = np.argsort(-fused_rr, axis=1, kind="stable")
    pos = np.empty_like(order)
    pos[np.arange(B)[:, None], order] = np.arange(n)[None, :]
    return pos[np.arange(B), true_idx] + 1.0
