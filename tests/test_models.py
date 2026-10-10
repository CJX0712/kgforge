"""Model-layer tests.

The GOLD STANDARD here: every model's analytic gradient must match a central
finite-difference gradient to ~1e-6 (this is what proves the maths is right and
that SGD can actually learn).

Author: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from kgforge.core.seed import set_all
from kgforge.data.synthetic import generate_synthetic_kg
from kgforge.models import MODEL_REGISTRY

ALL = ["transe", "distmult", "complex", "rotate"]


def _numeric_row_grad(model, which: str, row: int, H, R, T, eps: float = 1e-6):
    """Central-difference gradient of score w.r.t. one embedding row."""
    mat = model.ent_emb if which == "ent" else model.rel_emb
    g = np.zeros(mat.shape[1], dtype=np.float64)
    for j in range(mat.shape[1]):
        old = mat[row, j]
        mat[row, j] = old + eps
        sp = float(model.score(H, R, T)[0])
        mat[row, j] = old - eps
        sm = float(model.score(H, R, T)[0])
        mat[row, j] = old
        g[j] = (sp - sm) / (2.0 * eps)
    return g


@pytest.mark.parametrize("name", ALL)
def test_analytic_gradient_matches_central_difference(name):
    from conftest import small_cfg

    cfg = small_cfg(dim=6)
    rng = set_all(0)
    kg = generate_synthetic_kg(cfg, rng)
    m = MODEL_REGISTRY[name](cfg)
    m._init_embeddings(rng, kg.n_entities, kg.n_relations)

    H = np.array([0], dtype=np.int64)
    R = np.array([0], dtype=np.int64)
    T = np.array([1], dtype=np.int64)
    _s, gH, gR, gT = m.score_and_grad(H, R, T)

    nH = _numeric_row_grad(m, "ent", 0, H, R, T)
    nR = _numeric_row_grad(m, "rel", 0, H, R, T)
    nT = _numeric_row_grad(m, "ent", 1, H, R, T)

    for analytic, numeric, label in ((gH[0], nH, "head"), (gR[0], nR, "rel"), (gT[0], nT, "tail")):
        denom = np.maximum(np.abs(numeric), 1e-6)
        rel = float(np.max(np.abs(analytic - numeric) / denom))
        assert rel < 1e-5, f"{name}/{label}: analytic vs numeric rel err {rel:.2e}"


@pytest.mark.parametrize("name", ALL)
def test_score_convention_larger_is_better(name):
    from conftest import small_cfg

    cfg = small_cfg(dim=6)
    rng = set_all(1)
    kg = generate_synthetic_kg(cfg, rng)
    m = MODEL_REGISTRY[name](cfg)
    m._init_embeddings(rng, kg.n_entities, kg.n_relations)
    # TransE: score is 0 exactly when h + r == t, and negative otherwise
    if name == "transe":
        m.ent_emb[0] = np.array([1.0] + [0.0] * (m.dim - 1))
        m.rel_emb[0] = np.array([1.0] + [0.0] * (m.dim - 1))
        m.ent_emb[1] = np.array([2.0] + [0.0] * (m.dim - 1))
        exact = float(m.score(np.array([0]), np.array([0]), np.array([1]))[0])
        assert exact == pytest.approx(0.0, abs=1e-9)
        m.ent_emb[1] = np.array([5.0] + [0.0] * (m.dim - 1))
        worse = float(m.score(np.array([0]), np.array([0]), np.array([1]))[0])
        assert worse < exact


def test_rotate_relations_stay_on_unit_circle():
    from conftest import small_cfg

    cfg = small_cfg(dim=6, epochs=10)
    rng = set_all(3)
    kg = generate_synthetic_kg(cfg, rng)
    m = MODEL_REGISTRY["rotate"](cfg)
    m.fit(kg, rng)
    d = m.dim
    re = m.rel_emb[:, :d]
    im = m.rel_emb[:, d:]
    norms = np.sqrt(re * re + im * im)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_fit_is_deterministic_for_same_seed():
    from conftest import small_cfg

    cfg = small_cfg(dim=6, epochs=6)
    out = []
    for _ in range(2):
        rng = set_all(11)
        kg = generate_synthetic_kg(cfg, rng)
        m = MODEL_REGISTRY["complex"](cfg)
        m.fit(kg, rng)
        out.append(m.ent_emb.copy())
    assert np.array_equal(out[0], out[1])


def test_prediction_shapes():
    from conftest import small_cfg

    cfg = small_cfg(dim=6, epochs=3)
    rng = set_all(5)
    kg = generate_synthetic_kg(cfg, rng)
    m = MODEL_REGISTRY["distmult"](cfg)
    m.fit(kg, rng)
    H = np.array([t.head for t in kg.test], dtype=np.int64)
    R = np.array([t.relation for t in kg.test], dtype=np.int64)
    T = np.array([t.tail for t in kg.test], dtype=np.int64)
    assert m.predict_scores_head(R, T).shape == (len(kg.test), kg.n_entities)
    assert m.predict_scores_tail(H, R).shape == (len(kg.test), kg.n_entities)
