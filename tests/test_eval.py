"""Evaluation tests: filtered protocol, rank semantics, metric bounds.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from kgforge.core.types import KnowledgeGraph, Triple
from kgforge.eval.metrics import _mask_known, _ranks_descending, evaluate_link_prediction


def _kg() -> KnowledgeGraph:
    return KnowledgeGraph(
        n_entities=5,
        n_relations=1,
        train=[Triple(0, 0, 1), Triple(0, 0, 2)],
        valid=[],
        test=[Triple(0, 0, 3)],
    )


def test_mask_known_filters_other_true_answers():
    kg = _kg()
    S = np.zeros((1, kg.n_entities), dtype=np.float64)
    S[0, 1] = 10.0  # a different true tail for (0,0) -> must be masked
    S[0, 2] = 9.0
    S[0, 3] = 1.0  # the queried answer -> must survive
    out = _mask_known(S.copy(), kg, "tail")
    assert np.isneginf(out[0, 1])
    assert np.isneginf(out[0, 2])
    assert out[0, 3] == 1.0


def test_mask_known_leaves_unrelated_entities_untouched():
    kg = _kg()
    S = np.zeros((1, kg.n_entities), dtype=np.float64)
    out = _mask_known(S.copy(), kg, "tail")
    assert out[0, 0] == 0.0 and out[0, 4] == 0.0


def test_ranks_descending_one_is_best():
    S = np.array([[1.0, 5.0, 3.0, -2.0]])
    ranks = _ranks_descending(S)
    assert ranks[0, 1] == 1  # largest score -> rank 1
    assert ranks[0, 0] == 3
    assert ranks[0, 3] == 4  # smallest score -> worst rank


def test_metrics_bounds_and_monotonic_hits():
    from conftest import small_cfg

    from kgforge.core.seed import set_all
    from kgforge.data.synthetic import generate_synthetic_kg
    from kgforge.models import MODEL_REGISTRY

    cfg = small_cfg(dim=6, epochs=8)
    rng = set_all(7)
    kg = generate_synthetic_kg(cfg, rng)
    m = MODEL_REGISTRY["complex"](cfg)
    m.fit(kg, rng)
    ev = evaluate_link_prediction(m, kg, (1, 3, 10))
    assert 0.0 < ev["mrr"] <= 1.0
    assert ev["hits"][1] <= ev["hits"][3] <= ev["hits"][10]
    assert ev["n_test"] == len(kg.test)


def test_empty_test_set_is_handled():
    from conftest import small_cfg

    from kgforge.core.seed import set_all
    from kgforge.data.synthetic import generate_synthetic_kg
    from kgforge.models import MODEL_REGISTRY

    cfg = small_cfg(dim=4, epochs=1)
    rng = set_all(7)
    kg = generate_synthetic_kg(cfg, rng)
    kg.test = []
    m = MODEL_REGISTRY["transe"](cfg)
    m.fit(kg, rng)
    ev = evaluate_link_prediction(m, kg, (1, 3, 10))
    assert ev["n_test"] == 0
