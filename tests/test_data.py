"""Data-layer tests: reproducibility, split hygiene, difficulty knob, IO.

Author: 晨星
"""

from __future__ import annotations

import pytest

from kgforge.core.errors import DataError
from kgforge.core.seed import set_all
from kgforge.data.synthetic import (
    _build_relation_types,
    generate_synthetic_kg,
    load_kg_from_tsv,
    save_kg_to_tsv,
)


def test_generation_is_reproducible():
    from conftest import small_cfg

    cfg = small_cfg()
    kg1 = generate_synthetic_kg(cfg, set_all(7))
    kg2 = generate_synthetic_kg(cfg, set_all(7))
    assert [t.as_tuple() for t in kg1.train] == [t.as_tuple() for t in kg2.train]
    assert [t.as_tuple() for t in kg1.test] == [t.as_tuple() for t in kg2.test]


def test_splits_are_disjoint():
    from conftest import small_cfg

    cfg = small_cfg()
    kg = generate_synthetic_kg(cfg, set_all(7))
    tr = {t.as_tuple() for t in kg.train}
    va = {t.as_tuple() for t in kg.valid}
    te = {t.as_tuple() for t in kg.test}
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr) == len(kg.train)  # train itself has no duplicates
    assert len(kg.test) == cfg.n_test
    assert len(kg.valid) == cfg.n_valid


def test_difficulty_adds_noise_to_train_only():
    from conftest import small_cfg

    easy = generate_synthetic_kg(small_cfg(difficulty="easy"), set_all(7))
    hard = generate_synthetic_kg(small_cfg(difficulty="hard"), set_all(7))
    assert len(easy.test) == len(hard.test)
    assert len(hard.train) > len(easy.train)


def test_relation_types_cover_all_kinds():
    types = _build_relation_types(8)
    assert set(types) == {"symmetric", "antisymmetric", "composition", "onetomany"}


def test_symmetric_relation_is_truly_symmetric():
    from conftest import small_cfg

    # difficulty="easy" => no random distractor facts, so the structural property
    # must hold exactly. (With noise, random triples legitimately break it.)
    kg = generate_synthetic_kg(small_cfg(difficulty="easy"), set_all(7))
    sym = {i for i, t in enumerate(_build_relation_types(kg.n_relations)) if t == "symmetric"}
    facts = {(t.head, t.relation, t.tail) for t in kg.all_triples()}
    for r in sym:
        some = [(h, t) for (h, rr, t) in facts if rr == r]
        assert some, "symmetric relation produced no facts"
        assert all((t, h) in {(a, b) for (a, b) in some} for (h, t) in some)


def test_tsv_roundtrip(tmp_path):
    from conftest import small_cfg

    kg = generate_synthetic_kg(small_cfg(), set_all(7))
    p = tmp_path / "kg.tsv"
    save_kg_to_tsv(kg, str(p))
    kg2 = load_kg_from_tsv(str(p))
    assert len(kg2.all_triples()) == len(kg.all_triples())
    assert {t.as_tuple() for t in kg2.all_triples()} == {t.as_tuple() for t in kg.all_triples()}


def test_too_small_id_space_raises():
    from conftest import small_cfg

    cfg = small_cfg(n_entities=12, n_relations=1, n_train=400, n_valid=50, n_test=50)
    with pytest.raises(DataError):
        generate_synthetic_kg(cfg, set_all(7))
