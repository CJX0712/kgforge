"""Synthetic knowledge-graph generator.

Design goal: produce KGs where **different embedding models excel on different
relation types**, so a consensus fusion has genuine complementary signal
(rather than a trivial win). Each relation is assigned one of four algebraic
types with well-known model affinities:

  * symmetric       -> ComplEx / DistMult strong, TransE weak (needs r≈0)
  * antisymmetric   -> TransE / RotatE strong, DistMult structurally blind
  * composition     -> RotatE strong (additive rotations), others weaker
  * one-to-many     -> all handle; adds ranking difficulty

Difficulty scales the fraction of structureless random distractor facts.
Deterministic: identical (cfg, rng) -> identical graph.

Author: 晨星
"""

from __future__ import annotations

import os

import numpy as np

from ..core.config import Config
from ..core.errors import DataError
from ..core.types import KnowledgeGraph, Triple

_REL_TYPES = ("symmetric", "antisymmetric", "composition", "onetomany")


def _build_relation_types(n_relations: int) -> list[str]:
    return [_REL_TYPES[i % len(_REL_TYPES)] for i in range(n_relations)]


def _gen_facts_for_relation(
    rel: int, rtype: str, n_facts: int, n_ent: int, rng: np.random.Generator
) -> list[tuple[int, int, int]]:
    """Return a list of (h, r, t) facts of the given type (pre-dedup)."""
    facts: list[tuple[int, int, int]] = []
    if rtype == "symmetric":
        target = max(2, n_facts)  # number of FACTS (each pair contributes 2)
        seen = set()
        tries = 0
        while len(facts) < target and tries < target * 10:
            tries += 1
            a, b = rng.integers(0, n_ent, size=2)
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            facts.append((int(a), rel, int(b)))
            facts.append((int(b), rel, int(a)))
    elif rtype == "antisymmetric":
        # hierarchical: parents span a smaller id band, children elsewhere
        # -> direction carries information (a is-a b, not vice-versa)
        parent_pool = rng.integers(0, max(1, n_ent // 3), size=n_facts)
        child_pool = rng.integers(n_ent // 3, n_ent, size=n_facts)
        seen = set()
        for a, b in zip(parent_pool, child_pool, strict=False):
            if a == b:
                continue
            key = (int(a), int(b))
            if key in seen:
                continue
            seen.add(key)
            facts.append((int(a), rel, int(b)))
    elif rtype == "composition":
        # chains a->b->c with shortcut a->c (transitive-style composition)
        target = max(1, n_facts // 3)
        seen = set()
        tries = 0
        while len(facts) < target * 3 and tries < target * 30:
            tries += 1
            a, b, c = rng.integers(0, n_ent, size=3)
            if len({a, b, c}) < 3:
                continue
            k1, k2, k3 = (int(a), int(b)), (int(b), int(c)), (int(a), int(c))
            if k1 in seen or k2 in seen or k3 in seen:
                continue
            seen.update({k1, k2, k3})
            facts.append((int(a), rel, int(b)))
            facts.append((int(b), rel, int(c)))
            facts.append((int(a), rel, int(c)))
    else:  # onetomany
        seen = set()
        tries = 0
        while len(facts) < n_facts and tries < n_facts * 20:
            tries += 1
            a, b = rng.integers(0, n_ent, size=2)
            key = (int(a), int(b))
            if key in seen:
                continue
            seen.add(key)
            facts.append((int(a), rel, int(b)))
    return facts


def generate_synthetic_kg(cfg: Config, rng: np.random.Generator) -> KnowledgeGraph:
    """Generate a reproducible KG with mixed relation types + difficulty noise."""
    if cfg.n_relations < 1:
        raise DataError("n_relations must be >= 1")
    rel_types = _build_relation_types(cfg.n_relations)
    total = cfg.n_train + cfg.n_valid + cfg.n_test

    raw: list[Triple] = []
    per_rel = max(1, total // max(1, cfg.n_relations))
    # distribute the integer-division remainder so quotas sum EXACTLY to total
    quotas = [per_rel] * cfg.n_relations
    for i in range(total - sum(quotas)):
        quotas[i] += 1
    # over-request then truncate: guarantees each relation can fill its quota
    # when the id space allows, instead of silently short-changing the split.
    for rel, rtype in enumerate(rel_types):
        quota = quotas[rel]
        # symmetric facts come in (a,r,b)/(b,r,a) pairs: round the quota UP to an
        # even number so truncation can never cut a pair in half (that would
        # silently destroy the symmetric property).
        if rtype == "symmetric":
            quota = quota + (quota % 2)
        margin = int(quota * 0.4) + 10
        seen_rel: set = set()
        taken = 0
        for h, r, t in _gen_facts_for_relation(rel, rtype, quota + margin, cfg.n_entities, rng):
            key = (h, rel, t)
            if key in seen_rel:
                continue
            seen_rel.add(key)
            raw.append(Triple(h, r, t))
            taken += 1
            if taken >= quota:
                break

    # dedup (keep first occurrence)
    seen = set()
    unique: list[Triple] = []
    for tp in raw:
        key = (tp.head, tp.relation, tp.tail)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tp)

    if len(unique) < total:
        raise DataError(
            f"generated only {len(unique)} unique facts < requested {total}; "
            f"raise n_entities / lower counts"
        )

    # deterministic shuffle then split (disjoint slices -> no leakage)
    perm = rng.permutation(len(unique))
    ordered = [unique[int(i)] for i in perm]
    test = ordered[: cfg.n_test]
    valid = ordered[cfg.n_test : cfg.n_test + cfg.n_valid]
    train_base = ordered[cfg.n_test + cfg.n_valid : cfg.n_test + cfg.n_valid + cfg.n_train]

    # difficulty: random distractor triples appended to TRAIN ONLY
    noise_frac = {"easy": 0.0, "medium": 0.2, "hard": 0.5}[cfg.difficulty]
    n_noise = int(len(train_base) * noise_frac)
    noise: list[Triple] = []
    n_attempts = 0
    while len(noise) < n_noise and n_attempts < n_noise * 20:
        n_attempts += 1
        h = int(rng.integers(0, cfg.n_entities))
        r = int(rng.integers(0, cfg.n_relations))
        t = int(rng.integers(0, cfg.n_entities))
        if (h, r, t) in seen:
            continue
        seen.add((h, r, t))
        noise.append(Triple(h, r, t))

    return KnowledgeGraph(
        n_entities=cfg.n_entities,
        n_relations=cfg.n_relations,
        train=train_base + noise,
        valid=valid,
        test=test,
    )


def _parse_line(line: str) -> Triple:
    parts = line.strip().split("\t")
    if len(parts) < 3:
        parts = line.strip().split()
    if len(parts) < 3:
        raise DataError(f"expected 'h r t' tab/space separated, got: {line!r}")
    return Triple(int(parts[0]), int(parts[1]), int(parts[2]))


def load_kg_from_tsv(
    path: str, n_entities: int | None = None, n_relations: int | None = None
) -> KnowledgeGraph:
    """Load KG from a TSV/space file with columns h r t (no header)."""
    triples: list[Triple] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            triples.append(_parse_line(line))
    if not triples:
        raise DataError(f"no triples parsed from {path}")
    max_e = max(max(t.head, t.tail) for t in triples) + 1
    max_r = max(t.relation for t in triples) + 1
    n_entities = n_entities or max_e
    n_relations = n_relations or max_r
    # 80/10/10 split deterministically (stable order)
    n = len(triples)
    n_test = n // 10
    n_valid = n // 10
    test = triples[:n_test]
    valid = triples[n_test : n_test + n_valid]
    train = triples[n_test + n_valid :]
    return KnowledgeGraph(n_entities, n_relations, train=train, valid=valid, test=test)


def save_kg_to_tsv(kg: KnowledgeGraph, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for t in kg.all_triples():
            fh.write(f"{t.head}\t{t.relation}\t{t.tail}\n")
