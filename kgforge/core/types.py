"""Shared dataclasses for KGForge.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Triple:
    """A single knowledge-graph fact ``(head, relation, tail)`` as integer ids."""

    head: int
    relation: int
    tail: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.head, self.relation, self.tail)


@dataclass
class KnowledgeGraph:
    """A knowledge graph with integer-mapped entities / relations.

    All splits share the same ``n_entities`` / ``n_relations`` id space.
    """

    n_entities: int
    n_relations: int
    train: list[Triple] = field(default_factory=list)
    valid: list[Triple] = field(default_factory=list)
    test: list[Triple] = field(default_factory=list)

    def all_triples(self) -> list[Triple]:
        return list(self.train) + list(self.valid) + list(self.test)

    def true_tail_per_head_rel(self) -> dict[tuple[int, int], set[int]]:
        out: dict[tuple[int, int], set[int]] = {}
        for t in self.all_triples():
            out.setdefault((t.head, t.relation), set()).add(t.tail)
        return out

    def true_head_per_rel_tail(self) -> dict[tuple[int, int], set[int]]:
        out: dict[tuple[int, int], set[int]] = {}
        for t in self.all_triples():
            out.setdefault((t.relation, t.tail), set()).add(t.head)
        return out

    def to_arrays(self) -> dict[str, np.ndarray]:
        def _arr(ts: list[Triple]) -> np.ndarray:
            if not ts:
                return np.zeros((0, 3), dtype=np.int64)
            return np.array([[t.head, t.relation, t.tail] for t in ts], dtype=np.int64)

        return {
            "train": _arr(self.train),
            "valid": _arr(self.valid),
            "test": _arr(self.test),
        }
