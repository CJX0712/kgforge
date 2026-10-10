# KGForge Architecture

Author: 晨星 (CJX0712)

## 1. Dependency direction (single, acyclic)

```
cli ──→ pipeline ──→ { data, models, eval } ──→ core
```

`core` depends on nothing but stdlib + numpy; there is no upward or cyclic import.
Every layer is independently testable.

## 2. Module responsibilities

### `core/`
| File | Responsibility |
|---|---|
| `types.py` | `Triple`, `KnowledgeGraph` (shared id space across splits) |
| `errors.py` | error taxonomy with stable codes E100–E500 |
| `config.py` | all tunables, `KG_*` env overrides, schema validation, capacity guard |
| `interfaces.py` | `KGEmbedder` / `FusionStrategy` Protocols |
| `seed.py` | `set_all(seed)` — one Generator + BLAS/OpenMP thread pinning = bit-identical runs |

### `data/`
- `generate_synthetic_kg(cfg, rng)` builds a KG where **different models win on different
  relation types** (that complementarity is the entire source of the fusion gain):
  - `symmetric` → ComplEx/DistMult strong, TransE weak (needs r≈0)
  - `antisymmetric` → TransE/RotatE strong, DistMult structurally blind
  - `composition` → RotatE strong (additive rotations)
  - `one-to-many` → all weak (adds ranking difficulty)
- Difficulty knob adds structureless random distractors to **train only**
  (`easy` 0%, `medium` 20%, `hard` 50%).
- Splits are produced by slicing a permutation of a **deduplicated** fact list, so
  train/valid/test are disjoint by construction (no leakage).

### `models/`
All four models implement `score_and_grad(H, R, T)`:
- returns `scores` with the **unified convention larger = more plausible**
- returns analytic gradients w.r.t. the head / relation / tail embedding rows

| Model | Score | Embedding layout |
|---|---|---|
| TransE | `-‖h + r − t‖²` | real `(n, d)` |
| DistMult | `Σ hᵢrᵢtᵢ` | real `(n, d)` |
| ComplEx | `Re⟨h, r, conj(t)⟩` | real blocks `[re(d), im(d)]` |
| RotatE | `-‖h ∘ r − t‖²` | real blocks `[re(d), im(d)]`, relations constrained `|rᵢ| = 1` |

Training (shared in `base.py`): margin-ranking loss with negative sampling,
L2 weight decay on involved rows, per-row L2 gradient clipping, relation re-normalisation
hook `_post_update()` for RotatE.

### `eval/`
Filtered link prediction (Bordes et al. 2013): when ranking candidates for a test triple,
every *other* true fact sharing the same `(h, r)` or `(r, t)` is masked to `-inf` so it
cannot outrank the query. Reports MRR and Hits@1/3/10 averaged over head and tail queries.

### `pipeline/`
`KGFuse` fuses per-model **ranks**, not scores, because the base scores live on
incomparable scales. Strategies:
- `rrf` (default, adopted): `Σ_m 1/(k + rank_m)`, k = 60 — Cormack et al. 2009
- `mean_rr`: average reciprocal rank
- `max_consensus`: **rejected** — see CHANGELOG

`KGFusePipeline.run()` produces the benchmark: multi-seed mean±std, best-base comparison,
significance criterion ½(σ₁+σ₂), drop-one ablation, strategy ablation, and per-relation-type
failure analysis with attribution.

## 3. Determinism strategy

1. One `np.random.Generator` created by `set_all(seed)` and threaded through
   data generation → model init → training; no hidden RNGs.
2. `OMP/OPENBLAS/MKL/NUMEXPR` threads pinned to 1 (multi-threaded BLAS is the classic
   source of non-reproducible reduction order).
3. No `set`/`dict` iteration order affects any numeric path.
4. Verified by re-running the full benchmark with identical seeds and asserting
   `max|Δ| = 0` (a hard gate in `examples/run_demo.py`).

## 4. Extension points

- new embedding model → subclass `BaseKGEModel`, register in `models/MODEL_REGISTRY`
- new fusion rule → add to `eval/metrics.FUSIONS`
- new data source → `data/synthetic.load_kg_from_tsv`, or implement `KnowledgeGraph`
