# Changelog

All notable changes to KGForge. Author: 晨星 (CJX0712).

## v0.1.0 — 2026-10-11

Initial release.

### Added
- `core`: types / errors (E100–E500) / config (`KG_*` env overrides + schema validation) /
  interfaces (Protocol) / `seed.set_all` (single deterministic entry + BLAS thread pinning).
- `data`: synthetic KG generator with four relation semantics
  (symmetric / antisymmetric / composition / one-to-many) and a `easy|medium|hard`
  difficulty knob; TSV load/save.
- `models`: TransE (Bordes 2013), DistMult (Yang 2015), ComplEx (Trouillon 2016),
  RotatE (Sun 2019) — pure NumPy, unified "larger = more plausible" score convention,
  analytic gradients + per-row L2 gradient clipping.
- `eval`: filtered link-prediction protocol (Bordes 2013) — MRR and Hits@1/3/10.
- `pipeline`: **KGFuse** fusion flagship (RRF default) + `KGFusePipeline` benchmark with
  multi-seed mean±std, drop-one ablation, fusion-strategy ablation, per-relation-type
  failure analysis, and a pre-registered significance criterion ½(σ₁+σ₂).
- `cli` (`benchmark` / `eval` / `gen`) and `examples/run_demo.py`.
- 38 tests, ruff hard gate, CI matrix py3.12×3.13 × ubuntu×windows, Dockerfile, Makefile.

### Fixed (real bugs found during this delivery)
1. **Gradient sign inverted** in the shared SGD loop — positives were pushed *down* and
   negatives *up*, so MRR landed **below random**. Correct update is
   `emb_pos += lr·g_pos`, `emb_neg -= lr·g_neg`.
2. **RotatE layout mismatch** — init/`_post_update` used an interleaved
   `[re0,im0,re1,im1]` layout while scoring used the block `[real(d), imag(d)]` layout,
   so the unit-modulus constraint was enforced on the wrong pairs. RotatE went from
   0.0288 (≈random) to 0.1101 after the fix.
3. **RotatE score not reduced over dimensions** — `s = -(u²+v²)` returned `(B, d)` instead
   of `(B,)`, making the loss mask 2-D and crashing the indexer.
4. **Symmetric relation truncated mid-pair** — an odd per-relation quota cut a
   `(a,r,b)`/`(b,r,a)` pair in half, silently destroying the symmetric property.
5. **Quota remainder** — `total // n_relations` left the fact count short of
   `n_train+n_valid+n_test`; remainder is now distributed.
6. **Capacity guard too strict** — usable triples are `n_entities² × n_relations`,
   not `n_entities × n_relations`.

### Rejected (honest negative result)
- `max_consensus` fusion: 0.1166 MRR, **worse than the strongest single model (0.1717)**.
  Retained in code and reports as a documented counterexample, not used as the flagship.
