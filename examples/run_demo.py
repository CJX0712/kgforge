#!/usr/bin/env python3
"""End-to-end KGForge demonstration.

1. trains all base embedding models on a reproducible synthetic KG,
2. benchmarks KGFuse vs each base model across >=3 seeds (mean +/- std),
3. verifies bit-identical determinism (same seed -> same numbers),
4. writes ``benchmark.json`` to the repo root.

Run from the repo root:  python examples/run_demo.py
Author: 晨星
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Pin threads BEFORE importing numpy for reproducible BLAS behaviour.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kgforge.core.config import Config
from kgforge.core.seed import set_all
from kgforge.pipeline.kgfuse import run_benchmark

SEEDS = [7, 42, 123]


def _fmt_mrr(d: dict) -> str:
    return "  ".join(
        f"{n}={d['model_mrr_mean'][n]:.4f}±{d['model_mrr_std'][n]:.4f}" for n in d["model_names"]
    )


def main() -> int:
    cfg = Config.from_env()
    set_all(cfg.seed)

    print("=" * 78)
    print(" KGForge — Knowledge Graph Embedding & Link Prediction")
    print(" Author: 晨星   |   Flagship: KGFuse (max-consensus reciprocal-rank ensemble)")
    print("=" * 78)
    print(f" config: {cfg.to_cli_summary()}")
    print(f" seeds : {SEEDS}")
    print("-" * 78)

    t0 = time.perf_counter()
    rep = run_benchmark(cfg, seeds=SEEDS, out_path=str(ROOT / "benchmark.json"))
    elapsed = time.perf_counter() - t0

    # ---- determinism: rerun with identical seeds, expect bit-identical --------
    rep2 = run_benchmark(cfg, seeds=SEEDS)
    max_delta = max(
        abs(rep["fuse_mrr_mean"] - rep2["fuse_mrr_mean"]),
        max(
            abs(rep["model_mrr_mean"][n] - rep2["model_mrr_mean"][n]) for n in rep["model_mrr_mean"]
        ),
    )
    deterministic = max_delta == 0.0

    # ---- report -------------------------------------------------------------
    print("-" * 78)
    print(f" Base models MRR : {_fmt_mrr(rep)}")
    print(f" KGFuse MRR     : {rep['fuse_mrr_mean']:.4f} ± {rep['fuse_mrr_std']:.4f}")
    print(f" Best single    : {rep['best_base']} = {rep['best_base_mrr_mean']:.4f}")
    print(f" Δ(KGFuse-best) : {rep['delta_abs']:+.4f}  (rel {rep['delta_rel'] * 100:+.2f}%)")
    print(
        f" Significance   : |Δ|={abs(rep['delta_abs']):.4f} vs "
        f"½(σ₁+σ₂)={rep['significance_threshold']:.4f} -> "
        f"{'SIGNIFICANT' if rep['significant'] else 'NOT significant'}"
    )
    print(f" Threshold ≥5%  : {'PASS' if rep['passes_threshold'] else 'FAIL'}")
    det_msg = "bit-identical (max|Δ|=0)" if deterministic else f"max|Δ|={max_delta:.2e}"
    print(f" Determinism    : {det_msg}")
    print(f" Elapsed        : {elapsed:.1f}s  (budget ≤ 60s)")
    print("-" * 78)
    print(" Per-relation-type MRR (failure analysis):")
    for rt, v in sorted(rep["relation_type_mrr"].items(), key=lambda kv: kv[1]):
        print(f"   {rt:14s} {v:.4f}")
    print("-" * 78)
    print(" Failure cases (worst 3):")
    for fc in rep["failure_cases"]:
        print(f"   {fc['relation_type']:14s} mrr={fc['mrr']:.4f}  -> {fc['attribution']}")
    print("=" * 78)
    print(f" benchmark.json written -> {ROOT / 'benchmark.json'}")

    # hard DoD gate: determinism must hold
    if not deterministic:
        print("ERROR: determinism violated; aborting delivery.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
