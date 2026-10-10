#!/usr/bin/env python3
"""KGForge command-line interface.

Subcommands:
  benchmark   run the full KGFuse benchmark and write benchmark.json
  eval        train base models + KGFuse on a generated KG and print metrics
  gen         write a synthetic KG to a TSV file

Thread pins are set before importing numpy so runs are reproducible.

Author: 晨星
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

from kgforge.core.config import Config  # noqa: E402
from kgforge.core.seed import set_all  # noqa: E402
from kgforge.data.synthetic import (  # noqa: E402
    generate_synthetic_kg,
    save_kg_to_tsv,
)
from kgforge.eval.metrics import evaluate_link_prediction  # noqa: E402
from kgforge.models import MODEL_REGISTRY  # noqa: E402
from kgforge.pipeline.kgfuse import KGFuse, run_benchmark  # noqa: E402


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x != ""]


def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    set_all(cfg.seed)
    rep = run_benchmark(cfg, seeds=_parse_seeds(args.seeds), out_path=args.out)
    print(
        f"KGFuse MRR={rep['fuse_mrr_mean']:.4f}±{rep['fuse_mrr_std']:.4f}  "
        f"best_base={rep['best_base']}({rep['best_base_mrr_mean']:.4f})  "
        f"Δrel={rep['delta_rel'] * 100:+.2f}%  "
        f"threshold={'PASS' if rep['passes_threshold'] else 'FAIL'}"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    seeds = _parse_seeds(args.seeds)
    set_all(seeds[0])
    rng = set_all(seeds[0])
    kg = generate_synthetic_kg(cfg, rng)
    names = ["transe", "distmult"]
    if cfg.use_complex:
        names.append("complex")
    if cfg.use_rotate:
        names.append("rotate")
    models = [MODEL_REGISTRY[n](cfg) for n in names]
    for m in models:
        m.fit(kg, rng)
    print(
        f"KG: entities={kg.n_entities} relations={kg.n_relations} "
        f"train={len(kg.train)} test={len(kg.test)}"
    )
    for m in models:
        ev = evaluate_link_prediction(m, kg, cfg.top_k)
        print(
            f"  {m.name:10s} MRR={ev['mrr']:.4f}  "
            f"hits@1={ev['hits'][1]:.3f} h@3={ev['hits'][3]:.3f} h@10={ev['hits'][10]:.3f}"
        )
    fuse = KGFuse(models)
    fev = fuse.evaluate(kg, cfg.top_k)
    print(
        f"  {'kgfuse':10s} MRR={fev['mrr']:.4f}  "
        f"hits@1={fev['hits'][1]:.3f} h@3={fev['hits'][3]:.3f} h@10={fev['hits'][10]:.3f}"
    )
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    rng = set_all(args.seed)
    kg = generate_synthetic_kg(cfg, rng)
    save_kg_to_tsv(kg, args.out)
    print(f"wrote {args.out}  (train={len(kg.train)} valid={len(kg.valid)} test={len(kg.test)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="kgforge", description="KGForge CLI (Author: 晨星)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="run full KGFuse benchmark")
    b.add_argument("--seeds", default="7,42,123")
    b.add_argument("--out", default=str(ROOT / "benchmark.json"))
    b.set_defaults(func=cmd_benchmark)

    e = sub.add_parser("eval", help="train + evaluate on one generated KG")
    e.add_argument("--seeds", default="7")
    e.set_defaults(func=cmd_eval)

    g = sub.add_parser("gen", help="generate a synthetic KG TSV")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", default=str(ROOT / "kg_synthetic.tsv"))
    g.set_defaults(func=cmd_gen)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
