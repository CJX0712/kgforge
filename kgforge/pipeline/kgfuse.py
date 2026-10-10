"""KGFuse: rank-fusion flagship + benchmark harness.

Flagship strategy is **Reciprocal Rank Fusion** (Cormack et al. 2009):
score(e) = sum_m 1/(k + rank_m(e)), k = 60. RRF is scale-free across models
(their raw scores live on incomparable scales — squared distance for
TransE/RotatE vs bilinear dot products for DistMult/ComplEx), so fusing *ranks*
rather than scores is the correct way to combine them.

Two alternatives are retained as honest ablations and are reported even though
they lose: ``mean_rr`` (average reciprocal rank) and ``max_consensus``
(per-candidate max, which inflates false candidates and ends up WORSE than every
base model).

Author: 晨星
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..core.config import Config
from ..core.seed import set_all
from ..data.synthetic import _build_relation_types, generate_synthetic_kg
from ..eval.metrics import (
    FUSIONS,
    evaluate_link_prediction,
    filtered_ranks,
    reciprocal_rank_arrays,
)
from ..models import MODEL_REGISTRY

_REL_WEAKNESS = {
    "symmetric": "TransE 需 r≈0 易退化，ComplEx/DistMult 互补弥补",
    "antisymmetric": "DistMult 结构对称、方向盲，TransE/RotatE 互补弥补",
    "composition": "RotatE 旋转加法建模组合，其余模型较弱",
    "onetomany": "高扇出稀释信号，所有模型排名熵增大",
}


class KGFuse:
    """Fusion flagship over base embedding models.

    Default strategy ``mean_rr``: average the per-model reciprocal ranks of each
    candidate entity, then re-rank. ``rrf`` (Cormack et al. 2009) and
    ``max_consensus`` are retained for ablation; max_consensus is a documented
    negative result.
    """

    name = "kgfuse"

    def __init__(self, models: list, strategy: str = "rrf"):
        if not models:
            raise ValueError("KGFuse requires at least one base model")
        if strategy not in FUSIONS:
            raise ValueError(f"unknown fusion strategy {strategy!r}; use {list(FUSIONS)}")
        self.models = list(models)
        self.strategy = strategy
        self._fuse = FUSIONS[strategy]

    def evaluate(self, kg, top_k=(1, 3, 10)) -> dict[str, object]:
        rr_h = reciprocal_rank_arrays(self.models, kg, "head")
        rr_t = reciprocal_rank_arrays(self.models, kg, "tail")
        th = np.array([t.head for t in kg.test], dtype=np.int64)
        tt = np.array([t.tail for t in kg.test], dtype=np.int64)
        rank_h = filtered_ranks(self._fuse(rr_h), th)
        rank_t = filtered_ranks(self._fuse(rr_t), tt)
        rr = np.concatenate([1.0 / rank_h, 1.0 / rank_t])
        ranks = 1.0 / rr
        return {
            "mrr": float(np.mean(rr)),
            "hits": {int(k): float(np.mean(ranks <= k)) for k in top_k},
            "n_test": len(kg.test),
        }

    def evaluate_detailed(self, kg, top_k=(1, 3, 10)):
        rr_h = reciprocal_rank_arrays(self.models, kg, "head")
        rr_t = reciprocal_rank_arrays(self.models, kg, "tail")
        th = np.array([t.head for t in kg.test], dtype=np.int64)
        tt = np.array([t.tail for t in kg.test], dtype=np.int64)
        rank_h = filtered_ranks(self._fuse(rr_h), th)
        rank_t = filtered_ranks(self._fuse(rr_t), tt)
        rr_all = np.concatenate([1.0 / rank_h, 1.0 / rank_t])
        rel = np.array([t.relation for t in kg.test], dtype=np.int64)
        rel_all = np.concatenate([rel, rel])
        return rr_all, rel_all

    def evaluate_strategies(self, kg, top_k=(1, 3, 10)) -> dict[str, float]:
        """MRR of every fusion strategy on the same KG (ablation)."""
        out: dict[str, float] = {}
        rr_h = reciprocal_rank_arrays(self.models, kg, "head")
        rr_t = reciprocal_rank_arrays(self.models, kg, "tail")
        th = np.array([t.head for t in kg.test], dtype=np.int64)
        tt = np.array([t.tail for t in kg.test], dtype=np.int64)
        for sname, fn in FUSIONS.items():
            rank_h = filtered_ranks(fn(rr_h), th)
            rank_t = filtered_ranks(fn(rr_t), tt)
            rr = np.concatenate([1.0 / rank_h, 1.0 / rank_t])
            out[sname] = float(np.mean(rr))
        return out


class KGFusePipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _build_models(self) -> list:
        names = ["transe", "distmult"]
        if self.cfg.use_complex:
            names.append("complex")
        if self.cfg.use_rotate:
            names.append("rotate")
        return [MODEL_REGISTRY[n](self.cfg) for n in names]

    def run(self, seeds: list[int] | None = None, verbose: bool = True) -> dict:
        cfg = self.cfg
        seeds = seeds or [7, 42, 123]
        per_seed: list[dict] = []
        t0 = time.perf_counter()

        for seed in seeds:
            rng = set_all(seed)
            kg = generate_synthetic_kg(cfg, rng)
            models = self._build_models()
            for m in models:
                m.fit(kg, rng)

            pm: dict[str, dict] = {}
            for m in models:
                ev = evaluate_link_prediction(m, kg, cfg.top_k)
                pm[m.name] = {"mrr": ev["mrr"], "hits": ev["hits"]}

            fuse = KGFuse(models)
            fev = fuse.evaluate(kg, cfg.top_k)
            strat = fuse.evaluate_strategies(kg, cfg.top_k)

            abl: dict[str, float] = {}
            for i, m in enumerate(models):
                sub = models[:i] + models[i + 1 :]
                abl[m.name] = KGFuse(sub).evaluate(kg, cfg.top_k)["mrr"]

            rr_all, rel_all = fuse.evaluate_detailed(kg, cfg.top_k)
            per_seed.append(
                {
                    "seed": seed,
                    "n_train": len(kg.train),
                    "n_test": len(kg.test),
                    "per_model": pm,
                    "fuse": {"mrr": fev["mrr"], "hits": fev["hits"]},
                    "ablation": abl,
                    "strategies": strat,
                    "_rr_all": rr_all,
                    "_rel_all": rel_all,
                    "model_names": [m.name for m in models],
                }
            )
            if verbose:
                print(
                    f"  seed={seed}: fuse MRR={fev['mrr']:.4f} "
                    f"bases={ {k: round(v['mrr'], 4) for k, v in pm.items()} }"
                )

        elapsed = time.perf_counter() - t0
        report = self._aggregate(per_seed, elapsed, seeds)
        return report

    def _aggregate(self, per_seed: list[dict], elapsed: float, seeds) -> dict:
        cfg = self.cfg
        model_names = per_seed[0]["model_names"]

        # per-model MRR across seeds
        model_mrr: dict[str, list[float]] = {n: [] for n in model_names}
        for ps in per_seed:
            for n in model_names:
                model_mrr[n].append(ps["per_model"][n]["mrr"])

        fuse_mrr = [ps["fuse"]["mrr"] for ps in per_seed]

        def ms(xs):
            return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0

        model_stats = {n: ms(v) for n, v in model_mrr.items()}
        fuse_mean, fuse_std = ms(fuse_mrr)

        # strongest single base by mean MRR
        best_base = max(model_names, key=lambda n: model_stats[n][0])
        best_base_mean = model_stats[best_base][0]
        delta_abs = fuse_mean - best_base_mean
        delta_rel = (delta_abs / best_base_mean) if best_base_mean > 0 else 0.0
        # Significance criterion (pre-registered): mean gap > 1/2 (sigma_1 + sigma_2)
        sig_thresh = 0.5 * (fuse_std + model_stats[best_base][1])
        significant = bool(delta_abs > sig_thresh)

        # ablation summary (mean over seeds of dropped-model MRR)
        abl_mean: dict[str, float] = {}
        for n in model_names:
            abl_mean[n] = float(np.mean([ps["ablation"][n] for ps in per_seed]))

        strat_mean: dict[str, float] = {}
        for s in per_seed[0]["strategies"]:
            strat_mean[s] = float(np.mean([ps["strategies"][s] for ps in per_seed]))

        # per-relation-type MRR (failure analysis) aggregated across seeds
        rel_types = _build_relation_types(cfg.n_relations)
        type_mrr: dict[str, list[float]] = {}
        for ps in per_seed:
            rr = ps["_rr_all"]
            rel = ps["_rel_all"]
            for rti in set(rel_types):
                mask = rel == rel_types.index(rti)
                if mask.any():
                    type_mrr.setdefault(rti, []).append(float(np.mean(rr[mask])))
        type_mrr_mean = {k: float(np.mean(v)) for k, v in type_mrr.items()}
        worst_types = sorted(type_mrr_mean, key=type_mrr_mean.get)[:3]

        report = {
            "config": cfg.as_dict(),
            "quality_threshold_rel": 0.05,
            "seeds": list(seeds),
            "elapsed_sec": round(elapsed, 3),
            "model_names": model_names,
            "model_mrr_mean": {n: round(model_stats[n][0], 6) for n in model_names},
            "model_mrr_std": {n: round(model_stats[n][1], 6) for n in model_names},
            "fuse_mrr_mean": round(fuse_mean, 6),
            "fuse_mrr_std": round(fuse_std, 6),
            "best_base": best_base,
            "best_base_mrr_mean": round(best_base_mean, 6),
            "delta_abs": round(delta_abs, 6),
            "delta_rel": round(delta_rel, 6),
            "significance_threshold": round(sig_thresh, 6),
            "significant": significant,
            "ablation_drop_one_mrr_mean": {n: round(v, 6) for n, v in abl_mean.items()},
            "ablation_strategy_mrr_mean": {s: round(v, 6) for s, v in strat_mean.items()},
            "relation_type_mrr": {k: round(v, 6) for k, v in type_mrr_mean.items()},
            "worst_relation_types": worst_types,
            "failure_cases": [
                {
                    "relation_type": t,
                    "mrr": round(type_mrr_mean[t], 6),
                    "attribution": _REL_WEAKNESS.get(t, "未知"),
                }
                for t in worst_types
            ],
            "per_seed": [
                {
                    "seed": ps["seed"],
                    "n_train": ps["n_train"],
                    "n_test": ps["n_test"],
                    "per_model_mrr": {n: round(ps["per_model"][n]["mrr"], 6) for n in model_names},
                    "fuse_mrr": round(ps["fuse"]["mrr"], 6),
                    "fuse_hits": ps["fuse"]["hits"],
                    "ablation_mrr": {n: round(ps["ablation"][n], 6) for n in model_names},
                }
                for ps in per_seed
            ],
            "passes_threshold": delta_rel >= 0.05,
        }
        return report


def run_benchmark(cfg: Config, seeds: list[int] | None = None, out_path: str | None = None) -> dict:
    """Run the full KGFuse benchmark and optionally write ``benchmark.json``."""
    pipe = KGFusePipeline(cfg)
    report = pipe.run(seeds=seeds)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    return report
