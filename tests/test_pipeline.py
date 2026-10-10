"""Pipeline tests: benchmark structure, determinism, performance gate, CLI smoke.

Author: 晨星
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kgforge.pipeline.kgfuse import KGFuse, run_benchmark

ROOT = Path(__file__).resolve().parents[1]


def _bench_cfg(**kw):
    from conftest import small_cfg

    base = dict(
        dim=16,
        epochs=120,
        batch_size=64,
        n_entities=150,
        n_relations=8,
        n_train=500,
        n_valid=100,
        n_test=100,
    )
    base.update(kw)
    return small_cfg(**base)


def test_benchmark_report_structure():
    rep = run_benchmark(_bench_cfg(), seeds=[7, 42])
    for key in (
        "model_mrr_mean",
        "fuse_mrr_mean",
        "best_base",
        "delta_rel",
        "ablation_drop_one_mrr_mean",
        "ablation_strategy_mrr_mean",
        "relation_type_mrr",
        "failure_cases",
    ):
        assert key in rep, f"missing report key {key}"
    assert rep["model_names"] == ["transe", "distmult", "complex", "rotate"]
    assert len(rep["per_seed"]) == 2


def test_determinism_identical_numbers_for_same_seed():
    cfg = _bench_cfg()
    a = run_benchmark(cfg, seeds=[7])
    b = run_benchmark(cfg, seeds=[7])
    assert a["fuse_mrr_mean"] == b["fuse_mrr_mean"]
    for n in a["model_names"]:
        assert a["model_mrr_mean"][n] == b["model_mrr_mean"][n]


@pytest.mark.slow
def test_fusion_beats_strongest_single_model():
    """Performance gate: KGFuse must beat the best base model by >= 5% rel."""
    rep = run_benchmark(_bench_cfg(), seeds=[7, 42, 123])
    assert rep["best_base"] in rep["model_names"]
    assert rep["delta_rel"] >= 0.05, (
        f"KGFuse {rep['fuse_mrr_mean']:.4f} vs best base {rep['best_base']} "
        f"{rep['best_base_mrr_mean']:.4f}: rel {rep['delta_rel'] * 100:+.2f}% < 5%"
    )
    assert rep["passes_threshold"] is True


def test_max_consensus_is_a_documented_negative_result():
    """max_consensus must lose to the flagship (kept as an honest ablation)."""
    rep = run_benchmark(_bench_cfg(), seeds=[7])
    strat = rep["ablation_strategy_mrr_mean"]
    assert set(strat) >= {"mean_rr", "rrf", "max_consensus"}
    assert strat["rrf"] > strat["max_consensus"]


def test_drop_one_model_ablation_is_reported():
    rep = run_benchmark(_bench_cfg(), seeds=[7])
    assert len(rep["ablation_drop_one_mrr_mean"]) == 4
    for v in rep["ablation_drop_one_mrr_mean"].values():
        assert 0.0 < v <= 1.0


def test_failure_cases_present_and_derived_from_results():
    rep = run_benchmark(_bench_cfg(), seeds=[7])
    assert len(rep["failure_cases"]) >= 1
    for fc in rep["failure_cases"]:
        assert fc["relation_type"] in rep["relation_type_mrr"]
        assert fc["attribution"]


def test_cli_benchmark_smoke():
    env = dict(os.environ)
    env.update(
        {
            "KG_DIM": "8",
            "KG_EPOCHS": "3",
            "KG_N_ENTITIES": "60",
            "KG_N_RELATIONS": "4",
            "KG_N_TRAIN": "200",
            "KG_N_VALID": "40",
            "KG_N_TEST": "40",
            "KG_BATCH_SIZE": "32",
        }
    )
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "kgforge.cli",
            "benchmark",
            "--seeds",
            "7",
            "--out",
            str(ROOT / "_tmp_bench.json"),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "KGFuse MRR=" in out.stdout


def test_pipeline_rejects_empty_model_list():
    with pytest.raises(ValueError):
        KGFuse([])
    with pytest.raises(ValueError):
        from kgforge.models import MODEL_REGISTRY

        KGFuse([MODEL_REGISTRY["transe"](_bench_cfg())], strategy="nope")
