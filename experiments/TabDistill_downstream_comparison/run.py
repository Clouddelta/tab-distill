"""
Local batch runner for method comparison experiments.

Replaces the CHTC cluster dispatch (HTCondor .sub + .sh files).

Two experiments are run for each task:
  1. SPEX index comparison (compare_index_performance.py):
     Compares EBM performance using interactions from different SPEX indices
     (FBII, FSII, STII, BII, SII, Fourier, Mobius).
     Requires interaction pickles from the interaction_search step.

  2. Baseline comparison (rulefit_baseline_ebm.py):
     Compares EBM with FAST auto-discovered interactions vs RuleFit-selected
     interactions. Standalone -- no pre-computed pickles needed.

Prerequisites:
    pip install interpret imodels openml

Usage:
    python run.py                                      # All tasks, both experiments
    python run.py --experiment baseline                 # Only FAST vs RuleFit
    python run.py --experiment index                    # Only SPEX index comparison
    python run.py --tasks 363615 363698                # Specific tasks
    python run.py --interaction-dir ../interaction_search/interaction_output
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

TABARENA_TASK_IDS = [
    363621,  # blood-transfusion-service-center
    363629,  # diabetes
    363698,  # QSAR_fish_toxicity
    363685,  # maternal_health_risk
    363625,  # concrete_compressive_strength
    363671,  # Fitness_Club
    363612,  # airfoil_self_noise
    363615,  # Another-Dataset-on-used-Fiat-500
    363674,  # hazelnut-spread-contaminant-detection
    363700,  # seismic-bumps
]


def run_one(script: str, task_id: int, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPT_DIR / script), str(task_id), *extra_args]
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    return result.returncode


def run_index_comparison(
    task_ids: list[int],
    interaction_dir: str,
    output_dir: str,
    n_interactions: int = 6,
    max_order: int = 3,
) -> dict[int, bool]:
    results = {}
    for i, tid in enumerate(task_ids, 1):
        print(f"\n{'='*60}")
        print(f"[Index Comparison] Task {i}/{len(task_ids)}: {tid}")
        print(f"{'='*60}")
        t0 = time.time()
        rc = run_one(
            "compare_index_performance.py",
            tid,
            [
                str(n_interactions),
                "--max-interaction-order", str(max_order),
                "--output", f"{output_dir}/{tid}",
                "--interaction-dir", interaction_dir,
                "--no-show",
            ],
        )
        elapsed = time.time() - t0
        ok = rc == 0
        results[tid] = ok
        status = "OK" if ok else f"FAILED (exit {rc})"
        print(f"  -> {status} ({elapsed:.1f}s)")
    return results


def run_baseline_comparison(
    task_ids: list[int],
    output_dir: str,
    n_interactions: int = 10,
    max_order: int = 4,
) -> dict[int, bool]:
    results = {}
    for i, tid in enumerate(task_ids, 1):
        print(f"\n{'='*60}")
        print(f"[Baseline Comparison] Task {i}/{len(task_ids)}: {tid}")
        print(f"{'='*60}")
        t0 = time.time()
        rc = run_one(
            "rulefit_baseline_ebm.py",
            tid,
            [
                str(n_interactions),
                "--max-interaction-order", str(max_order),
                "--output", f"{output_dir}/{tid}",
                "--no-show",
            ],
        )
        elapsed = time.time() - t0
        ok = rc == 0
        results[tid] = ok
        status = "OK" if ok else f"FAILED (exit {rc})"
        print(f"  -> {status} ({elapsed:.1f}s)")
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks", type=int, nargs="+", default=None,
        help="OpenML task IDs (default: all 10 TabArena tasks)",
    )
    parser.add_argument(
        "--experiment", choices=["all", "index", "baseline"], default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--interaction-dir", type=str,
        default=str(SCRIPT_DIR.parent / "interaction_search" / "interaction_output"),
        help="Directory with interaction pickle files from interaction_search step",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(SCRIPT_DIR / "output"),
        help="Base output directory for results",
    )
    args = parser.parse_args()

    task_ids = args.tasks or TABARENA_TASK_IDS
    all_results: dict[str, dict[int, bool]] = {}

    if args.experiment in ("all", "index"):
        print("\n" + "#" * 60)
        print("# SPEX Index Comparison")
        print("#" * 60)
        all_results["index"] = run_index_comparison(
            task_ids, args.interaction_dir, f"{args.output_dir}/index_comparison"
        )

    if args.experiment in ("all", "baseline"):
        print("\n" + "#" * 60)
        print("# Baseline Comparison (FAST vs RuleFit)")
        print("#" * 60)
        all_results["baseline"] = run_baseline_comparison(
            task_ids, f"{args.output_dir}/baseline_comparison"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for exp_name, res in all_results.items():
        ok = sum(1 for v in res.values() if v)
        fail = sum(1 for v in res.values() if not v)
        print(f"  {exp_name}: {ok} succeeded, {fail} failed out of {len(res)} tasks")
        if fail > 0:
            failed_ids = [tid for tid, v in res.items() if not v]
            print(f"    Failed tasks: {failed_ids}")


if __name__ == "__main__":
    main()
