"""
Run the full Table 1 benchmark pipeline.

This script sequentially executes:
  1. compare_pmlb_regression_table1.py
     Benchmarks EBM (with FBII interactions), LinearRegression, DecisionTree,
     XGBoost, TabPFN, and PyGAM on PMLB regression datasets.
     Reads interactions from PMLB_reg_interaction/clean_results_*_3way.csv.
     Writes per-task CSVs under result/.

  2. compute_table1_result_ranks.py
     Ranks methods per task (lower rank = better for error metrics, higher R2).
     Writes per-task rank CSVs under result/rank/.

  3. aggregate_table1_ranks.py
     Aggregates per-task ranks into a summary table with mean and SEM.
     Writes result/rank/summary_avg_rank_by_method.csv.

Prerequisites:
    pip install interpret pmlb pygam scikit-learn xgboost tabpfn-extensions

Usage:
    python run.py                  # Full pipeline
    python run.py --steps 2 3      # Only ranking + aggregation (skip model fitting)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PIPELINE = [
    ("compare_pmlb_regression_table1.py", "Fit models and compare methods on PMLB datasets"),
    ("compute_table1_result_ranks.py", "Compute per-task method ranks"),
    ("aggregate_table1_ranks.py", "Aggregate ranks across tasks"),
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=None,
        help="Which steps to run (1-indexed). Default: all three steps.",
    )
    args = parser.parse_args()

    steps = args.steps or list(range(1, len(PIPELINE) + 1))

    for step_num in steps:
        if step_num < 1 or step_num > len(PIPELINE):
            print(f"Invalid step {step_num} (valid: 1-{len(PIPELINE)})")
            sys.exit(1)

        script, description = PIPELINE[step_num - 1]

        print(f"\n{'='*60}")
        print(f"Step {step_num}/{len(PIPELINE)}: {description}")
        print(f"  Script: {script}")
        print(f"{'='*60}\n")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / script)],
            cwd=str(SCRIPT_DIR),
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\nStep {step_num} failed with exit code {result.returncode} ({elapsed:.1f}s)")
            sys.exit(result.returncode)

        print(f"\nStep {step_num} completed ({elapsed:.1f}s)")

    print(f"\n{'='*60}")
    print("Table 1 pipeline complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
