"""
Aggregate per-task ranks in ``Table1/result/rank/*_rank.csv`` across tasks.

For each Method and each rank metric (MSE_rank, RMSE_rank, MAE_rank, R2_rank, avg_rank),
computes the mean across tasks and the standard error of the mean (SE = sample_std / sqrt(n), ddof=1).

Writes ``Table1/result/rank/summary_avg_rank_by_method.csv``.

Run after ``compute_table1_result_ranks.py`` (or whenever task rank files are updated).
"""

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RANK_DIR = SCRIPT_DIR / "result" / "rank"

RANK_COLS = ["MSE_rank", "RMSE_rank", "MAE_rank", "R2_rank", "avg_rank"]
SUMMARY_NAME = "summary_avg_rank_by_method.csv"


def task_from_rank_filename(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_rank"):
        return stem[: -len("_rank")]
    return stem


def sem_series(s: pd.Series) -> float:
    """Standard error of the mean; NaN if fewer than 2 observations."""
    s = s.dropna()
    n = len(s)
    if n < 2:
        return np.nan
    return float(s.std(ddof=1) / np.sqrt(n))


def load_all_task_ranks() -> pd.DataFrame:
    paths = sorted(p for p in RANK_DIR.glob("*_rank.csv") if p.name != SUMMARY_NAME)
    if not paths:
        raise FileNotFoundError(f"No *_rank.csv under {RANK_DIR}")

    pieces = []
    for path in paths:
        df = pd.read_csv(path)
        if "Method" not in df.columns:
            continue
        missing = [c for c in RANK_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        task = task_from_rank_filename(path)
        chunk = df[["Method", *RANK_COLS]].copy()
        chunk["Task"] = task
        pieces.append(chunk)

    if not pieces:
        raise ValueError("No valid rank files loaded")
    return pd.concat(pieces, ignore_index=True)


def aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, grp in long_df.groupby("Method", sort=False):
        row = {"Method": method, "n_tasks": len(grp)}
        for col in RANK_COLS:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_se"] = sem_series(grp[col])
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("avg_rank_mean", ascending=True).reset_index(drop=True)
    return out


def main():
    long_df = load_all_task_ranks()
    summary = aggregate(long_df)
    out_path = RANK_DIR / SUMMARY_NAME
    summary.to_csv(out_path, index=False)
    print(f"Tasks: {long_df['Task'].nunique()}, rows (method-task): {len(long_df)}")
    print(f"Wrote {out_path} ({len(summary)} methods)")


if __name__ == "__main__":
    main()
