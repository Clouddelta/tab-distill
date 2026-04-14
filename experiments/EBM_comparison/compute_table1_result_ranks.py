"""
Build per-task method ranks from ``Table1/result/*_regression_method_compare.csv``.

Ranks: lower is better for MSE, RMSE, MAE; higher R2 is better (rank 1 = best).
Ties use average ranks. Rows with missing metrics or non-empty ``Error`` are excluded.

Writes ``Table1/result/rank/{task}_rank.csv`` (Method + *_rank + avg_rank).
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "result"
RANK_DIR = RESULT_DIR / "rank"

COMPARE_SUFFIX = "_regression_method_compare"


def task_name_from_compare_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith(COMPARE_SUFFIX):
        return stem[: -len(COMPARE_SUFFIX)]
    return stem


def valid_metric_rows(df: pd.DataFrame) -> pd.Series:
    err = df.get("Error", pd.Series("", index=df.index))
    err = err.fillna("").astype(str).str.strip()
    ok_err = err.eq("")
    for col in ("MSE", "RMSE", "MAE", "R2"):
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        ok_err = ok_err & df[col].notna()
    return ok_err


def rank_one_task(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.loc[valid_metric_rows(df)].copy()
    if len(sub) == 0:
        return pd.DataFrame(
            columns=["Method", "MSE_rank", "RMSE_rank", "MAE_rank", "R2_rank", "avg_rank"]
        )

    sub["MSE_rank"] = sub["MSE"].rank(method="average", ascending=True)
    sub["RMSE_rank"] = sub["RMSE"].rank(method="average", ascending=True)
    sub["MAE_rank"] = sub["MAE"].rank(method="average", ascending=True)
    sub["R2_rank"] = sub["R2"].rank(method="average", ascending=False)
    sub["avg_rank"] = sub[
        ["MSE_rank", "RMSE_rank", "MAE_rank", "R2_rank"]
    ].mean(axis=1)

    out = sub[
        ["Method", "MSE_rank", "RMSE_rank", "MAE_rank", "R2_rank", "avg_rank"]
    ].sort_values("avg_rank", ascending=True)
    return out.reset_index(drop=True)


def main():
    RANK_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(RESULT_DIR.glob(f"*{COMPARE_SUFFIX}.csv"))
    if not paths:
        print(f"No files matching *{COMPARE_SUFFIX}.csv under {RESULT_DIR}")
        return

    for path in paths:
        task = task_name_from_compare_path(path)
        df = pd.read_csv(path)
        ranked = rank_one_task(df)
        out_path = RANK_DIR / f"{task}_rank.csv"
        ranked.to_csv(out_path, index=False)
        print(f"{task}: {len(ranked)} methods -> {out_path}")


if __name__ == "__main__":
    main()
