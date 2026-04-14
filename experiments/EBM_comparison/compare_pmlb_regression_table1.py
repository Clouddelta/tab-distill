"""
PMLB regression benchmark: EBM (FBII-selected interactions, n=4), linear regression,
decision tree, XGBoost, TabPFN, and PyGAM (same FBII interactions).

Interaction tuples are read from ``PMLB_reg_interaction/clean_results_*_3way.csv``:
rows with ``Index == fbii`` and ``Num_Interactions == 4``.

Each completed task writes one CSV under ``Table1/result/``.

Dependencies: pmlb, scikit-learn, interpret-core, xgboost, pygam; TabPFN uses a lazy
import of ``torch`` and ``tabpfn_extensions`` (if missing, that row records the error).
Train/test split: ``test_size=0.2``, ``random_state=42`` (same preprocessing style as
``PyGam/interaction/compare_gam_regression_methods.py``).
"""

from __future__ import annotations

import argparse
import ast
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from pmlb import fetch_data
from pygam import LinearGAM, s, te
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
INTERACTION_DIR = SCRIPT_DIR / "PMLB_reg_interaction"
RESULT_DIR = SCRIPT_DIR / "result"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_INTERACTIONS = 4


def extract_task_name(csv_path: Path) -> str:
    stem = csv_path.stem
    prefix = "clean_results_"
    suffix = "_3way"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return stem


def load_fbii_interactions_row(csv_path: Path) -> pd.Series | None:
    df = pd.read_csv(csv_path)
    if "Index" not in df.columns or "Num_Interactions" not in df.columns:
        return None
    idx = df["Index"].astype(str).str.strip().str.lower() == "fbii"
    n = df["Num_Interactions"] == N_INTERACTIONS
    sub = df[idx & n]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def parse_interactions(interactions_str: str) -> list[tuple[int, ...]]:
    try:
        parsed = ast.literal_eval(interactions_str)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    out: list[tuple[int, ...]] = []
    for x in parsed:
        if isinstance(x, (list, tuple)):
            out.append(tuple(int(v) for v in x))
        else:
            out.append((int(x),))
    return out


def load_pmlb_regression_split(
    dataset_name: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    df = fetch_data(dataset_name)
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    cat_cols = [
        col
        for col in X.columns
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category")
    ]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])

    if str(y.dtype) == "object" or "string" in str(y.dtype).lower():
        y = pd.to_numeric(y, errors="coerce")

    mask_valid = ~pd.isna(y)
    X = X.loc[mask_valid].copy()
    y = y.loc[mask_valid].copy()

    if X.isna().any().any():
        X = X.fillna(-1)

    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train = X_train_df.to_numpy(dtype=np.float64)
    X_test = X_test_df.to_numpy(dtype=np.float64)
    y_train = np.asarray(y_train_s.to_numpy(), dtype=np.float64).ravel()
    y_test = np.asarray(y_test_s.to_numpy(), dtype=np.float64).ravel()
    return X_train, y_train, X_test, y_test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def validate_interactions(interactions: list[tuple[int, ...]], n_features: int) -> str | None:
    if len(interactions) != N_INTERACTIONS:
        return f"expected {N_INTERACTIONS} interactions, got {len(interactions)}"
    for inter in interactions:
        if len(inter) < 2 or len(inter) > 4:
            return f"unsupported interaction order {len(inter)}: {inter}"
        if max(inter) >= n_features or min(inter) < 0:
            return f"indices out of range for n_features={n_features}: {inter}"
    return None


def build_gam_terms(n_features: int, interactions: list[tuple[int, ...]], n_splines: int = 10):
    terms = [s(i, n_splines=n_splines) for i in range(n_features)]
    for inter in interactions:
        if len(inter) == 2:
            terms.append(te(inter[0], inter[1], n_splines=n_splines))
        elif len(inter) == 3:
            terms.append(te(inter[0], inter[1], inter[2], n_splines=n_splines))
        elif len(inter) == 4:
            terms.append(
                te(inter[0], inter[1], inter[2], inter[3], n_splines=n_splines)
            )
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    return expr


def _row(
    task: str,
    method: str,
    train_s: float,
    metrics: dict[str, float] | None,
    interactions_repr: str,
    error: str = "",
    *,
    num_interactions_fbii: int | None = None,
) -> dict:
    base = {
        "Task": task,
        "Method": method,
        "Num_Interactions": num_interactions_fbii if num_interactions_fbii is not None else "",
        "Interactions": interactions_repr,
        "Train_Time_s": train_s,
        "MSE": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "R2": np.nan,
        "Error": error,
    }
    if metrics:
        base.update(
            {
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
            }
        )
    return base


def fit_predict_metrics(model, X_train, y_train, X_test, y_test) -> tuple[float, dict[str, float]]:
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return train_s, regression_metrics(y_test, y_pred)


def run_task(task: str, interaction_csv: Path) -> pd.DataFrame:
    row_meta = load_fbii_interactions_row(interaction_csv)
    if row_meta is None:
        raise RuntimeError(f"no fbii row with Num_Interactions={N_INTERACTIONS} in {interaction_csv}")

    inter_str = str(row_meta["Interactions"])
    interactions = parse_interactions(inter_str)
    X_train, y_train, X_test, y_test = load_pmlb_regression_split(task)
    n_features = X_train.shape[1]
    err = validate_interactions(interactions, n_features)
    if err:
        raise RuntimeError(f"{task}: {err}")

    rows: list[dict] = []
    empty_inter = ""

    # --- EBM + FBII interactions ---
    try:
        ebm = ExplainableBoostingRegressor(
            random_state=RANDOM_STATE,
            interactions=interactions,
            outer_bags=4,
            max_bins=256,
        )
        train_s, metrics = fit_predict_metrics(ebm, X_train, y_train, X_test, y_test)
        rows.append(
            _row(
                task,
                "EBM_FBII",
                train_s,
                metrics,
                inter_str,
                num_interactions_fbii=N_INTERACTIONS,
            )
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            _row(
                task,
                "EBM_FBII",
                float("nan"),
                None,
                inter_str,
                error=repr(exc),
                num_interactions_fbii=N_INTERACTIONS,
            )
        )

    # --- Linear regression ---
    try:
        train_s, metrics = fit_predict_metrics(
            LinearRegression(), X_train, y_train, X_test, y_test
        )
        rows.append(_row(task, "LinearRegression", train_s, metrics, empty_inter))
    except Exception as exc:  # noqa: BLE001
        rows.append(
            _row(
                task,
                "LinearRegression",
                float("nan"),
                None,
                empty_inter,
                error=repr(exc),
            )
        )

    # --- Decision tree ---
    try:
        train_s, metrics = fit_predict_metrics(
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            X_train,
            y_train,
            X_test,
            y_test,
        )
        rows.append(_row(task, "DecisionTree", train_s, metrics, empty_inter))
    except Exception as exc:  # noqa: BLE001
        rows.append(
            _row(
                task,
                "DecisionTree",
                float("nan"),
                None,
                empty_inter,
                error=repr(exc),
            )
        )

    # --- XGBoost ---
    try:
        xgb = XGBRegressor(
            random_state=RANDOM_STATE,
            n_estimators=200,
            max_depth=6,
            tree_method="hist",
            n_jobs=-1,
        )
        train_s, metrics = fit_predict_metrics(xgb, X_train, y_train, X_test, y_test)
        rows.append(_row(task, "XGBoost", train_s, metrics, empty_inter))
    except Exception as exc:  # noqa: BLE001
        rows.append(
            _row(task, "XGBoost", float("nan"), None, empty_inter, error=repr(exc))
        )

    # --- TabPFN (lazy import: optional if tabpfn_extensions not installed) ---
    try:
        import torch
        from tabpfn_extensions import TabPFNRegressor

        if torch.cuda.is_available():
            device_to_use = "cuda"
            ignore_limits = False
        else:
            device_to_use = "auto"
            ignore_limits = True
            os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

        t0 = time.perf_counter()
        tab = TabPFNRegressor(device=device_to_use, ignore_pretraining_limits=ignore_limits)
        tab.fit(X_train, y_train)
        train_s = time.perf_counter() - t0
        y_pred = tab.predict(X_test)
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        metrics = regression_metrics(y_test, y_pred)
        rows.append(_row(task, "TabPFN", train_s, metrics, empty_inter))
    except Exception as exc:  # noqa: BLE001
        rows.append(
            _row(task, "TabPFN", float("nan"), None, empty_inter, error=repr(exc))
        )

    # --- PyGAM + same FBII interactions ---
    try:
        terms = build_gam_terms(n_features, interactions, n_splines=10)
        gam = LinearGAM(terms, n_splines=10, max_iter=500)
        train_s, metrics = fit_predict_metrics(gam, X_train, y_train, X_test, y_test)
        rows.append(
            _row(
                task,
                "PyGAM_FBII",
                train_s,
                metrics,
                inter_str,
                num_interactions_fbii=N_INTERACTIONS,
            )
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            _row(
                task,
                "PyGAM_FBII",
                float("nan"),
                None,
                inter_str,
                error=repr(exc),
                num_interactions_fbii=N_INTERACTIONS,
            )
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="PMLB regression task names (default: all tasks with interaction CSVs)",
    )
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INTERACTION_DIR.glob("clean_results_*_3way.csv"))
    task_map = {extract_task_name(p): p for p in files}

    if args.tasks:
        tasks = args.tasks
    else:
        tasks = sorted(task_map.keys())

    for task in tasks:
        path = task_map.get(task)
        if path is None:
            cand = INTERACTION_DIR / f"clean_results_{task}_3way.csv"
            path = cand if cand.exists() else None
        if path is None:
            print(f"Skip {task}: no interaction CSV in {INTERACTION_DIR}")
            continue

        print(f"=== {task} ===", flush=True)
        try:
            out_df = run_task(task, path)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}", flush=True)
            continue

        out_path = RESULT_DIR / f"{task}_regression_method_compare.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  -> wrote {out_path} ({len(out_df)} rows)", flush=True)


if __name__ == "__main__":
    main()
