#!/usr/bin/env python
"""
Compare performance between Baseline EBM and RuleFit-EBM models.

Usage:
    python rulefit_baseline_ebm.py <task_id> <N_interactions> [--output OUTPUT_PATH]

Example:
    python rulefit_baseline_ebm.py 359950 6 --output comparison_plot.png
"""

import argparse
import os
import re
import time
import openml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, log_loss
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from imodels import RuleFitRegressor
from typing import cast


def load_openml_data(task_id):
    """Load data from OpenML and determine task type."""
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    print(f"Task ID: {task_id}")
    print(f"Dataset ID: {dataset.id}, Dataset Name: {dataset.name}")

    task_type_id = getattr(task, 'task_type_id', None)
    if task_type_id == 1:
        task_type = "classification"
    elif task_type_id == 2:
        task_type = "regression"
    else:
        eval_measure = getattr(task, 'evaluation_measure', None)
        if eval_measure is not None:
            eval_measure = str(eval_measure).lower()
            if 'auc' in eval_measure or 'accuracy' in eval_measure or 'log_loss' in eval_measure:
                task_type = "classification"
            elif 'rmse' in eval_measure or 'mse' in eval_measure or 'r2' in eval_measure or 'mae' in eval_measure:
                task_type = "regression"
            else:
                task_type = None
        else:
            task_type = None

    print(f"Task type (from OpenML): {task_type} (task_type_id: {task_type_id})")

    target_name = getattr(task, 'target_name', None) or dataset.default_target_attribute

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=target_name,
        dataset_format="dataframe"
    )

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    cat_cols = [
        col for col in X.columns
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category")
    ]
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols])
        print(f"\nEncoded {len(cat_cols)} categorical features")

    train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0)

    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)

    y_train_raw = y.iloc[train_indices]
    y_test_raw = y.iloc[test_indices]

    if task_type is None:
        y_series = pd.Series(y_train_raw) if not isinstance(y_train_raw, pd.Series) else y_train_raw
        y_unique = y_series.nunique()
        y_total = len(y_series)
        y_dtype_str = str(y_series.dtype)
        if y_dtype_str == 'object' or 'string' in y_dtype_str.lower() or (y_unique < 20 and y_unique / y_total < 0.1):
            task_type = "classification"
        else:
            task_type = "regression"
        print(f"Task type (inferred from data): {task_type} (unique values: {y_unique}/{y_total})")

    if task_type == "classification":
        y_train_series = pd.Series(y_train_raw) if not isinstance(y_train_raw, pd.Series) else y_train_raw
        y_test_series = pd.Series(y_test_raw) if not isinstance(y_test_raw, pd.Series) else y_test_raw

        try:
            y_train_test_series = pd.to_numeric(y_train_series, errors='coerce')
            has_nan = pd.isna(y_train_test_series).any() if isinstance(y_train_test_series, pd.Series) else False

            if has_nan:
                raise ValueError("Contains non-numeric values")

            y_train = np.asarray(pd.to_numeric(y_train_series, errors='coerce'), dtype=np.int64)
            y_test = np.asarray(pd.to_numeric(y_test_series, errors='coerce'), dtype=np.int64)
            unique_labels = np.unique(y_train)
            if len(unique_labels) > 0 and unique_labels.min() != 0:
                label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                y_train = np.array([label_map.get(label, -1) for label in y_train], dtype=np.int64)
                y_test = np.array([label_map.get(label, -1) for label in y_test], dtype=np.int64)
                print(f"\nMapped labels to start from 0. Original labels: {unique_labels}")
        except (ValueError, TypeError, AttributeError):
            label_encoder = LabelEncoder()
            y_train_str = y_train_series.astype(str).values
            y_test_str = y_test_series.astype(str).values
            y_train = label_encoder.fit_transform(y_train_str)
            y_test = label_encoder.transform(y_test_str)
            print(f"\nEncoded labels using LabelEncoder. Classes: {label_encoder.classes_}")
    else:
        y_train_series = pd.Series(y_train_raw) if not isinstance(y_train_raw, pd.Series) else y_train_raw
        y_test_series = pd.Series(y_test_raw) if not isinstance(y_test_raw, pd.Series) else y_test_raw

        y_train_dtype_str = str(y_train_series.dtype)
        if 'object' in y_train_dtype_str or 'string' in y_train_dtype_str.lower():
            y_train = np.asarray(pd.to_numeric(y_train_series, errors="coerce"), dtype=np.float64)
            y_test = np.asarray(pd.to_numeric(y_test_series, errors="coerce"), dtype=np.float64)
            print("\nConverted labels to numeric for regression")
        else:
            y_train = np.asarray(y_train_series.values, dtype=np.float64).ravel()
            y_test = np.asarray(y_test_series.values, dtype=np.float64).ravel()

    y_train = y_train.flatten() if y_train.ndim > 1 else y_train
    y_test = y_test.flatten() if y_test.ndim > 1 else y_test

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"y_train dtype: {y_train.dtype}, shape: {y_train.shape}, unique values: {np.unique(y_train)[:10]}")
    print(f"y_test dtype: {y_test.dtype}, shape: {y_test.shape}, unique values: {np.unique(y_test)[:10]}")

    return X_train, y_train, X_test, y_test, attribute_names, task_type


def _df_to_numeric_matrix_preserve_columns(X_df: pd.DataFrame) -> np.ndarray:
    """Convert a DataFrame to a numeric numpy matrix without changing column count/order."""
    X_num = X_df.copy()
    for col in X_num.columns:
        s = X_num[col]
        if pd.api.types.is_categorical_dtype(s):
            X_num[col] = s.cat.codes
        else:
            X_num[col] = pd.to_numeric(s, errors="coerce")

    X_num = X_num.astype(float)
    med = np.nanmedian(X_num.to_numpy(), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    arr = X_num.to_numpy(dtype=float)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = np.take(med, np.where(nan_mask)[1])
    return arr


def train_baseline_ebm_models(X_train, y_train, X_test, y_test, n_interactions, task_type="regression"):
    """Train Baseline EBM models with auto-discovered interactions."""
    baseline_ebm_models = []
    baseline_all_results = []

    print("\n" + "="*80)
    print(f"=== Training Baseline EBM Models ({task_type}) with Different Number of Interactions ===\n")

    for num_interactions in range(1, n_interactions + 1):
        print(f"Model {num_interactions}/{n_interactions}: Training EBM with {num_interactions} interaction(s) (auto-discovered)")

        if task_type == "classification":
            ebm = ExplainableBoostingClassifier(
                random_state=42,
                interactions=num_interactions,
                outer_bags=4,
                max_bins=256
            )
        else:
            ebm = ExplainableBoostingRegressor(
                random_state=42,
                interactions=num_interactions,
                outer_bags=4,
                max_bins=256
            )

        t0 = time.perf_counter()
        ebm.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0
        baseline_ebm_models.append(ebm)

        ebm_test_pred = ebm.predict(X_test)

        if task_type == "classification":
            ebm_test_acc = accuracy_score(y_test, ebm_test_pred)
            ebm_test_pred_proba = ebm.predict_proba(X_test)
            ebm_test_logloss = log_loss(y_test, ebm_test_pred_proba)

            result_dict = {
                'Model': f'Model {num_interactions}',
                'Num_Interactions': num_interactions,
                'Train_Time_s': train_time_s,
                'Accuracy': ebm_test_acc,
                'Log_Loss': ebm_test_logloss
            }
            print(f"  - Accuracy: {ebm_test_acc:.4f}, Log Loss: {ebm_test_logloss:.4f}\n")
        else:
            ebm_test_mse = mean_squared_error(y_test, ebm_test_pred)
            ebm_test_mae = mean_absolute_error(y_test, ebm_test_pred)
            ebm_test_r2 = r2_score(y_test, ebm_test_pred)
            ebm_test_rmse = np.sqrt(ebm_test_mse)

            result_dict = {
                'Model': f'Model {num_interactions}',
                'Num_Interactions': num_interactions,
                'Train_Time_s': train_time_s,
                'MSE': ebm_test_mse,
                'RMSE': ebm_test_rmse,
                'MAE': ebm_test_mae,
                'R2': ebm_test_r2
            }
            print(f"  - MSE: {ebm_test_mse:.4f}, RMSE: {ebm_test_rmse:.4f}, MAE: {ebm_test_mae:.4f}, R2: {ebm_test_r2:.4f}\n")

        discovered_interactions = []
        try:
            if hasattr(ebm, 'term_features_') and ebm.term_features_ is not None:
                terms = ebm.term_features_
                all_interactions = [t for t in terms if len(t) >= 2]
                discovered_interactions = [
                    tuple(sorted(interaction)) if isinstance(interaction, (list, tuple)) else interaction
                    for interaction in all_interactions[:num_interactions]
                ]
            elif hasattr(ebm, 'term_names_'):
                n_features = X_train.shape[1]
                term_names = ebm.term_names_
                interaction_terms = term_names[n_features:]
                for term in interaction_terms[:num_interactions]:
                    if ' & ' in term:
                        parts = term.split(' & ')
                        feature_indices = []
                        for part in parts:
                            part = part.strip()
                            if part.startswith('feature_'):
                                try:
                                    idx = int(part.replace('feature_', ''))
                                    feature_indices.append(idx)
                                except ValueError:
                                    pass
                            else:
                                try:
                                    idx = int(part)
                                    if 0 <= idx < n_features:
                                        feature_indices.append(idx)
                                except ValueError:
                                    pass
                        if len(feature_indices) > 1:
                            feature_indices.sort()
                            discovered_interactions.append(tuple(feature_indices))
        except Exception:
            pass

        interactions_str = str(discovered_interactions) if discovered_interactions else ''
        result_dict['Interactions'] = interactions_str

        baseline_all_results.append(result_dict)

    return baseline_ebm_models, baseline_all_results


def print_performance_table(results, title, task_type="regression"):
    """Print performance comparison table."""
    print("\n" + "="*80)
    print(title)
    print("="*80)

    results_df = pd.DataFrame(results)

    print("\n" + "-"*80)
    has_time = 'Train_Time_s' in results_df.columns
    if task_type == "classification":
        if has_time:
            print(f"{'Model':<12} {'# Interact':<12} {'Time(s)':<10} {'Accuracy':<12} {'Log_Loss':<12}")
        else:
            print(f"{'Model':<12} {'# Interact':<12} {'Accuracy':<12} {'Log_Loss':<12}")
        print("-"*80)
        for _, row in results_df.iterrows():
            if has_time:
                tt_val = row.get('Train_Time_s', np.nan)
                tt = float(tt_val) if tt_val is not None and not pd.isna(tt_val) else float('nan')
                acc = row.get('Accuracy', np.nan)
                logloss = row.get('Log_Loss', np.nan)
                print(f"{row['Model']:<12} {row['Num_Interactions']:<12} {tt:<10.2f} {acc:<12.4f} {logloss:<12.4f}")
            else:
                acc = row.get('Accuracy', np.nan)
                logloss = row.get('Log_Loss', np.nan)
                print(f"{row['Model']:<12} {row['Num_Interactions']:<12} {acc:<12.4f} {logloss:<12.4f}")
    else:
        if has_time:
            print(f"{'Model':<12} {'# Interact':<12} {'Time(s)':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R2':<12}")
        else:
            print(f"{'Model':<12} {'# Interact':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R2':<12}")
        print("-"*80)
        for _, row in results_df.iterrows():
            if has_time:
                tt_val = row.get('Train_Time_s', np.nan)
                tt = float(tt_val) if tt_val is not None and not pd.isna(tt_val) else float('nan')
                print(f"{row['Model']:<12} {row['Num_Interactions']:<12} {tt:<10.2f} {row['MSE']:<12.4f} {row['RMSE']:<12.4f} {row['MAE']:<12.4f} {row['R2']:<12.4f}")
            else:
                print(f"{row['Model']:<12} {row['Num_Interactions']:<12} {row['MSE']:<12.4f} {row['RMSE']:<12.4f} {row['MAE']:<12.4f} {row['R2']:<12.4f}")
    print("-"*80)
    print()

    return results_df


def replace_feature_names_with_indices(rule_str, attribute_names_list):
    """Replace feature names in rule string with feature indices."""
    if pd.isna(rule_str) or rule_str == '':
        return rule_str

    result = rule_str

    names_with_indices = [(name, idx) for idx, name in enumerate(attribute_names_list)]
    sorted_names_with_indices = sorted(names_with_indices, key=lambda x: len(x[0]), reverse=True)

    for name, idx in sorted_names_with_indices:
        if name not in result:
            continue
        escaped_name = re.escape(name)
        result = re.sub(escaped_name, str(idx), result)

    return result


def extract_feature_indices(rule_str):
    """Extract feature indices from rule string and return as tuple sorted by feature index."""
    if pd.isna(rule_str) or rule_str == '':
        return tuple()

    if rule_str.strip().isdigit():
        return (int(rule_str.strip()),)

    pattern = r'(\d+)\s*(?:<=|>=|<|>|==)'
    matches = re.findall(pattern, rule_str)

    if matches:
        seen = set()
        indices = []
        for m in matches:
            idx = int(m)
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        indices.sort()
        return tuple(indices)

    return tuple()


def get_rulefit_interactions(
    X_train,
    y_train,
    attribute_names,
    N_interactions,
    sort_by: str = "importance",
    max_interaction_order: int = 2,
    task_type: str = "regression",
):
    """Extract top N interactions from RuleFit model."""
    print("\n" + "="*80)
    if max_interaction_order not in (2, 3, 4):
        raise ValueError(f"max_interaction_order must be 2, 3, or 4, got: {max_interaction_order}")
    print(f"=== Extracting Interactions from RuleFit Model (sort_by={sort_by}, max_order={max_interaction_order}, task_type={task_type}) ===\n")

    if isinstance(X_train, pd.DataFrame):
        X_train_rf = _df_to_numeric_matrix_preserve_columns(X_train)
    else:
        X_train_rf = np.asarray(X_train, dtype=float)

    if task_type == "classification":
        y_train_rf = y_train.astype(float)
        print("Note: RuleFit only supports regression. Converting classification labels to float for interaction extraction.")
    else:
        y_train_rf = y_train

    rf = RuleFitRegressor(max_rules=3000, random_state=42)
    rf.fit(X_train_rf, y_train_rf, feature_names=attribute_names)

    rules_df = cast(pd.DataFrame, rf._get_rules(exclude_zero_coef=True))
    if sort_by not in {"importance", "support"}:
        raise ValueError(f"sort_by must be 'importance' or 'support', got: {sort_by}")
    rules_df = rules_df.sort_values(sort_by, ascending=False)

    rules_df['rule'] = rules_df['rule'].apply(
        lambda x: replace_feature_names_with_indices(x, attribute_names)
    )

    rules_df['feature_indices'] = rules_df['rule'].apply(extract_feature_indices)

    summary = rules_df[['feature_indices', 'coef', 'support', 'importance']].copy()
    summary.index.name = 'rule_index'

    seen_indices = set()
    rulefit_interactions = []

    for idx, row in summary.iterrows():
        feature_tuple = row['feature_indices']

        if len(feature_tuple) < 2 or len(feature_tuple) > max_interaction_order:
            continue

        if feature_tuple not in seen_indices:
            seen_indices.add(feature_tuple)
            rulefit_interactions.append(feature_tuple)

            if len(rulefit_interactions) >= N_interactions:
                break

    unique_count = len(rulefit_interactions)
    if unique_count < N_interactions:
        if unique_count > 0:
            last_interaction = rulefit_interactions[-1]
            while len(rulefit_interactions) < N_interactions:
                rulefit_interactions.append(last_interaction)
            print(f"Warning: Only found {unique_count} unique interactions, padding with the last interaction to reach {N_interactions}.\n")
        else:
            raise ValueError("No valid 2-way to 3-way interactions found in RuleFit model!")

    if max_interaction_order == 2:
        desc = "2-way only"
    else:
        desc = f"2-way to {max_interaction_order}-way"
    print(f"Selected Top {N_interactions} Interactions ({desc}, skipping duplicates):")
    for i, interaction in enumerate(rulefit_interactions[:N_interactions], 1):
        print(f"  {i}. {interaction}")
    print()

    return rulefit_interactions


def train_rulefit_ebm_models(X_train, y_train, X_test, y_test, rulefit_interactions, N_interactions, task_type="regression"):
    """Train RuleFit-EBM models with RuleFit-discovered interactions."""
    ebm_models = []
    all_results = []

    print("\n" + "="*80)
    print(f"=== Training RuleFit-EBM Models ({task_type}) with Different Number of Interactions ===\n")

    for num_interactions in range(1, N_interactions + 1):
        current_interactions = rulefit_interactions[:num_interactions]

        print(f"Model {num_interactions}/{N_interactions}: Using {num_interactions} interaction(s): {current_interactions}")

        if task_type == "classification":
            ebm = ExplainableBoostingClassifier(
                random_state=42,
                interactions=current_interactions,
                outer_bags=4,
                max_bins=256
            )
        else:
            ebm = ExplainableBoostingRegressor(
                random_state=42,
                interactions=current_interactions,
                outer_bags=4,
                max_bins=256
            )

        t0 = time.perf_counter()
        ebm.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0
        ebm_models.append(ebm)

        y_pred = ebm.predict(X_test)

        if task_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            y_pred_proba = ebm.predict_proba(X_test)
            logloss = log_loss(y_test, y_pred_proba)

            all_results.append({
                'Model': f'Model {num_interactions}',
                'Num_Interactions': num_interactions,
                'Interactions': str(current_interactions),
                'Train_Time_s': train_time_s,
                'Accuracy': acc,
                'Log_Loss': logloss
            })
            print(f"  - Accuracy: {acc:.4f}, Log Loss: {logloss:.4f}\n")
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            all_results.append({
                'Model': f'Model {num_interactions}',
                'Num_Interactions': num_interactions,
                'Interactions': str(current_interactions),
                'Train_Time_s': train_time_s,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
            print(f"  - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}\n")

    return ebm_models, all_results


def _resolve_output_dir(output_path):
    """Interpret output_path like the notebook: filename -> '.', directory -> itself, path/filename -> dirname."""
    if not output_path:
        return None

    dir_part = os.path.dirname(output_path)
    image_extensions = ('.png', '.jpg', '.jpeg', '.pdf', '.svg')
    has_image_ext = output_path.lower().endswith(image_extensions)

    if dir_part:
        return dir_part
    if has_image_ext:
        return '.'
    return output_path


def plot_comparison(
    baseline_all_results,
    rulefit_support_all_results,
    task_id,
    output_path=None,
    show=True,
    file_suffix: str | None = None,
    task_type="regression",
):
    """Create and save comparison plots for Baseline EBM and RuleFit-Support methods."""
    baseline_df = pd.DataFrame(baseline_all_results)
    baseline_interactions = np.array(baseline_df['Num_Interactions'].values)

    rulefit_support_df = pd.DataFrame(rulefit_support_all_results)
    rulefit_support_interactions = np.array(rulefit_support_df['Num_Interactions'].values)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Performance Comparison: Baseline EBM vs RuleFit-Support ({task_type})', fontsize=16, fontweight='bold', y=0.995)

    baseline_color = '#A23B72'
    rulefit_support_color = '#2D6A4F'
    baseline_marker = 's'
    rulefit_support_marker = 'D'

    max_interactions = max(len(baseline_interactions), len(rulefit_support_interactions))

    if task_type == "classification":
        baseline_acc = np.array(baseline_df['Accuracy'].values)
        baseline_logloss = np.array(baseline_df['Log_Loss'].values)
        rulefit_support_acc = np.array(rulefit_support_df['Accuracy'].values)
        rulefit_support_logloss = np.array(rulefit_support_df['Log_Loss'].values)

        ax1 = axes[0, 0]
        ax1.plot(baseline_interactions, baseline_acc, marker=baseline_marker, linewidth=2, markersize=7,
                 label='Baseline EBM', color=baseline_color)
        ax1.plot(rulefit_support_interactions, rulefit_support_acc, marker=rulefit_support_marker, linewidth=2, markersize=7,
                 label='RuleFit-Support', color=rulefit_support_color)
        ax1.set_xlabel('Number of Interactions', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax1.set_title('Accuracy', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(range(1, max_interactions + 1))

        ax2 = axes[0, 1]
        ax2.plot(baseline_interactions, baseline_logloss, marker=baseline_marker, linewidth=2, markersize=7,
                 label='Baseline EBM', color=baseline_color)
        ax2.plot(rulefit_support_interactions, rulefit_support_logloss, marker=rulefit_support_marker, linewidth=2, markersize=7,
                 label='RuleFit-Support', color=rulefit_support_color)
        ax2.set_xlabel('Number of Interactions', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Log Loss', fontsize=11, fontweight='bold')
        ax2.set_title('Log Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticks(range(1, max_interactions + 1))

        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
    else:
        baseline_mse = np.array(baseline_df['MSE'].values)
        baseline_rmse = np.array(baseline_df['RMSE'].values)
        baseline_mae = np.array(baseline_df['MAE'].values)
        baseline_r2 = np.array(baseline_df['R2'].values)
        rulefit_support_mse = np.array(rulefit_support_df['MSE'].values)
        rulefit_support_rmse = np.array(rulefit_support_df['RMSE'].values)
        rulefit_support_mae = np.array(rulefit_support_df['MAE'].values)
        rulefit_support_r2 = np.array(rulefit_support_df['R2'].values)

        ax1 = axes[0, 0]
        ax1.plot(baseline_interactions, baseline_mse, marker=baseline_marker, linewidth=2, markersize=7,
                 label='Baseline EBM', color=baseline_color)
        ax1.plot(rulefit_support_interactions, rulefit_support_mse, marker=rulefit_support_marker, linewidth=2, markersize=7,
                 label='RuleFit-Support', color=rulefit_support_color)
        ax1.set_xlabel('Number of Interactions', fontsize=11, fontweight='bold')
        ax1.set_ylabel('MSE', fontsize=11, fontweight='bold')
        ax1.set_title('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(range(1, max_interactions + 1))

        ax2 = axes[0, 1]
        ax2.plot(baseline_interactions, baseline_rmse, marker=baseline_marker, linewidth=2, markersize=7,
                 label='Baseline EBM', color=baseline_color)
        ax2.plot(rulefit_support_interactions, rulefit_support_rmse, marker=rulefit_support_marker, linewidth=2, markersize=7,
                 label='RuleFit-Support', color=rulefit_support_color)
        ax2.set_xlabel('Number of Interactions', fontsize=11, fontweight='bold')
        ax2.set_ylabel('RMSE', fontsize=11, fontweight='bold')
        ax2.set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticks(range(1, max_interactions + 1))

        ax3 = axes[1, 0]
        ax3.plot(baseline_interactions, baseline_mae, marker=baseline_marker, linewidth=2, markersize=7,
                 label='Baseline EBM', color=baseline_color)
        ax3.plot(rulefit_support_interactions, rulefit_support_mae, marker=rulefit_support_marker, linewidth=2, markersize=7,
                 label='RuleFit-Support', color=rulefit_support_color)
        ax3.set_xlabel('Number of Interactions', fontsize=11, fontweight='bold')
        ax3.set_ylabel('MAE', fontsize=11, fontweight='bold')
        ax3.set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xticks(range(1, max_interactions + 1))

        ax4 = axes[1, 1]
        ax4.plot(baseline_interactions, baseline_r2, marker=baseline_marker, linewidth=2, markersize=7,
                 label='Baseline EBM', color=baseline_color)
        ax4.plot(rulefit_support_interactions, rulefit_support_r2, marker=rulefit_support_marker, linewidth=2, markersize=7,
                 label='RuleFit-Support', color=rulefit_support_color)
        ax4.set_xlabel('Number of Interactions', fontsize=11, fontweight='bold')
        ax4.set_ylabel('R2 Score', fontsize=11, fontweight='bold')
        ax4.set_title('Coefficient of Determination (R2)', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10, loc='best')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xticks(range(1, max_interactions + 1))

    plt.tight_layout(rect=(0, 0, 1, 0.99))

    if output_path:
        output_dir = _resolve_output_dir(output_path) or '.'
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_{file_suffix}" if file_suffix else ""
        save_path = os.path.join(output_dir, f'comparison_{task_id}{suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Compare performance between Baseline EBM and RuleFit-EBM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('task_id', type=int, help='OpenML task ID')
    parser.add_argument('N_interactions', type=int, help='Number of interactions to use')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for the plot (default: show plot)')
    parser.add_argument(
        '--max-interaction-order',
        type=int,
        choices=[2, 3, 4],
        default=2,
        help='Interaction order to use for RuleFit: 2 (2-way), 3 (2+3-way), 4 (2+3+4-way). Baseline EBM unchanged.',
    )
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display matplotlib windows (useful when saving plots)')

    args = parser.parse_args()
    order_tag = f"{args.max_interaction_order}way"

    print("Loading OpenML data...")
    X_train, y_train, X_test, y_test, attribute_names, task_type = load_openml_data(args.task_id)

    baseline_ebm_models, baseline_all_results = train_baseline_ebm_models(
        X_train, y_train, X_test, y_test, args.N_interactions, task_type
    )

    baseline_results_df = print_performance_table(
        baseline_all_results, "Baseline EBM Performance Comparison Table", task_type
    )

    rulefit_support_interactions = get_rulefit_interactions(
        X_train, y_train, attribute_names, args.N_interactions,
        sort_by="support",
        max_interaction_order=args.max_interaction_order,
        task_type=task_type,
    )

    rulefit_support_ebm_models, rulefit_support_all_results = train_rulefit_ebm_models(
        X_train, y_train, X_test, y_test, rulefit_support_interactions, args.N_interactions, task_type
    )

    rulefit_support_results_df = print_performance_table(
        rulefit_support_all_results, "RuleFit-Support-EBM Performance Comparison Table", task_type
    )

    last_baseline = baseline_ebm_models[-1]
    last_pred = last_baseline.predict(X_test)

    print(f"\n=== Baseline EBM (Last Model: {baseline_results_df.iloc[-1]['Num_Interactions']} interactions) ===")
    print(f"Sample number: {int(np.asarray(y_test).shape[0])}")

    if task_type == "classification":
        last_acc = accuracy_score(y_test, last_pred)
        last_pred_proba = last_baseline.predict_proba(X_test)
        last_logloss = log_loss(y_test, last_pred_proba)
        print(f"Accuracy: {last_acc:.4f}")
        print(f"Log Loss: {last_logloss:.4f}")
    else:
        last_mse = mean_squared_error(y_test, last_pred)
        last_mae = mean_absolute_error(y_test, last_pred)
        last_r2 = r2_score(y_test, last_pred)
        last_rmse = np.sqrt(last_mse)
        print(f"MSE: {last_mse:.4f}")
        print(f"RMSE: {last_rmse:.4f}")
        print(f"MAE: {last_mae:.4f}")
        print(f"R2: {last_r2:.4f}")

    baseline_results_df['Model_Type'] = 'Baseline-EBM'
    rulefit_support_results_df['Model_Type'] = 'RuleFit-Support-EBM'

    combined_df = pd.concat([baseline_results_df, rulefit_support_results_df], ignore_index=True)

    if args.output:
        dir_part = os.path.dirname(args.output)
        image_extensions = ('.png', '.jpg', '.jpeg', '.pdf', '.svg')
        has_image_ext = args.output.lower().endswith(image_extensions)

        if dir_part:
            output_dir = dir_part
        elif has_image_ext:
            output_dir = '.'
        else:
            output_dir = args.output
    else:
        output_dir = '.'

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f'results_{args.task_id}_{order_tag}.csv')
    combined_df.to_csv(csv_path, index=False)
    print(f"\nCombined results saved to: {csv_path}")

    plot_comparison(
        baseline_all_results,
        rulefit_support_all_results,
        args.task_id,
        output_path=args.output,
        show=(not args.no_show),
        file_suffix=order_tag,
        task_type=task_type,
    )


if __name__ == '__main__':
    main()
