#!/usr/bin/env python
"""
Compare performance between different index SPEX-EBM models (bii, fbii, sii, etc.).

Usage:
    python compare_index_performance.py <task_id> <N_interactions> [--output OUTPUT_PATH]
    
Example:
    python compare_index_performance.py 363615 6 --output comparison_plot.png
"""

import argparse
import os
import re
import time
import glob
import openml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, log_loss
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingRegressor
from typing import Dict, List, Tuple


def _coerce_target_to_numeric(y_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Convert target to numeric; fall back to categorical codes if needed.

    Returns:
        y_numeric: float64 array
        valid_mask: bool array where y_numeric is finite
    """
    y_numeric = pd.to_numeric(y_series, errors="coerce")
    if y_numeric.isna().any():
        y_cat = pd.Categorical(y_series)
        y_numeric = pd.Series(y_cat.codes, index=y_series.index, name=y_series.name)
        print("Target is non-numeric; converted to categorical codes.")
    y_numeric = y_numeric.replace([np.inf, -np.inf], np.nan)
    valid_mask = ~y_numeric.isna()
    return y_numeric.to_numpy(dtype=np.float64), valid_mask.to_numpy()


def _infer_task_type_from_y(y_series: pd.Series) -> str:
    y_series = y_series if isinstance(y_series, pd.Series) else pd.Series(y_series)
    y_unique = y_series.nunique(dropna=True)
    y_total = len(y_series)
    y_dtype_str = str(y_series.dtype)
    if (
        y_dtype_str == "object"
        or "string" in y_dtype_str.lower()
        or "category" in y_dtype_str.lower()
        or (y_unique < 20 and y_unique / max(y_total, 1) < 0.1)
    ):
        return "classification"
    return "regression"


def find_available_indices(task_id, interaction_dir='./interaction_12_14_2025'):
    """Find all available index types for a given task_id."""
    pattern = os.path.join(interaction_dir, f'interactions_summary_{task_id}_*.pkl')
    files = glob.glob(pattern)
    
    indices = []
    for file in files:
        # Extract index from filename: interactions_summary_{task_id}_{index}.pkl
        basename = os.path.basename(file)
        match = re.match(rf'interactions_summary_{task_id}_(.+)\.pkl', basename)
        if match:
            index = match.group(1)
            indices.append(index)
    
    return sorted(indices)


def load_interaction_data(task_id, index, interaction_dir='./interaction_12_14_2025'):
    """Load interaction data from pickle file for a specific index."""
    pickle_path = os.path.join(interaction_dir, f'interactions_summary_{task_id}_{index}.pkl')
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_openml_data(task_id):
    """Load data from OpenML."""
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    print(f"Task ID: {task_id}")
    print(f"Dataset ID: {dataset.id}, Dataset Name: {dataset.name}")
    
    # Get target column name
    target_name = getattr(task, 'target_name', None) or dataset.default_target_attribute
    
    # Get data (as pandas)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=target_name, 
        dataset_format="dataframe"
    )
    
    # Get train/test split
    train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0)

    # Normalize types: keep X as DataFrame (preserve category dtypes), y as 1D float ndarray
    X = pd.DataFrame(X)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name=target_name)

    task_type = _infer_task_type_from_y(y)
    print(f"Inferred task type: {task_type}")

    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)

    y_train_series = y.iloc[train_indices].reset_index(drop=True)
    y_test_series = y.iloc[test_indices].reset_index(drop=True)

    y_train, train_mask = _coerce_target_to_numeric(y_train_series)
    y_test, test_mask = _coerce_target_to_numeric(y_test_series)

    if not train_mask.all():
        dropped = int((~train_mask).sum())
        print(f"Warning: dropped {dropped} training rows with missing/inf targets.")
        X_train = X_train.loc[train_mask].reset_index(drop=True)
        y_train = y_train[train_mask]

    if not test_mask.all():
        dropped = int((~test_mask).sum())
        print(f"Warning: dropped {dropped} test rows with missing/inf targets.")
        X_test = X_test.loc[test_mask].reset_index(drop=True)
        y_test = y_test[test_mask]

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")

    return X_train, y_train, X_test, y_test, attribute_names, task_type


def get_manual_interactions(interaction_data, N_interactions, max_interaction_order: int = 2):
    """Extract top N interactions from interaction data.

    Args:
        max_interaction_order: 2 -> 2-way only
                              3 -> 2-way + 3-way
                              4 -> 2-way + 3-way + 4-way
    """
    if max_interaction_order not in (2, 3, 4):
        raise ValueError(f"max_interaction_order must be 2, 3, or 4, got: {max_interaction_order}")

    # Filter interactions: at least 2-way, at most max_interaction_order-way
    filtered_counts = {
        k: v
        for k, v in interaction_data["interaction_counts"].items()
        if 2 <= len(k) <= max_interaction_order
    }
    
    # Get top N interactions (sorted by count value)
    sorted_interactions = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
    top_interactions = sorted_interactions[:N_interactions]
    manual_interactions_set = [interaction[0] for interaction in top_interactions]
    
    if max_interaction_order == 2:
        desc = "2-way only"
    else:
        desc = f"2-way to {max_interaction_order}-way"
    print(f"\nSelected {len(manual_interactions_set)} interactions ({desc}):")
    for i, interaction in enumerate(manual_interactions_set, 1):
        count = dict(filtered_counts)[interaction]
        print(f"  {i}. {interaction}: count={count}")
    
    return manual_interactions_set


def train_spex_ebm_models(X_train, y_train, X_test, y_test, manual_interactions_set, index_name, task_type):
    """Train SPEX-EBM models with manual interactions."""
    n_interactions = len(manual_interactions_set)
    
    # Store all models and results
    ebm_models = []
    all_results = []
    
    print("\n" + "="*80)
    print(f"=== Training {index_name.upper()}-SPEX-EBM Models with Different Number of Interactions ===\n")
    
    for num_interactions in range(1, n_interactions + 1):
        # Take first num_interactions interactions
        current_interactions = manual_interactions_set[:num_interactions]
        
        print(f"Model {num_interactions}/{n_interactions}: Using {num_interactions} interaction(s): {current_interactions}")
        
        # Initialize
        if task_type == "classification":
            from interpret.glassbox import ExplainableBoostingClassifier
            ebm_spex = ExplainableBoostingClassifier(
                random_state=42,
                interactions=current_interactions,
                outer_bags=4,
                max_bins=256
            )
        else:
            ebm_spex = ExplainableBoostingRegressor(
                random_state=42,
                interactions=current_interactions,
                outer_bags=4,
                max_bins=256
            )
        
        # Train
        t0 = time.perf_counter()
        ebm_spex.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0
        ebm_models.append(ebm_spex)
        
        # Evaluate
        y_pred = ebm_spex.predict(X_test)
        if task_type == "classification":
            y_proba = ebm_spex.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            ll = log_loss(y_test, y_proba, labels=ebm_spex.classes_)
            all_results.append({
                'Model': f'Model {num_interactions}',
                'Num_Interactions': num_interactions,
                'Interactions': str(current_interactions),
                'Train_Time_s': train_time_s,
                'Accuracy': acc,
                'LogLoss': ll
            })
            print(f"  - Accuracy: {acc:.4f}, LogLoss: {ll:.4f}\n")
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


def print_performance_table(results, title):
    """Print performance comparison table."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    results_df = pd.DataFrame(results)
    
    # Format and print table
    print("\n" + "-"*80)
    has_time = 'Train_Time_s' in results_df.columns
    is_classification = 'Accuracy' in results_df.columns
    if is_classification:
        if has_time:
            print(f"{'Model':<12} {'# Interact':<12} {'Time(s)':<10} {'Accuracy':<12} {'LogLoss':<12}")
        else:
            print(f"{'Model':<12} {'# Interact':<12} {'Accuracy':<12} {'LogLoss':<12}")
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
            if is_classification:
                print(f"{row['Model']:<12} {row['Num_Interactions']:<12} {tt:<10.2f} {row['Accuracy']:<12.4f} {row['LogLoss']:<12.4f}")
            else:
                print(f"{row['Model']:<12} {row['Num_Interactions']:<12} {tt:<10.2f} {row['MSE']:<12.4f} {row['RMSE']:<12.4f} {row['MAE']:<12.4f} {row['R2']:<12.4f}")
        else:
            if is_classification:
                print(f"{row['Model']:<12} {row['Num_Interactions']:<12} {row['Accuracy']:<12.4f} {row['LogLoss']:<12.4f}")
            else:
                print(f"{row['Model']:<12} {row['Num_Interactions']:<12} {row['MSE']:<12.4f} {row['RMSE']:<12.4f} {row['MAE']:<12.4f} {row['R2']:<12.4f}")
    print("-"*80)
    print()
    
    return results_df


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
    all_results_dict: Dict[str, List[Dict]],
    task_id: int,
    output_path=None,
    show=True,
    file_suffix: str | None = None,
):
    """Create and save comparison plots for different index SPEX-EBM models."""
    # Define colors for different indices (use distinct colors)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#2D6A4F', '#6C3483', '#1B4332', '#D90429', '#FB8500']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    
    # Determine metrics type
    sample_index = next(iter(all_results_dict))
    sample_df = pd.DataFrame(all_results_dict[sample_index])
    is_classification = 'Accuracy' in sample_df.columns

    if is_classification:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Performance Comparison: Different Index SPEX-EBM Models (Task {task_id})',
                     fontsize=16, fontweight='bold', y=0.995)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Performance Comparison: Different Index SPEX-EBM Models (Task {task_id})', 
                     fontsize=16, fontweight='bold', y=0.995)
    
    # Get all indices and determine max interactions
    indices = sorted(all_results_dict.keys())
    max_interactions = 0
    for index in indices:
        df = pd.DataFrame(all_results_dict[index])
        max_interactions = max(max_interactions, len(df))
    
    # Plot each index
    for idx_idx, index in enumerate(indices):
        df = pd.DataFrame(all_results_dict[index])
        interactions = np.array(df['Num_Interactions'].values)
        if is_classification:
            acc = np.array(df['Accuracy'].values)
            ll = np.array(df['LogLoss'].values)
        else:
            mse = np.array(df['MSE'].values)
            rmse = np.array(df['RMSE'].values)
            mae = np.array(df['MAE'].values)
            r2 = np.array(df['R2'].values)
        
        color = colors[idx_idx % len(colors)]
        marker = markers[idx_idx % len(markers)]
        label = f'{index.upper()}-SPEX'
        
        if is_classification:
            ax1 = axes[0]
            ax1.plot(interactions, acc, marker=marker, linewidth=2, markersize=7,
                     label=label, color=color)
            ax2 = axes[1]
            ax2.plot(interactions, ll, marker=marker, linewidth=2, markersize=7,
                     label=label, color=color)
        else:
            ax1 = axes[0, 0]
            ax1.plot(interactions, mse, marker=marker, linewidth=2, markersize=7, 
                    label=label, color=color)
            
            ax2 = axes[0, 1]
            ax2.plot(interactions, rmse, marker=marker, linewidth=2, markersize=7, 
                    label=label, color=color)
            
            ax3 = axes[1, 0]
            ax3.plot(interactions, mae, marker=marker, linewidth=2, markersize=7, 
                    label=label, color=color)
            
            ax4 = axes[1, 1]
            ax4.plot(interactions, r2, marker=marker, linewidth=2, markersize=7, 
                    label=label, color=color)
    
    # Configure axes
    axes_flat = axes.flat if hasattr(axes, "flat") else axes
    for ax in axes_flat:
        ax.set_xlabel('Number of Interactions', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(range(1, max_interactions + 1))

    if is_classification:
        axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        axes[0].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('LogLoss', fontsize=11, fontweight='bold')
        axes[1].set_title('LogLoss', fontsize=12, fontweight='bold')
    else:
        axes[0, 0].set_ylabel('MSE', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
        
        axes[0, 1].set_ylabel('RMSE', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
        
        axes[1, 0].set_ylabel('MAE', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
        
        axes[1, 1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Coefficient of Determination (R²)', fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    
    # Save
    if output_path:
        output_dir = _resolve_output_dir(output_path) or '.'
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_{file_suffix}" if file_suffix else ""
        save_path = os.path.join(output_dir, f'index_comparison_{task_id}{suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    # Show/close
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Compare performance between different index SPEX-EBM models',
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
        help='Interaction order to use: 2 (2-way), 3 (2+3-way), 4 (2+3+4-way).',
    )
    parser.add_argument('--interaction-dir', type=str,
                        default=str(os.path.join(os.path.dirname(__file__), '..', 'interaction_search', 'interaction_output')),
                        help='Directory containing interaction pickle files (default: ../interaction_search/interaction_output)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display matplotlib windows (useful when saving plots)')
    parser.add_argument('--indices', type=str, nargs='+', default=None,
                        help='Specific indices to compare (default: all available indices)')
    
    args = parser.parse_args()
    order_tag = f"{args.max_interaction_order}way"
    
    # Find available indices
    if args.indices:
        available_indices = args.indices
        print(f"Using specified indices: {available_indices}")
    else:
        available_indices = find_available_indices(args.task_id, args.interaction_dir)
        print(f"\nFound {len(available_indices)} available indices: {available_indices}")
    
    if not available_indices:
        raise ValueError(f"No interaction files found for task_id {args.task_id} in {args.interaction_dir}")
    
    # Load OpenML data (only once)
    print("\nLoading OpenML data...")
    X_train, y_train, X_test, y_test, attribute_names, task_type = load_openml_data(args.task_id)
    
    # Store results for all indices
    all_results_dict = {}
    all_results_dfs = []
    
    # Process each index
    for index in available_indices:
        try:
            print(f"\n{'='*80}")
            print(f"Processing index: {index.upper()}")
            print(f"{'='*80}")
            
            # Load interaction data
            print(f"Loading interaction data for {index}...")
            interaction_data = load_interaction_data(args.task_id, index, args.interaction_dir)
            
            # Get manual interactions
            manual_interactions_set = get_manual_interactions(
                interaction_data,
                args.N_interactions,
                max_interaction_order=args.max_interaction_order,
            )
            
            # Train SPEX-EBM models
            ebm_models, results = train_spex_ebm_models(
                X_train, y_train, X_test, y_test, manual_interactions_set, index, task_type
            )
            
            # Print performance table
            results_df = print_performance_table(results, f"{index.upper()}-SPEX-EBM Performance Comparison Table")
            
            # Store results
            all_results_dict[index] = results
            results_df['Index'] = index
            results_df['Model_Type'] = f'{index.upper()}-SPEX-EBM'
            all_results_dfs.append(results_df)
            
        except FileNotFoundError as e:
            print(f"Warning: {e}, skipping index {index}")
            continue
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results_dict:
        raise ValueError(f"No valid results obtained for any index. Task ID: {args.task_id}")
    
    # Determine output directory
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
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine and save all results to CSV
    combined_df = pd.concat(all_results_dfs, ignore_index=True)
    csv_path = os.path.join(output_dir, f'index_results_{args.task_id}_{order_tag}.csv')
    combined_df.to_csv(csv_path, index=False)
    print(f"\nCombined results saved to: {csv_path}")
    
    # Create and save comparison plot
    plot_comparison(
        all_results_dict,
        args.task_id,
        output_path=args.output,
        show=(not args.no_show),
        file_suffix=order_tag,
    )


if __name__ == '__main__':
    main()

