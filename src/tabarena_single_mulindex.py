import openml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from collections import Counter

# Import TabPFN and spectralexplain
from tabpfn_extensions import TabPFNRegressor
import torch

# Setup spectralexplain path
# Use environment variable or relative path for cross-platform compatibility
# import sys
# spectral_explain_path = os.environ.get('SPECTRAL_EXPLAIN_PATH', 
                                    #    os.path.join(os.path.dirname(__file__), 'spectral-explain', 'src'))
# sys.path.insert(0, spectral_explain_path)
import src.spectralexplain as spex


def process_task(
    task_id,
    num_samples=2,
    index_types=None,
    output_dir='interaction_result'
):
    """
    Process a single task and save interaction results
    
    Args:
        task_id: OpenML task ID
        num_samples: Number of training samples to process (default: 2)
        index_types: List of interaction index types to compute
        output_dir: Output directory for saved files (default: 'interaction_result')
    
    Returns:
        tuple: (result_data, summary_data) dictionaries
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load task and data
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    print(f"Task ID: {task_id}")
    print(f"Dataset ID: {dataset.id}, Dataset Name: {dataset.name}")

    # Get target column name
    target_name = getattr(task, 'target_name', None) or dataset.default_target_attribute

    # Get data
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=target_name, 
        dataset_format="dataframe"
    )

    # ============================
    # Minimal fix: encode categorical columns in X
    # ============================
    cat_cols = [
        col for col in X.columns
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category")
    ]
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols])

    # Get train/test split
    train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0)

    # Minimal fix: convert AFTER encoding
    X_train = X.iloc[train_indices].to_numpy(dtype=np.float64)
    X_test  = X.iloc[test_indices].to_numpy(dtype=np.float64)
    y_train = y.iloc[train_indices].to_numpy(dtype=np.float64)
    y_test  = y.iloc[test_indices].to_numpy(dtype=np.float64)

    # Ensure y is a 1D array
    y_train = y_train.flatten() if y_train.ndim > 1 else y_train
    y_test  = y_test.flatten() if y_test.ndim > 1 else y_test

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")



    ######################################################## TABPFN ########################################################

    # Check if GPU is available
    print("\n=== GPU Detection ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        device_to_use = "cuda"
    else:
        print("Warning: CUDA is not available, will use CPU")
        print("If you have a GPU, please check if PyTorch is correctly installed with CUDA version")
        device_to_use = "auto"
        # If must run on CPU, set environment variable to allow large datasets
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

    # Train model using training set
    print("\n=== Training TabPFN Model ===")
    print(f"Using device: {device_to_use}")
    print(f"Training set size: {X_train.shape[0]} samples")

    # Create model, set device and ignore_pretraining_limits (if GPU is not available)
    model = TabPFNRegressor(
        device=device_to_use,
        ignore_pretraining_limits=not torch.cuda.is_available()  # Allow CPU run if GPU is not available
    )
    model.fit(X_train, y_train)
    print("TabPFN model training completed!")

    ######################################################## SPEX ########################################################

    print("\nLoaded:", spex.__file__)

    if index_types is None:
        index_types = ["fbii", "fsii", "stii", "bii", "sii", "fourier", "mobius"]

    # Use all samples from training set
    train_set = X_train.copy().astype(np.float64)
    train_labels = y_train.copy().astype(np.float64)
    print(f"\nTotal training set size: {X_train.shape}")
    num_samples_to_use = len(train_set) if (num_samples is None or num_samples <= 0) else min(num_samples, len(train_set))
    num_samples_to_process = min(num_samples_to_use, 1000)  # cap to first 1000 samples
    print(f"Will process {num_samples_to_process} samples")

    # Calculate mean of entire training set as baseline value
    train_mean = X_train.mean(axis=0).astype(np.float64)

    # Calculate interactions for each sample in training set (capped at 1000)
    print(f"\n=== TabPFN Feature Interaction Analysis (Processing {num_samples_to_process} Samples) ===")
    print(f"Index types: {index_types}")

    # Store interactions per index type
    all_interactions = {index_type: [] for index_type in index_types}
    for idx, train_point in enumerate(train_set[:num_samples_to_process]):
        print(f"\nProcessing training sample {idx + 1}/{num_samples_to_process}...")
        # if idx ==3:
        #     break
        def tabular_masking(X):
            # X is a boolean mask array (batch_size x num_features), indicating which features are kept
            # For each sample, if X[i, j] == True, use train_point[j], otherwise use train_mean[j]
            # Need to handle batch input
            if X.ndim == 1:
                # Single sample, convert to batch
                X = X.reshape(1, -1)
            
            # Expand train_point and train_mean to match batch size
            batch_size = X.shape[0]
            train_point_expanded = np.tile(train_point, (batch_size, 1))
            train_mean_expanded = np.tile(train_mean, (batch_size, 1))
            
            # Apply mask
            masked_data = np.where(X, train_point_expanded, train_mean_expanded)
            
            # Ensure float64 type
            masked_data = masked_data.astype(np.float64)
            
            return model.predict(masked_data)
        
        explainer = spex.Explainer(
            value_function=tabular_masking,
            features=range(len(train_point)),
            sample_budget=1000
        )
        
        for index_type in index_types:
            interactions = explainer.interactions(index=index_type)
            all_interactions[index_type].append(interactions)
            
            print(f"Interactions for sample {idx + 1} (index={index_type}):")
            print(interactions)

    print(f"\n=== Completed! Processed {num_samples_to_process} training samples ===")
    print(f"All interaction results saved in all_interactions dict keyed by index type")



    ######################################################## SAVE RESULTS ########################################################
    # Extract top interactions information from interaction objects
    def extract_interaction_info(interaction_obj, top_k=5):
        """Extract key information from interaction object, only keeping top K interactions"""
        # Get top k interactions (excluding baseline value, i.e., empty tuple)
        interactions_dict = interaction_obj.interactions
        # Filter out empty tuple (baseline), sort by absolute value and take top k
        filtered_interactions = {k: float(v) for k, v in interactions_dict.items() if len(k) > 0}
        sorted_interactions = dict(sorted(filtered_interactions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k])
        
        return {
            'index': interaction_obj.index,
            'max_order': int(interaction_obj.max_order),
            'baseline_value': float(interaction_obj.baseline_value),
            'num_features': int(interaction_obj.num_features),
            'sample_budget': int(interaction_obj.sample_budget),
            'top_interactions': sorted_interactions  # Only save top k interactions
        }

    # Extract top interactions information for all samples per index type
    interactions_data = {
        index_type: [
            extract_interaction_info(interaction, top_k=5)
            for interaction in interactions_list
        ]
        for index_type, interactions_list in all_interactions.items()
    }

    # Save results to pickle file per index type - only store top interactions information
    index_tag = "_".join(index_types)
    for index_type in index_types:
        result_data = {
            'task_id': task_id,
            'dataset_id': dataset.id,
            'dataset_name': dataset.name,
            'num_samples_processed': num_samples_to_process,
            'num_features': X_train.shape[1],
            'index_type': index_type,
            'interactions': interactions_data[index_type]  # Only save extracted info, not full objects
        }

        filename = output_path / f'interactions_{task_id}_{index_type}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(result_data, f)

        print(f"\n=== Results Saved ({index_type}) ===")
        print(f"Saved to: {filename}")
        print(f"Task ID: {task_id}")
        print(f"Dataset: {dataset.name}")
        print(f"Number of samples: {num_samples_to_process}")
        print("Each sample contains top 5 interactions")

    # Calculate interaction summary statistics
    print(f"\n=== Calculating Interaction Summary ===")

    # Extract top interactions from all interaction results (top 5 for each sample) per index type
    per_index_summary = {}
    for index_type, interaction_list in interactions_data.items():
        all_top_interactions = []
        for idx, interaction_data in enumerate(interaction_list):
            try:
                top_interactions_dict = interaction_data['top_interactions']
                top_interactions = list(top_interactions_dict.keys())

                # Convert interactions to hashable tuples (ensure feature index ordering is consistent)
                for interaction in top_interactions:
                    if isinstance(interaction, (list, tuple)):
                        interaction_tuple = tuple(sorted(interaction))
                        all_top_interactions.append(interaction_tuple)
                    else:
                        all_top_interactions.append(interaction)
            except Exception as e:
                print(f"Warning: Error processing sample {idx + 1} for index {index_type}: {e}")
                continue

        interaction_counts = Counter(all_top_interactions)
        sorted_interactions = sorted(interaction_counts.items(), key=lambda x: x[1], reverse=True)
        num_samples_processed = len(interaction_list)

        per_index_summary[index_type] = {
            'num_samples': num_samples_processed,
            'num_unique_interactions': len(interaction_counts),
            'interaction_counts': dict(sorted_interactions),
            'interaction_frequencies': {
                interaction: (count / num_samples_processed * 100) if num_samples_processed > 0 else 0
                for interaction, count in sorted_interactions
            }
        }

    # Save summary to pickle file per index type
    for index_type, summary in per_index_summary.items():
        summary_data = {
            'task_id': task_id,
            'dataset_id': dataset.id,
            'dataset_name': dataset.name,
            'num_samples': summary['num_samples'],
            'index_type': index_type,
            'num_unique_interactions': summary['num_unique_interactions'],
            'interaction_counts': summary['interaction_counts'],
            'interaction_frequencies': summary['interaction_frequencies']
        }

        summary_filename = output_path / f'interactions_summary_{task_id}_{index_type}.pkl'
        with open(summary_filename, 'wb') as f:
            pickle.dump(summary_data, f)

        print(f"\n=== Summary Saved ({index_type}) ===")
        print(f"Saved to: {summary_filename}")
        print(f"  Samples: {summary['num_samples']}")
        print(f"  Unique interactions: {summary['num_unique_interactions']}")
        print(f"  Top 10 most common interactions:")
        sorted_interactions = list(summary['interaction_counts'].items())
        for interaction, count in sorted_interactions[:10]:
            percentage = summary['interaction_frequencies'].get(interaction, 0)
            print(f"    {interaction}: {count} occurrences ({percentage:.1f}%)")
    
    return result_data, summary_data


# Main execution
if __name__ == "__main__":
    # Directly specify task ID
    task_id = 363698  # QSAR_fish_toxicity
    process_task(task_id, num_samples=2)