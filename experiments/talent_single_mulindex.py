import numpy as np
import json
import torch
import os
from pathlib import Path
from collections import Counter
import pickle

from tabpfn_extensions import TabPFNRegressor
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Setup spectralexplain path
import sys
spectral_explain_path = os.environ.get('SPECTRAL_EXPLAIN_PATH', 
                                       os.path.join(os.path.dirname(__file__), 'spectral-explain', 'src'))
sys.path.insert(0, spectral_explain_path)
import spectralexplain as spex


def load_talent_data(dataset_name, data_dir="Talent_data"):
    """
    Load dataset from Talent_data folder.
    
    Args:
        dataset_name: Name of the dataset folder
        data_dir: Root directory containing dataset folders
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, info_dict)
    """
    data_path = Path(data_dir) / dataset_name
    
    # Read info.json
    with open(data_path / "info.json", "r") as f:
        info = json.load(f)
    
    print(f"Dataset: {dataset_name}")
    print(f"Task type: {info['task_type']}")
    print(f"Number of numeric features: {info['n_num_features']}")
    print(f"Number of categorical features: {info['n_cat_features']}")
    
    # Load training data
    N_train = np.load(data_path / "N_train.npy", allow_pickle=True) if (data_path / "N_train.npy").exists() else None
    C_train = np.load(data_path / "C_train.npy", allow_pickle=True) if (data_path / "C_train.npy").exists() else None
    y_train = np.load(data_path / "y_train.npy", allow_pickle=True)
    
    # Load test data
    N_test = np.load(data_path / "N_test.npy", allow_pickle=True) if (data_path / "N_test.npy").exists() else None
    C_test = np.load(data_path / "C_test.npy", allow_pickle=True) if (data_path / "C_test.npy").exists() else None
    y_test = np.load(data_path / "y_test.npy", allow_pickle=True)
    
    # Encode labels if they contain strings
    if y_train.dtype == object or np.issubdtype(y_train.dtype, np.str_):
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train).astype(np.float64)
        y_test = label_encoder.transform(y_test).astype(np.float64)
        print(f"\nEncoded labels using LabelEncoder. Classes: {label_encoder.classes_}")
    else:
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        y_test = np.asarray(y_test, dtype=np.float64).ravel()
    
    print(f"\ny_train dtype: {y_train.dtype}, shape: {y_train.shape}, unique values: {np.unique(y_train)[:10]}")
    print(f"y_test dtype: {y_test.dtype}, shape: {y_test.shape}, unique values: {np.unique(y_test)[:10]}")
    
    # Encode categorical features and combine with numeric features
    if N_train is not None and C_train is not None:
        assert N_test is not None and C_test is not None, "Test data must exist if train data exists"
        # Encode categorical features if they contain strings
        if C_train.dtype == object or np.issubdtype(C_train.dtype, np.str_):
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            C_train_encoded = encoder.fit_transform(C_train).astype(np.float64)
            C_test_encoded = encoder.transform(C_test).astype(np.float64)
            print(f"\nEncoded categorical features (train shape: {C_train_encoded.shape})")
        else:
            C_train_encoded = C_train.astype(np.float64)
            C_test_encoded = C_test.astype(np.float64)
        
        # Convert numeric features to float64
        N_train_float = N_train.astype(np.float64) if N_train.dtype != np.float64 else N_train
        N_test_float = N_test.astype(np.float64) if N_test.dtype != np.float64 else N_test
        
        X_train = np.hstack([N_train_float, C_train_encoded]).astype(np.float64)
        X_test = np.hstack([N_test_float, C_test_encoded]).astype(np.float64)
    elif N_train is not None:
        assert N_test is not None, "Test data must exist if train data exists"
        X_train = N_train.astype(np.float64)
        X_test = N_test.astype(np.float64)
    elif C_train is not None:
        assert C_test is not None, "Test data must exist if train data exists"
        # Only categorical features - encode if needed
        if C_train.dtype == object or np.issubdtype(C_train.dtype, np.str_):
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train = encoder.fit_transform(C_train).astype(np.float64)
            X_test = encoder.transform(C_test).astype(np.float64)
            print(f"\nEncoded categorical features (train shape: {X_train.shape})")
        else:
            X_train = C_train.astype(np.float64)
            X_test = C_test.astype(np.float64)
    else:
        raise ValueError("No features found!")
    
    print(f"\nTraining set shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"Test set shape: {X_test.shape}, dtype: {X_test.dtype}")
    
    return X_train, y_train, X_test, y_test, info


def process_talent_dataset(
    dataset_name,
    num_samples=None,
    index_types=None,
    output_dir='talent_interaction_result',
    data_dir="Talent_data"
):
    """
    Process a single Talent dataset and save interaction results.
    
    Args:
        dataset_name: Name of the dataset folder in Talent_data
        num_samples: Number of training samples to process (default: None, process all)
        index_types: List of interaction index types to compute (default: ["fbii", "fsii", "stii", "bii", "sii", "fourier", "mobius"])
        output_dir: Output directory for saved files (default: 'talent_interaction_result')
        data_dir: Root directory containing dataset folders (default: 'Talent_data')
    
    Returns:
        tuple: (result_data, summary_data) dictionaries
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, y_train, X_test, y_test, info = load_talent_data(dataset_name, data_dir)
    
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
        ignore_pretraining_limits=not torch.cuda.is_available()
    )
    model.fit(X_train, y_train)
    print("TabPFN model training completed!")
    
    ######################################################## SPEX ########################################################
    
    print("\nLoaded:", spex.__file__)
    
    if index_types is None:
        index_types = ["fbii", "fsii", "stii", "bii", "sii", "fourier", "mobius"]
    
    # Use samples from training set
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
    for index_type in index_types:
        result_data = {
            'dataset_name': dataset_name,
            'task_type': info['task_type'],
            'num_numeric_features': info['n_num_features'],
            'num_categorical_features': info['n_cat_features'],
            'num_samples_processed': num_samples_to_process,
            'num_features': X_train.shape[1],
            'index_type': index_type,
            'interactions': interactions_data[index_type]  # Only save extracted info, not full objects
        }
        
        filename = output_path / f'interactions_{dataset_name}_{index_type}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(result_data, f)
        
        print(f"\n=== Results Saved ({index_type}) ===")
        print(f"Saved to: {filename}")
        print(f"Dataset: {dataset_name}")
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
            'dataset_name': dataset_name,
            'task_type': info['task_type'],
            'num_numeric_features': info['n_num_features'],
            'num_categorical_features': info['n_cat_features'],
            'num_samples': summary['num_samples'],
            'index_type': index_type,
            'num_unique_interactions': summary['num_unique_interactions'],
            'interaction_counts': summary['interaction_counts'],
            'interaction_frequencies': summary['interaction_frequencies']
        }
        
        summary_filename = output_path / f'interactions_summary_{dataset_name}_{index_type}.pkl'
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
    # Available Talent datasets
    datasets = [
        "abalone",
        "analcatdata_supreme",
        "chscase_foot",
        "combined_cycle_power_plant",
        "Data_Science_Salaries",
        "debutanizer",
        "delta_ailerons",
        "delta_elevators",
        "Goodreads-Computer-Books",
        "qsar_aquatic_toxicity",
    ]
    
    # Specify dataset name
    dataset_name = "qsar_aquatic_toxicity"  # Change to process different dataset
    
    # Process with multiple index types
    process_talent_dataset(
        dataset_name=dataset_name,
        num_samples=1,  # Number of samples to process (can set to None to process all, capped at 1000)
        index_types=["fbii", "fsii", "stii", "bii", "sii", "fourier", "mobius"]  # Or use None for all default types
    )

