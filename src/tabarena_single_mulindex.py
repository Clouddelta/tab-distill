from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from collections import Counter
import src.config
import os
import src.spectralexplain as spex
import joblib


def _openml_get_task(task_id):
    fname = os.path.join(src.config.cache_dir_openml, str(task_id) + '.pkl')
    if os.path.exists(fname):
        print(f'found cached task {task_id}')
        return joblib.load(fname)
    import openml
    print('caching...')
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    joblib.dump((task, dataset), fname)
    return task, dataset

def get_data(task_id):
    # Load task and data
    # task = openml.tasks.get_task(task_id)
    # dataset = task.get_dataset()
    # openml.config.cache_directory = src.config.cache_dir_openml
    # print(f"CACHE_DIR: {openml.config.get_cache_directory()}")
    task, dataset = _openml_get_task(task_id)
    print(f"Task ID: {task_id}")
    print(f"Dataset ID: {dataset.id}, Dataset Name: {dataset.name}")

    # Determine task type (classification or regression)
    # OpenML task_type_id: 1 = Supervised Classification, 2 = Supervised Regression
    task_type_id = getattr(task, 'task_type_id', None)
    if task_type_id == 1:
        task_type = "classification"
    elif task_type_id == 2:
        task_type = "regression"
    else:
        # Fallback: try to infer from evaluation measure
        eval_measure = getattr(task, 'evaluation_measure', None)
        if eval_measure is not None:
            eval_measure = str(eval_measure).lower()
            if 'auc' in eval_measure or 'accuracy' in eval_measure or 'log_loss' in eval_measure:
                task_type = "classification"
            elif 'rmse' in eval_measure or 'mse' in eval_measure or 'r2' in eval_measure or 'mae' in eval_measure:
                task_type = "regression"
            else:
                task_type = None  # Will be determined from y data
        else:
            task_type = None  # Will be determined from y data

    print(f"Task type (from OpenML): {task_type} (task_type_id: {task_type_id})")

    # Get target column name
    target_name = getattr(task, 'target_name', None) or dataset.default_target_attribute

    # Get data
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=target_name, 
        dataset_format="dataframe"
    )

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Encode categorical columns in X (before train/test split)
    cat_cols = [
        col for col in X.columns
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category")
    ]
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols])
        print(f"\nEncoded {len(cat_cols)} categorical features")

    # Determine final task type based on y data if not determined from OpenML
    if task_type is None:
        # Check if y contains strings or if unique values suggest classification
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        y_unique = y_series.nunique()
        y_total = len(y_series)
        # If unique values are few relative to total (e.g., < 10% of total or < 20 unique values)
        # and y contains non-numeric data, likely classification
        y_dtype_str = str(y_series.dtype)
        if y_dtype_str == 'object' or 'string' in y_dtype_str.lower() or (y_unique < 20 and y_unique / y_total < 0.1):
            task_type = "classification"
        else:
            task_type = "regression"
        print(f"Task type (inferred from data): {task_type} (unique values: {y_unique}/{y_total})")

    # Get train/test split
    train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0)

    # Convert X to numpy arrays AFTER encoding (always float64 for features)
    X_train = X.iloc[train_indices].to_numpy(dtype=np.float64)
    X_test = X.iloc[test_indices].to_numpy(dtype=np.float64)

    # Convert y based on task type (don't convert to float64 yet for classification)
    y_train_raw = y.iloc[train_indices]
    y_test_raw = y.iloc[test_indices]

    # Ensure y_train and y_test are properly formatted based on task type
    if task_type == "classification":
        # For classification, encode labels to integers
        y_train_series = pd.Series(y_train_raw) if not isinstance(y_train_raw, pd.Series) else y_train_raw
        y_test_series = pd.Series(y_test_raw) if not isinstance(y_test_raw, pd.Series) else y_test_raw
        
        # Check if we need to encode (if contains non-numeric values)
        # Try to convert to numeric first to check if all values are numeric
        try:
            # Try numeric conversion - if successful and no NaN, use numeric labels
            y_train_test_series = pd.to_numeric(y_train_series, errors='coerce')
            # Check if conversion produced any NaN values (indicates non-numeric input)
            has_nan = pd.isna(y_train_test_series).any() if isinstance(y_train_test_series, pd.Series) else False
            
            if has_nan:
                # Contains non-numeric values, use LabelEncoder
                raise ValueError("Contains non-numeric values")
            
            # All numeric, proceed with numeric conversion
            y_train = np.asarray(pd.to_numeric(y_train_series, errors='coerce'), dtype=np.int64)
            y_test = np.asarray(pd.to_numeric(y_test_series, errors='coerce'), dtype=np.int64)
            # Ensure labels start from 0
            unique_labels = np.unique(y_train)
            if len(unique_labels) > 0 and unique_labels.min() != 0:
                label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                y_train = np.array([label_map.get(label, -1) for label in y_train], dtype=np.int64)
                y_test = np.array([label_map.get(label, -1) for label in y_test], dtype=np.int64)
                print(f"\nMapped labels to start from 0. Original labels: {unique_labels}")
        except (ValueError, TypeError, AttributeError):
            # Contains non-numeric values (strings, etc.), use LabelEncoder
            label_encoder = LabelEncoder()
            # Convert to string array first to handle categorical types
            y_train_str = y_train_series.astype(str).values
            y_test_str = y_test_series.astype(str).values
            y_train = label_encoder.fit_transform(y_train_str)
            y_test = label_encoder.transform(y_test_str)
            print(f"\nEncoded labels using LabelEncoder. Classes: {label_encoder.classes_}")
    else:
        # For regression, ensure numeric float64
        y_train_series = pd.Series(y_train_raw) if not isinstance(y_train_raw, pd.Series) else y_train_raw
        y_test_series = pd.Series(y_test_raw) if not isinstance(y_test_raw, pd.Series) else y_test_raw
        
        y_train_dtype_str = str(y_train_series.dtype)
        if 'object' in y_train_dtype_str or 'string' in y_train_dtype_str.lower():
            # Try to convert to numeric
            y_train = np.asarray(pd.to_numeric(y_train_series, errors='coerce'), dtype=np.float64)
            y_test = np.asarray(pd.to_numeric(y_test_series, errors='coerce'), dtype=np.float64)
            print("\nConverted labels to numeric for regression")
        else:
            # Already numeric, just convert to float64 and ensure 1D
            y_train = np.asarray(y_train_series.values, dtype=np.float64).ravel()
            y_test = np.asarray(y_test_series.values, dtype=np.float64).ravel()

    # Ensure y is a 1D array
    y_train = y_train.flatten() if y_train.ndim > 1 else y_train
    y_test = y_test.flatten() if y_test.ndim > 1 else y_test

    print(f"\nTraining set shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"Test set shape: {X_test.shape}, dtype: {X_test.dtype}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"\ny_train dtype: {y_train.dtype}, shape: {y_train.shape}, unique values: {np.unique(y_train)[:10]}")
    print(f"y_test dtype: {y_test.dtype}, shape: {y_test.shape}, unique values: {np.unique(y_test)[:10]}")
    return dataset, task_type, X_train, X_test, y_train, y_test

def get_fitted_model(task_type, X_train, y_train, model_type='tabpfn'):
    ######################################################## TABPFN ########################################################
    # import torch
    # Check if GPU is available
    # print("\n=== GPU Detection ===")
    # print(f"CUDA available: {torch.cuda.is_available()}")
    # if torch.cuda.is_available():
    #     print(f"GPU device: {torch.cuda.get_device_name(0)}")
    #     print(f"Number of GPUs: {torch.cuda.device_count()}")
    #     print(f"Current GPU: {torch.cuda.current_device()}")
    #     device_to_use = "cuda"
    # else:
    #     print("Warning: CUDA is not available, will use CPU")
    #     print("If you have a GPU, please check if PyTorch is correctly installed with CUDA version")
    #     device_to_use = "auto"
    #     # If must run on CPU, set environment variable to allow large datasets
    #     os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    # device_to_use = 'auto'
    device_to_use = 'cpu'

    # Train model using training set - select appropriate model based on task type
    print(f"\n=== Training TabPFN Model ({task_type}) ===")
    print(f"Using device: {device_to_use}")
    print(f"Training set size: {X_train.shape[0]} samples")

    # Create model, set device and ignore_pretraining_limits (if GPU is not available)
    # os.environ['TABPFN_MODEL_CACHE_DIR'] = src.config.cache_dir_tabpfn
    if model_type == 'tabpfn':
        if task_type == "classification":
            from tabpfn.classifier import TabPFNClassifier
            model = TabPFNClassifier(
                device=device_to_use,
                ignore_pretraining_limits=True,
                # ignore_pretraining_limits=not torch.cuda.is_available(),  # Allow CPU run if GPU is not available
                # model_path=os.path.join(src.config.cache_dir_tabpfn, 'tabpfn-v2.5-classifier-v2.5_default.ckpt'),
            )
            n_classes = len(np.unique(y_train))
            print(f"Number of classes: {n_classes}")
        else:
            from tabpfn.regressor import TabPFNRegressor
            model = TabPFNRegressor(
                device=device_to_use,
                ignore_pretraining_limits=True,
                # ignore_pretraining_limits=not torch.cuda.is_available(),  # Allow CPU run if GPU is not available
                # model_path=os.path.join(src.config.cache_dir_tabpfn, 'tabpfn-v2.5-regressor-v2.5_default.ckpt'),
            )
    
    elif model_type == 'rulefit':
        from imodels import RuleFitClassifier, RuleFitRegressor
        if task_type == "classification":
            model = RuleFitClassifier()
            n_classes = len(np.unique(y_train))
            print(f"Number of classes: {n_classes}")
        else:
            model = RuleFitRegressor()

    # fit ridge model instead
    elif model_type == 'ridge':
        if task_type == "classification":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
            n_classes = len(np.unique(y_train))
            print(f"Number of classes: {n_classes}")
        else:
            from sklearn.linear_model import Ridge
            model = Ridge()
    elif model_type == 'tabicl':
        if task_type == "classification":
            from tabicl import TabICLClassifier
            # from tabpfn.classifier import TabPFNClassifier
            model = TabICLClassifier(
                device=device_to_use,
                # ignore_pretraining_limits=True,
                # ignore_pretraining_limits=not torch.cuda.is_available(),  # Allow CPU run if GPU is not available
                # model_path=os.path.join(src.config.cache_dir_tabpfn, 'tabpfn-v2.5-classifier-v2.5_default.ckpt'),
            )
            n_classes = len(np.unique(y_train))
            print(f"Number of classes: {n_classes}")
        else:
            raise NotImplementedError("Regression task not implemented with TabICL yet")
            # from tabpfn.regressor import TabPFNRegressor
            # model = TabPFNRegressor(
            #     device=device_to_use,
            #     ignore_pretraining_limits=True,
            #     # ignore_pretraining_limits=not torch.cuda.is_available(),  # Allow CPU run if GPU is not available
            #     # model_path=os.path.join(src.config.cache_dir_tabpfn, 'tabpfn-v2.5-regressor-v2.5_default.ckpt'),
            # )
    
    model.fit(X_train, y_train)
    print(f"{task_type} model training completed!")
    return model

def eval_fitted_model(model, task_type, X_train, y_train, X_test, y_test):
    # Evaluate model performance
    print(f"\n=== Model Evaluation ===")
    if task_type == "classification":
        y_train_pred = model.predict(X_train[:100])  # Sample for speed
        y_test_pred = model.predict(X_test[:100])
        train_acc = accuracy_score(y_train[:100], y_train_pred)
        test_acc = accuracy_score(y_test[:100], y_test_pred)
        print(f"Training Accuracy (100 samples): {train_acc:.4f}")
        print(f"Test Accuracy (100 samples): {test_acc:.4f}")
    else:
        y_train_pred = model.predict(X_train[:100])  # Sample for speed
        y_test_pred = model.predict(X_test[:100])
        train_mse = mean_squared_error(y_train[:100], y_train_pred)
        test_mse = mean_squared_error(y_test[:100], y_test_pred)
        train_r2 = r2_score(y_train[:100], y_train_pred)
        test_r2 = r2_score(y_test[:100], y_test_pred)
        print(f"Training MSE (100 samples): {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"Test MSE (100 samples): {test_mse:.4f}, R²: {test_r2:.4f}")

def run_spex(model, X_train, num_samples, task_type, index_types):
    print("\nLoaded:", spex.__file__)

    if index_types is None:
        index_types = ["fbii", "fsii", "stii", "bii", "sii", "fourier", "mobius"]

    # Use all samples from training set
    train_set = X_train.copy().astype(np.float64)
    # train_labels = y_train.copy().astype(np.float64)
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
    print(f"Task type: {task_type}")
    
    for idx, train_point in enumerate(train_set[:num_samples_to_process]):
        print(f"\nProcessing training sample {idx + 1}/{num_samples_to_process}...")
        # if idx ==3:
        #     break
        # set up masking function which gets called in explainer
        def _tabular_masking(X):
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
            
            # For classification, use predict_proba to get continuous probability values
            # For regression, use predict to get continuous prediction values
            if task_type == "classification":
                # Get probability predictions (continuous values)
                # TabPFNClassifier should have predict_proba method (sklearn-compatible)
                try:
                    proba = model.predict_proba(masked_data)
                except AttributeError:
                    # Fallback: if predict_proba doesn't exist, try predict and convert to probability-like values
                    # This should not happen with TabPFNClassifier, but just in case
                    predictions = model.predict(masked_data)
                    # Convert predictions to probabilities (this is not ideal but works as fallback)
                    n_classes = len(np.unique(y_train))
                    proba = np.zeros((len(predictions), n_classes))
                    for i, pred in enumerate(predictions):
                        proba[i, int(pred)] = 1.0
                
                # For binary classification, return log-odds of positive class (better for interaction analysis)
                # For multiclass, return probability of most likely class (simpler and more stable)
                if proba.shape[1] == 2:
                    # Binary classification: return log-odds (logit) of positive class
                    # log(p / (1-p)) with numerical stability
                    p = proba[:, 1]  # probability of positive class
                    eps = 1e-10  # small epsilon to avoid log(0) or log(1)
                    p = np.clip(p, eps, 1 - eps)  # clip to avoid numerical issues
                    log_odds = np.log(p / (1 - p))
                    
                    # Debug: check if values are too similar (causes empty Fourier transform)
                    if len(log_odds) > 1:
                        value_range = np.max(log_odds) - np.min(log_odds)
                        if value_range < 1e-6:
                            # Values are too similar, use probability directly instead
                            if idx == 0:  # Only print warning once
                                print(f"Warning: Log-odds values too similar (range: {value_range:.2e}), using probability directly")
                            return p  # Return probability instead of log-odds
                    
                    return log_odds
                else:
                    # Multiclass: return probability of most confident class directly
                    # Probability values are already continuous (0-1), which is what spectral explain needs
                    # This avoids numerical issues with log-odds conversion for multiclass
                    p_max = proba.max(axis=1)  # probability of most likely class
                    # Ensure values are in valid range and return directly
                    p_max = np.clip(p_max, 1e-10, 1.0 - 1e-10)
                    
                    # Debug: check if values are too similar
                    if len(p_max) > 1:
                        value_range = np.max(p_max) - np.min(p_max)
                        if value_range < 1e-6 and idx == 0:
                            print(f"Warning: Probability values too similar (range: {value_range:.2e})")
                    
                    return p_max
            else:
                # Regression: return continuous predictions
                predictions = model.predict(masked_data)
                
                # Debug: check if values are too similar
                if len(predictions) > 1:
                    value_range = np.max(predictions) - np.min(predictions)
                    if value_range < 1e-6 and idx == 0:
                        print(f"Warning: Prediction values too similar (range: {value_range:.2e})")
                
                return predictions
        
        explainer = spex.Explainer(
            value_function=_tabular_masking,
            features=range(len(train_point)),
            sample_budget=1000
        )
        
        for index_type in index_types:
            try:
                interactions = explainer.interactions(index=index_type)
                all_interactions[index_type].append(interactions)
                
                print(f"Interactions for sample {idx + 1} (index={index_type}):")
                print(interactions)
            except AssertionError as e:
                print(f"Error processing sample {idx + 1} (index={index_type}): {e}")
                print("This might indicate that the model predictions are not continuous enough.")
                # Skip this sample for this index type but continue with others
                continue

    print(f"\n=== Completed! Processed {num_samples_to_process} training samples ===")
    print(f"All interaction results saved in all_interactions dict keyed by index type")
    return all_interactions, num_samples_to_process

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

def save_results(task_id, dataset, task_type, X_train, all_interactions, num_samples_to_process, output_path, index_types):

    # Extract top interactions information for all samples per index type
    interactions_data = {
        index_type: [
            extract_interaction_info(interaction, top_k=5)
            for interaction in interactions_list
        ]
        for index_type, interactions_list in all_interactions.items()
    }

    # Save results to pickle file per index type - only store top interactions information
    # index_tag = "_".join(index_types)
    # Initialize result_data and summary_data for return (will store last index_type's data)
    result_data = None
    summary_data = None
    
    for index_type in index_types:
        result_data = {
            'task_id': task_id,
            'dataset_id': dataset.id,
            'dataset_name': dataset.name,
            'task_type': task_type,  # Add task type to result
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
        print(f"Task Type: {task_type}")
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
            'task_type': task_type,  # Add task type to summary
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
        print(f"  Task Type: {task_type}")
        print(f"  Samples: {summary['num_samples']}")
        print(f"  Unique interactions: {summary['num_unique_interactions']}")
        print(f"  Top 10 most common interactions:")
        sorted_interactions = list(summary['interaction_counts'].items())
        for interaction, count in sorted_interactions[:10]:
            percentage = summary['interaction_frequencies'].get(interaction, 0)
            print(f"    {interaction}: {count} occurrences ({percentage:.1f}%)")
    
    # Return the last processed index_type's data (for backward compatibility)
    # If no interactions were computed, return None
    if result_data is None or summary_data is None:
        print("Warning: No interactions were computed successfully")
        result_data = {
            'task_id': task_id,
            'dataset_id': dataset.id,
            'dataset_name': dataset.name,
            'task_type': task_type,
            'num_samples_processed': 0,
            'num_features': X_train.shape[1],
            'index_type': 'none',
            'interactions': []
        }
        summary_data = {
            'task_id': task_id,
            'dataset_id': dataset.id,
            'dataset_name': dataset.name,
            'task_type': task_type,
            'num_samples': 0,
            'index_type': 'none',
            'num_unique_interactions': 0,
            'interaction_counts': {},
            'interaction_frequencies': {}
        }
    return result_data, summary_data

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
    # import openml
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if already done
    all_completed = True
    for index_type in index_types:
        summary_filename = output_path / f'interactions_summary_{task_id}_{index_type}.pkl'
        if not summary_filename.exists():
            all_completed = False
    if all_completed:
        print('Already done!')
        return None, None

            
    dataset, task_type, X_train, X_test, y_train, y_test = get_data(task_id)
    model = get_fitted_model(task_type, X_train, y_train)
    eval_fitted_model(model, task_type, X_train, y_train, X_test, y_test)
    all_interactions, num_samples_to_process = run_spex(
        model=model,
        X_train=X_train,
        num_samples=num_samples,
        task_type=task_type,
        index_types=index_types
    )
    result_data, summary_data = save_results(task_id, dataset, task_type, X_train, all_interactions, num_samples_to_process, output_path, index_types)
    return result_data, summary_data



# Main execution
if __name__ == "__main__":
    # Directly specify task ID
    # task_id = 363698  # QSAR_fish_toxicity
    # process_task(task_id, num_samples=2)

    # cache a bunch of task datasets
    for task_id in [
        363621,  # blood-transfusion-service-center: binary classification, blood donation return prediction
        363629,  # diabetes: binary classification, diabetes onset prediction
        363698,  # QSAR_fish_toxicity: regression, chemical toxicity prediction
        363685,  # maternal_health_risk: multiclass classification, maternal health risk levels
        363625,  # concrete_compressive_strength: regression, concrete strength prediction
        363671,  # Fitness_Club: binary classification, customer churn / subscription behavior
        363612,  # airfoil_self_noise: regression, airfoil noise level prediction
        363615,  # Another-Dataset-on-used-Fiat-500: regression, used car price prediction
        363674,  # hazelnut-spread-contaminant-detection: binary classification, food contamination detection
        363700,  # seismic-bumps: binary classification, seismic event bump prediction
    ]:
        print(task_id)
        _openml_get_task(task_id)