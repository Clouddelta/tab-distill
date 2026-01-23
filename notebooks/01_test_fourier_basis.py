"""
Experiment showing TabPFN learns functions sparse in the Fourier basis.

This experiment is based on the concept from "SPEX: Scaling Feature Interaction 
Explanations for LLMs" (arxiv.org/abs/2502.13870), which discusses Fourier 
sparsity in real-world data and models.

A function f: {0,1}^n -> R is k-sparse in the Fourier basis if it can be 
represented as a linear combination of at most k Fourier basis functions.

Results are averaged over multiple random seeds and saved to pickle files.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
import pickle
from pathlib import Path
from collections import defaultdict

# Try to import TabPFN, provide alternative if not available
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    print("TabPFN not installed. Install with: pip install tabpfn")
    print("Running experiment with sklearn models only.")
    TABPFN_AVAILABLE = False

# Try to import EBM from interpret
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    EBM_AVAILABLE = True
except ImportError:
    print("interpret not installed. Install with: pip install interpret")
    print("Running experiment without EBM baseline.")
    EBM_AVAILABLE = False

# Number of random seeds to average over
N_SEEDS = 30


def walsh_hadamard_basis(x, indices):
    """
    Compute Walsh-Hadamard (Fourier) basis functions for binary inputs.
    
    For x in {0,1}^n and S ⊆ {1,...,n}, the basis function is:
    χ_S(x) = (-1)^{sum(x_i for i in S)}
    
    Args:
        x: array of shape (n_samples, n_features) with values in {0, 1}
        indices: list of tuples, each representing a subset S
    
    Returns:
        array of shape (n_samples, len(indices))
    """
    n_samples, n_features = x.shape
    result = np.ones((n_samples, len(indices)))
    
    for j, subset in enumerate(indices):
        if len(subset) > 0:
            subset_sum = np.sum(x[:, list(subset)], axis=1)
            result[:, j] = (-1) ** subset_sum
    
    return result


def generate_fourier_sparse_function(n_features, sparsity, seed=42):
    """
    Generate a random function that is k-sparse in the Fourier basis.
    
    Args:
        n_features: number of binary input features
        sparsity: number of non-zero Fourier coefficients (k)
        seed: random seed
    
    Returns:
        tuple of (fourier_indices, fourier_coeffs)
    """
    np.random.seed(seed)
    
    max_order = min(3, n_features)
    
    fourier_indices = []
    fourier_indices.append(tuple())
    
    while len(fourier_indices) < sparsity:
        order = np.random.randint(1, max_order + 1)
        subset = tuple(sorted(np.random.choice(n_features, size=order, replace=False)))
        if subset not in fourier_indices:
            fourier_indices.append(subset)
    
    fourier_coeffs = np.random.randn(sparsity) * 2  # Coefficients ~ N(0, 4) since std=2
    
    return fourier_indices, fourier_coeffs


def evaluate_fourier_function(x, fourier_indices, fourier_coeffs):
    """Evaluate a Fourier-sparse function on input x."""
    basis = walsh_hadamard_basis(x, fourier_indices)
    return basis @ fourier_coeffs


def generate_dataset(n_samples, n_features, sparsity, noise_std=0.1, seed=42):
    """
    Generate a dataset with a Fourier-sparse target function.
    
    Args:
        n_samples: number of samples
        n_features: number of binary features
        sparsity: Fourier sparsity (k)
        noise_std: standard deviation of additive noise
        seed: random seed
    
    Returns:
        X, y, fourier_indices, fourier_coeffs
    """
    np.random.seed(seed)
    
    X = np.random.randint(0, 2, size=(n_samples, n_features))
    
    fourier_indices, fourier_coeffs = generate_fourier_sparse_function(
        n_features, sparsity, seed
    )
    
    y = evaluate_fourier_function(X, fourier_indices, fourier_coeffs)
    y += np.random.randn(n_samples) * noise_std
    
    return X, y, fourier_indices, fourier_coeffs


def run_single_trial(X_train, y_train, X_test, y_test):
    """Run a single trial with given train/test split and return R² scores."""
    results = {}
    
    # RidgeCV (with cross-validation for alpha selection)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X_train, y_train)
    results['Ridge'] = r2_score(y_test, ridge.predict(X_test))
    
    # Random Forest
    n_trees = min(100, max(10, len(X_train) // 5))
    max_depth = 10 if len(X_train) > 100 else 5
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=42, max_depth=max_depth)
    rf.fit(X_train, y_train)
    results['Random Forest'] = r2_score(y_test, rf.predict(X_test))
    
    # k-NN
    k = min(5, max(1, len(X_train) // 10))
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    results['k-NN (k=5)'] = r2_score(y_test, knn.predict(X_test))
    
    # EBM
    if EBM_AVAILABLE:
        try:
            ebm = ExplainableBoostingRegressor(random_state=42)
            ebm.fit(X_train, y_train)
            results['EBM'] = r2_score(y_test, ebm.predict(X_test))
        except Exception as e:
            print(f"  EBM error: {e}")
            results['EBM'] = np.nan
    
    # TabPFN
    if TABPFN_AVAILABLE:
        try:
            tabpfn = TabPFNRegressor(device='cuda')
            tabpfn.fit(X_train, y_train)
            results['TabPFN'] = r2_score(y_test, tabpfn.predict(X_test))
        except Exception as e:
            print(f"  TabPFN error: {e}")
            results['TabPFN'] = np.nan
    
    return results


def run_experiment(n_features=8, sparsity=5, n_train=500, n_test=200):
    """
    Run experiment comparing TabPFN with baselines on Fourier-sparse functions.
    Averaged over multiple seeds.
    """
    print(f"\n{'='*70}")
    print(f"Experiment: n_features={n_features}, Fourier sparsity k={sparsity}")
    print(f"Training samples: {n_train}, Test samples: {n_test}")
    print(f"Averaging over {N_SEEDS} random seeds")
    print(f"{'='*70}\n")
    
    all_results = defaultdict(list)
    
    for seed in range(N_SEEDS):
        print(f"Running seed {seed+1}/{N_SEEDS}...")
        
        X, y, fourier_indices, fourier_coeffs = generate_dataset(
            n_train + n_test, n_features, sparsity, noise_std=0.5, seed=seed
        )
        
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        results = run_single_trial(X_train, y_train, X_test, y_test)
        for model, score in results.items():
            all_results[model].append(score)
    
    # Compute mean and std
    summary_results = {}
    print(f"\nResults (mean ± std over {N_SEEDS} seeds):")
    print(f"{'Model':<20} {'R² Score':<20}")
    print("-" * 45)
    for model in all_results.keys():
        scores = np.array(all_results[model])
        mean_score = np.nanmean(scores)
        std_score = np.nanstd(scores)
        summary_results[model] = {'mean': mean_score, 'std': std_score, 'all': scores}
        print(f"{model:<20} {mean_score:.4f} ± {std_score:.4f}")
    
    return summary_results


def run_low_data_regime_experiment():
    """Test model performance in the few-shot learning regime."""
    print("\n" + "="*70)
    print("LOW-DATA REGIME EXPERIMENT (Few-Shot Learning)")
    print("="*70)
    
    n_features = 10
    sparsity = 3
    train_sizes = [20, 50, 100, 200, 500]
    n_test = 200
    
    results_by_train_size = defaultdict(lambda: defaultdict(list))
    
    for n_train in train_sizes:
        print(f"\n--- Training samples = {n_train} ---")
        
        for seed in range(N_SEEDS):
            print(f"  Seed {seed+1}/{N_SEEDS}...", end=' ')
            
            X, y, _, _ = generate_dataset(
                n_train + n_test, n_features, sparsity, noise_std=0.5, seed=seed + n_train
            )
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]
            
            results = run_single_trial(X_train, y_train, X_test, y_test)
            for model, score in results.items():
                results_by_train_size[n_train][model].append(score)
            print("done")
    
    # Compute means and stds
    plot_data = {model: {'mean': [], 'std': []} for model in results_by_train_size[train_sizes[0]].keys()}
    
    for n_train in train_sizes:
        for model in results_by_train_size[n_train].keys():
            scores = np.array(results_by_train_size[n_train][model])
            plot_data[model]['mean'].append(np.nanmean(scores))
            plot_data[model]['std'].append(np.nanstd(scores) / np.sqrt(len(scores)))  # Standard error
    
    # Plot
    plt.figure(figsize=(10, 6))
    for model_name in plot_data.keys():
        means = np.array(plot_data[model_name]['mean'])
        sems = np.array(plot_data[model_name]['std'])  # Already computed as standard error
        plt.plot(train_sizes, means, 'o-', label=model_name, linewidth=2, markersize=8)
        plt.fill_between(train_sizes,
                        means - sems,
                        means + sems,
                        alpha=0.2)
    
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title(f'Sample Efficiency on Fourier-Sparse Functions (n={N_SEEDS} seeds, k={sparsity})\n(Higher is better, error bars show standard error)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('low_data_regime.png', dpi=150, bbox_inches='tight')
    
    # Save results
    with open('low_data_regime_results.pkl', 'wb') as f:
        pickle.dump({'train_sizes': train_sizes, 'results': dict(results_by_train_size), 'plot_data': plot_data}, f)
    
    print(f"\nResults saved to 'low_data_regime_results.pkl'")
    print(f"Plot saved to 'low_data_regime.png'")
    
    return results_by_train_size


def run_noise_robustness_experiment():
    """Test how model performance varies with different noise levels."""
    print("\n" + "="*70)
    print("NOISE ROBUSTNESS EXPERIMENT")
    print("="*70)
    
    n_features = 8
    sparsity = 3
    n_train = 300
    n_test = 200
    noise_levels = [0.1, 0.3, 0.5, 1.0, 2.0]
    
    results_by_noise = defaultdict(lambda: defaultdict(list))
    
    for noise_std in noise_levels:
        print(f"\n--- Noise std = {noise_std:.2f} ---")
        
        for seed in range(N_SEEDS):
            print(f"  Seed {seed+1}/{N_SEEDS}...", end=' ')
            
            X, y, _, _ = generate_dataset(
                n_train + n_test, n_features, sparsity, noise_std=noise_std, seed=seed + int(noise_std * 100)
            )
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]
            
            results = run_single_trial(X_train, y_train, X_test, y_test)
            for model, score in results.items():
                results_by_noise[noise_std][model].append(score)
            print("done")
    
    # Compute means and stds
    plot_data = {model: {'mean': [], 'std': []} for model in results_by_noise[noise_levels[0]].keys()}
    
    for noise_std in noise_levels:
        for model in results_by_noise[noise_std].keys():
            scores = np.array(results_by_noise[noise_std][model])
            plot_data[model]['mean'].append(np.nanmean(scores))
            plot_data[model]['std'].append(np.nanstd(scores) / np.sqrt(len(scores)))  # Standard error
    
    # Plot
    plt.figure(figsize=(10, 6))
    for model_name in plot_data.keys():
        means = np.array(plot_data[model_name]['mean'])
        sems = np.array(plot_data[model_name]['std'])  # Already computed as standard error
        plt.plot(noise_levels, means, 'o-', label=model_name, linewidth=2, markersize=8)
        plt.fill_between(noise_levels,
                        means - sems,
                        means + sems,
                        alpha=0.2)
    
    plt.xlabel('Noise Standard Deviation', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title(f'Noise Robustness on Fourier-Sparse Functions (n={N_SEEDS} seeds, k={sparsity})\n(Higher is better, error bars show standard error)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('noise_robustness.png', dpi=150, bbox_inches='tight')
    
    # Save results
    with open('noise_robustness_results.pkl', 'wb') as f:
        pickle.dump({'noise_levels': noise_levels, 'results': dict(results_by_noise), 'plot_data': plot_data}, f)
    
    print(f"\nResults saved to 'noise_robustness_results.pkl'")
    print(f"Plot saved to 'noise_robustness.png'")
    
    return results_by_noise


def run_extreme_sparsity_experiment():
    """Test on extremely sparse functions (k=1,2,3) - needle in a haystack."""
    print("\n" + "="*70)
    print("EXTREME SPARSITY EXPERIMENT (Needle in Haystack)")
    print("="*70)
    
    n_features = 15
    extreme_sparsities = [1, 2, 3]
    n_train = 400
    n_test = 200
    
    results_by_extreme_sparsity = defaultdict(lambda: defaultdict(list))
    
    for sparsity in extreme_sparsities:
        print(f"\n--- Sparsity k = {sparsity} (out of 2^{n_features} = {2**n_features} possible Fourier terms) ---")
        
        for seed in range(N_SEEDS):
            print(f"  Seed {seed+1}/{N_SEEDS}...", end=' ')
            
            X, y, fourier_indices, fourier_coeffs = generate_dataset(
                n_train + n_test, n_features, sparsity, noise_std=0.3, seed=seed + sparsity * 1000
            )
            
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]
            
            results = run_single_trial(X_train, y_train, X_test, y_test)
            for model, score in results.items():
                results_by_extreme_sparsity[sparsity][model].append(score)
            print("done")
    
    # Compute means and stds
    plot_data = {model: {'mean': [], 'std': []} for model in results_by_extreme_sparsity[extreme_sparsities[0]].keys()}
    
    for sparsity in extreme_sparsities:
        for model in results_by_extreme_sparsity[sparsity].keys():
            scores = np.array(results_by_extreme_sparsity[sparsity][model])
            plot_data[model]['mean'].append(np.nanmean(scores))
            plot_data[model]['std'].append(np.nanstd(scores) / np.sqrt(len(scores)))  # Standard error
    
    # Plot
    plt.figure(figsize=(10, 6))
    for model_name in plot_data.keys():
        means = np.array(plot_data[model_name]['mean'])
        sems = np.array(plot_data[model_name]['std'])  # Already computed as standard error
        plt.plot(extreme_sparsities, means, 'o-', label=model_name, linewidth=2, markersize=10)
        plt.fill_between(extreme_sparsities,
                        means - sems,
                        means + sems,
                        alpha=0.2)
    
    plt.xlabel('Fourier Sparsity (k)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title(f'Extreme Sparsity: Finding the Needle in the Haystack (n={N_SEEDS} seeds)\n({n_features} features, 2^{n_features} = {2**n_features} possible Fourier terms, error bars show standard error)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(extreme_sparsities)
    plt.tight_layout()
    plt.savefig('extreme_sparsity.png', dpi=150, bbox_inches='tight')
    
    # Save results
    with open('extreme_sparsity_results.pkl', 'wb') as f:
        pickle.dump({'sparsities': extreme_sparsities, 'results': dict(results_by_extreme_sparsity), 'plot_data': plot_data}, f)
    
    print(f"\nResults saved to 'extreme_sparsity_results.pkl'")
    print(f"Plot saved to 'extreme_sparsity.png'")
    
    return results_by_extreme_sparsity


if __name__ == "__main__":
    print("\nTabPFN Fourier Sparse Functions Experiment")
    print("=" * 70)
    print(f"\nAll experiments average results over {N_SEEDS} random seeds")
    print("Error bars show standard error of the mean")
    print("Results and plots are saved to pickle files for later analysis")
    print("\nConcept from: SPEX paper (arxiv.org/abs/2502.13870)")
    print("A function is k-sparse if only k Fourier coefficients are non-zero.")
    
    # Run single experiment
    run_experiment(n_features=8, sparsity=5, n_train=500, n_test=200)
    
    # Run low-data regime experiment
    run_low_data_regime_experiment()
    
    # Run noise robustness experiment
    run_noise_robustness_experiment()
    
    # Run extreme sparsity experiment
    run_extreme_sparsity_experiment()
    
    print("\n" + "="*70)
    print("All experiments complete!")
    print("\nGenerated files:")
    print("  Plots:")
    print("    - low_data_regime.png")
    print("    - noise_robustness.png")
    print("    - extreme_sparsity.png")
    print("  Results (pickle files):")
    print("    - low_data_regime_results.pkl")
    print("    - noise_robustness_results.pkl")
    print("    - extreme_sparsity_results.pkl")
    print("="*70)