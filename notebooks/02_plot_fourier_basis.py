"""
Generate a combined figure with all three experimental results side by side.
This script loads the saved pickle files and creates a publication-ready figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Set publication-quality plot parameters
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

def plot_combined_results():
    """
    Create a combined figure with three subplots showing all experimental results.
    """
    # Check if all result files exist
    required_files = [
        'low_data_regime_results.pkl',
        'noise_robustness_results.pkl',
        'extreme_sparsity_results.pkl'
    ]
    
    for filename in required_files:
        if not Path(filename).exists():
            print(f"Error: {filename} not found. Please run the main experiment first.")
            return
    
    # Load results
    with open('low_data_regime_results.pkl', 'rb') as f:
        low_data = pickle.load(f)
    
    with open('noise_robustness_results.pkl', 'rb') as f:
        noise_data = pickle.load(f)
    
    with open('extreme_sparsity_results.pkl', 'rb') as f:
        extreme_data = pickle.load(f)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.0))
    
    # Define colors for each model (consistent across all plots)
    colors = {
        'Ridge': '#1f77b4',
        'Random Forest': '#ff7f0e',
        'k-NN (k=5)': '#2ca02c',
        'EBM': '#9467bd',
        'TabPFN': '#d62728'
    }
    
    # Plot 1: Low-Data Regime
    ax = axes[0]
    train_sizes = low_data['train_sizes']
    plot_data = low_data['plot_data']
    
    for model_name in plot_data.keys():
        means = np.array(plot_data[model_name]['mean'])
        sems = np.array(plot_data[model_name]['std'])
        color = colors.get(model_name, None)
        ax.plot(train_sizes, means, 'o-', label=model_name, linewidth=2, 
                markersize=6, color=color)
        ax.fill_between(train_sizes, means - sems, means + sems, alpha=0.2, color=color)
    
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('$R^2$ Score')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('(a)', loc='left', fontweight='bold', fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 2: Noise Robustness
    ax = axes[1]
    noise_levels = noise_data['noise_levels']
    plot_data = noise_data['plot_data']
    
    for model_name in plot_data.keys():
        means = np.array(plot_data[model_name]['mean'])
        sems = np.array(plot_data[model_name]['std'])
        color = colors.get(model_name, None)
        ax.plot(noise_levels, means, 'o-', label=model_name, linewidth=2, 
                markersize=6, color=color)
        ax.fill_between(noise_levels, means - sems, means + sems, alpha=0.2, color=color)
    
    ax.set_xlabel('Noise Standard Deviation ($\sigma$)')
    ax.set_ylabel('$R^2$ Score')
    ax.grid(True, alpha=0.3)
    ax.set_title('(b)', loc='left', fontweight='bold', fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 3: Extreme Sparsity
    ax = axes[2]
    sparsities = extreme_data['sparsities']
    plot_data = extreme_data['plot_data']
    
    for model_name in plot_data.keys():
        means = np.array(plot_data[model_name]['mean'])
        sems = np.array(plot_data[model_name]['std'])
        color = colors.get(model_name, None)
        ax.plot(sparsities, means, 'o-', label=model_name, linewidth=2, 
                markersize=8, color=color)
        ax.fill_between(sparsities, means - sems, means + sems, alpha=0.2, color=color)
    
    ax.set_xlabel('Fourier Sparsity ($k$)')
    ax.set_ylabel('$R^2$ Score')
    ax.set_xticks(sparsities)
    ax.grid(True, alpha=0.3)
    ax.set_title('(c)', loc='left', fontweight='bold', fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create shared legend at the bottom
    # Get handles and labels from the last plot (all plots have same models)
    handles, labels = ax.get_legend_handles_labels()
    
    # Add legend below the subplots, centered, horizontal orientation
    fig.legend(handles, labels, loc='lower center', ncol=5, 
               bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=11)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
    plt.savefig('combined_fourier_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('combined_fourier_results.pdf', bbox_inches='tight')
    
    print("Combined figure saved as:")
    print("  - combined_fourier_results.png (300 DPI)")
    print("  - combined_fourier_results.pdf (vector format)")
    
    # Print LaTeX caption
    print("\n" + "="*70)
    print("LaTeX Figure Caption:")
    print("="*70)
    print_latex_caption()
    
    plt.show()


def print_latex_caption():
    """Print a publication-ready LaTeX figure caption."""
    caption = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{combined_fourier_results.pdf}
\caption{Performance comparison of TabPFN against baseline methods on Fourier-sparse functions. All results are averaged over 10 random seeds with shaded regions showing standard error. \textbf{(a) Low-data regime:} $R^2$ score vs.\ training set size with $n=10$ features and $k=3$ Fourier sparsity. TabPFN demonstrates superior sample efficiency, achieving high performance with as few as 20 training samples. \textbf{(b) Noise robustness:} $R^2$ score vs.\ noise standard deviation $\sigma$ with $n=8$ features, $k=3$ sparsity, and 300 training samples. TabPFN maintains robust performance across different signal-to-noise ratios. \textbf{(c) Extreme sparsity:} $R^2$ score vs.\ Fourier sparsity $k$ with $n=15$ features and 400 training samples. With only $k \in \{1,2,3\}$ non-zero coefficients out of $2^{15}=32{,}768$ possible Fourier terms, TabPFN successfully identifies the minimal feature interactions while baseline methods struggle with this ``needle in haystack'' setting.}
\label{fig:fourier_sparse_results}
\end{figure*}
"""
    print(caption)


if __name__ == "__main__":
    print("Generating combined results figure...")
    print("="*70)
    plot_combined_results()