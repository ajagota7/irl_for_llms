import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import pandas as pd

def load_metrics_for_loss_type(base_dir, loss_type, print_values=False):
    """Load all metrics for a specific loss type across all runs."""
    runs_metrics = []
    
    # Look for directories matching this loss type
    for dirname in sorted(os.listdir(base_dir)):
        if f"exp_70m_loss_{loss_type}_run_" in dirname:
            run_dir = os.path.join(base_dir, dirname)
            metrics_dir = os.path.join(run_dir, "metrics")
            
            # Load all epochs for this run
            run_metrics = []
            if os.path.exists(metrics_dir):
                for metrics_file in sorted(os.listdir(metrics_dir)):
                    if metrics_file.startswith("metrics_epoch_"):
                        with open(os.path.join(metrics_dir, metrics_file)) as f:
                            metrics = json.load(f)
                            run_metrics.append(metrics)
            
            if run_metrics:
                runs_metrics.append(run_metrics)
    
    return runs_metrics

def print_structured_metrics(loss_type, runs_metrics, metric_key):
    """Print metrics in a structured way using pandas DataFrame."""
    if not runs_metrics:
        print(f"No data for {loss_type}")
        return
    
    # Create a dictionary to store values for each run
    run_values = defaultdict(list)
    
    # Get values for each run
    for run_idx, run_metrics in enumerate(runs_metrics):
        values = [epoch[metric_key] for epoch in run_metrics]
        run_values[f'Run {run_idx + 1}'] = values
    
    # Calculate mean across runs
    mean_values = np.mean([[epoch[metric_key] for epoch in run] 
                          for run in runs_metrics], axis=0)
    run_values['Mean'] = mean_values
    
    # Create DataFrame
    df = pd.DataFrame(run_values)
    df.index.name = 'Epoch'
    
    print(f"\n{loss_type} - {metric_key}:")
    print("=" * 80)
    print(df.round(3))
    print("-" * 80)

def plot_loss_type_comparison(base_dir="/content/drive/MyDrive/irl_experiments_loss_type"):
    """Create comparison plots for different loss types showing individual runs."""
    loss_types = ["max_margin", "squared", "logistic", "hinge"]
    
    # Metrics to plot
    plot_metrics = {
        'correlation': 'Pearson Correlation',
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'kendall_tau': 'Kendall Tau',
        'spearman': 'Spearman Correlation',
        'average_loss': 'Average Loss'
    }
    
    # First print structured metrics for each loss type and metric
    print("\nDetailed Metrics for Each Loss Type and Run:")
    print("=" * 80)
    for metric_key in plot_metrics.keys():
        print(f"\n{metric_key.upper()}:")
        print("=" * 80)
        for loss_type in loss_types:
            runs_metrics = load_metrics_for_loss_type(base_dir, loss_type)
            print_structured_metrics(loss_type, runs_metrics, metric_key)
    
    # Create plots
    n_metrics = len(plot_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 6*n_metrics))
    fig.suptitle('Comparison of Different Loss Types (Individual Runs and Mean)', size=16, y=1.02)
    
    # Colors for different loss types
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_types)))
    
    for idx, (metric_key, metric_name) in enumerate(plot_metrics.items()):
        ax = axes[idx]
        
        for color_idx, loss_type in enumerate(loss_types):
            runs_metrics = load_metrics_for_loss_type(base_dir, loss_type)
            
            if runs_metrics:
                # Plot individual runs with light lines
                for run_idx, run_metrics in enumerate(runs_metrics):
                    values = [epoch[metric_key] for epoch in run_metrics]
                    epochs = range(len(values))
                    
                    ax.plot(epochs, values, 
                           alpha=0.3, 
                           color=colors[color_idx],
                           linestyle='--',
                           label=f'{loss_type} run {run_idx+1}' if run_idx == 0 else None)
                
                # Calculate and plot mean
                mean_values = np.mean([[epoch[metric_key] for epoch in run] 
                                     for run in runs_metrics], axis=0)
                
                ax.plot(epochs, mean_values,
                       color=colors[color_idx],
                       linewidth=2,
                       label=f'{loss_type} (mean)')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Over Epochs')
        ax.grid(True, alpha=0.3)
        
        # Move legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set reasonable y-axis limits for different metrics
        if metric_key == 'accuracy':
            ax.set_ylim(0, 1)  # Accuracy should be between 0 and 1
        elif metric_key in ['correlation', 'kendall_tau', 'spearman', 'cosine_similarity']:
            ax.set_ylim(-1, 1)  # Correlation metrics should be between -1 and 1
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(base_dir, 'loss_types_comparison_with_individual_runs.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"\nPlot saved to: {save_path}")
