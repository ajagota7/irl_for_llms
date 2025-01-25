import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_run_metrics(run_dir):
    """Load all metrics files from a single run directory."""
    metrics_dir = os.path.join(run_dir, "metrics")
    metrics_by_epoch = {}
    
    try:
        if not os.path.isdir(metrics_dir):
            print(f"Warning: {metrics_dir} is not a directory")
            return metrics_by_epoch
            
        for filename in sorted(os.listdir(metrics_dir)):
            if filename.startswith("metrics_epoch_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(metrics_dir, filename)) as f:
                        metrics = json.load(f)
                        epoch = metrics['epoch']
                        metrics_by_epoch[epoch] = metrics
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    except Exception as e:
        print(f"Error accessing directory {metrics_dir}: {e}")
    
    return metrics_by_epoch

def collect_all_runs(base_dir, exp_type="exp_70m"):
    """Collect metrics from all runs of a specific experiment type."""
    all_runs_metrics = []
    
    try:
        # Find all directories matching the experiment type
        for dirname in os.listdir(base_dir):
            if dirname.startswith(exp_type) and "_combined_metrics" not in dirname:  # Skip previously generated plot files
                run_dir = os.path.join(base_dir, dirname)
                if os.path.isdir(run_dir):  # Check if it's actually a directory
                    run_metrics = load_run_metrics(run_dir)
                    if run_metrics:  # Only add if we got metrics
                        all_runs_metrics.append(run_metrics)
    except Exception as e:
        print(f"Error accessing directory {base_dir}: {e}")
    
    print(f"Found {len(all_runs_metrics)} runs for {exp_type}")
    return all_runs_metrics

def compute_statistics(all_runs_metrics, metric_name):
    """Compute mean and confidence intervals for a specific metric across all runs."""
    if not all_runs_metrics:
        return [], []
        
    n_epochs = max(max(run.keys()) for run in all_runs_metrics) + 1
    all_values = []
    
    # Collect values for each epoch
    for epoch in range(n_epochs):
        epoch_values = []
        for run in all_runs_metrics:
            if epoch in run and metric_name in run[epoch]:
                epoch_values.append(run[epoch][metric_name])
        all_values.append(epoch_values)
    
    # Compute statistics
    means = [np.mean(values) if values else 0 for values in all_values]
    
    # Compute 95% confidence intervals
    conf_intervals = []
    for values in all_values:
        if len(values) > 1:
            mean = np.mean(values)
            se = stats.sem(values)
            ci = stats.t.interval(confidence=0.95, df=len(values)-1,
                                loc=mean,
                                scale=se)
            conf_intervals.append((ci[0], ci[1]))
        elif len(values) == 1:
            # If only one value, use it for both bounds
            conf_intervals.append((values[0], values[0]))
        else:
            conf_intervals.append((0, 0))
    
    return means, conf_intervals

def plot_metric_with_ci(ax, epochs, means, conf_intervals, metric_name, color):
    """Plot a single metric with confidence intervals."""
    if not means or not conf_intervals:
        print(f"No data to plot for {metric_name}")
        return
        
    lower_ci = [ci[0] for ci in conf_intervals]
    upper_ci = [ci[1] for ci in conf_intervals]
    
    ax.plot(epochs, means, label=metric_name, color=color)
    ax.fill_between(epochs, lower_ci, upper_ci, color=color, alpha=0.2)
    
    # Add mean ± std for last epoch
    last_mean = means[-1]
    last_ci = conf_intervals[-1]
    ax.text(0.02, 0.98, f'Final: {last_mean:.3f}\nCI: [{last_ci[0]:.3f}, {last_ci[1]:.3f}]',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_dual_axis_plot(base_dir="/content/drive/MyDrive/irl_experiments", exp_type="exp_70m"):
    """Create a dual-axis plot with accuracy and correlations."""
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Collect all runs
    all_runs_metrics = collect_all_runs(base_dir, exp_type)
    
    if not all_runs_metrics:
        print(f"No runs found for experiment type: {exp_type}")
        return
        
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    
    # Colors for different metrics
    colors = {
        'accuracy': '#2ecc71',  # green
        'correlation': '#3498db',  # blue
        'spearman': '#e74c3c',  # red
        'kendall_tau': '#9b59b6'  # purple
    }
    
    # Plot accuracy on left axis
    means_acc, conf_acc = compute_statistics(all_runs_metrics, 'accuracy')
    epochs = range(len(means_acc))
    ax1.plot(epochs, means_acc, color=colors['accuracy'], label='Accuracy', linewidth=2)
    ax1.fill_between(epochs, 
                     [ci[0] for ci in conf_acc],
                     [ci[1] for ci in conf_acc],
                     color=colors['accuracy'], alpha=0.2)
    
    # Plot correlations on right axis
    correlation_metrics = ['correlation', 'spearman', 'kendall_tau']
    correlation_labels = ['Pearson Correlation', 'Spearman Correlation', 'Kendall Tau']
    
    for metric, label, color in zip(correlation_metrics, correlation_labels, 
                                  [colors['correlation'], colors['spearman'], colors['kendall_tau']]):
        means, conf = compute_statistics(all_runs_metrics, metric)
        ax2.plot(epochs, means, color=color, label=label, linewidth=2)
        ax2.fill_between(epochs,
                        [ci[0] for ci in conf],
                        [ci[1] for ci in conf],
                        color=color, alpha=0.2)
    
    # Customize axes
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax2.set_ylabel('Correlation')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Customize title
    plt.title(f'{exp_type} IRL Accuracy and Correlation for Each Epoch over {len(epochs)} Epochs')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Place legend outside and to the right of the plot
    ax2.legend(lines1 + lines2, labels1 + labels2, 
              loc='center left', 
              bbox_to_anchor=(1.15, 0.5))
    
    # Save plot with explicit path
    save_path = os.path.join(base_dir, f'{exp_type}_accuracy_correlations.png')
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Successfully saved dual axis plot to: {save_path}")
    except Exception as e:
        print(f"Error saving plot to {save_path}: {e}")
    
    # Display plot
    plt.show()
    plt.close()

def create_plots(base_dir="/content/drive/MyDrive/irl_experiments", exp_type="exp_70m"):
    """Create all plots for metrics with confidence intervals."""
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Collect all runs
    all_runs_metrics = collect_all_runs(base_dir, exp_type)
    
    if not all_runs_metrics:
        print(f"No runs found for experiment type: {exp_type}")
        return
        
    num_runs = len(all_runs_metrics)
    n_epochs = max(max(run.keys()) for run in all_runs_metrics) + 1
    
    # All metrics to plot
    metrics_to_plot = {
        'correlation': 'Pearson Correlation',
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'average_loss': 'Average Loss',
        'euclidean_distance': 'Euclidean Distance',
        'kendall_tau': 'Kendall Tau',
        'spearman': 'Spearman Correlation',
        'cosine_similarity': 'Cosine Similarity'
    }
    
    # Create subplots version
    n_metrics = len(metrics_to_plot)
    n_rows = (n_metrics + 1) // 2  # Calculate number of rows needed
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
    title = f'IRL Accuracy and Correlation Over {num_runs} runs of {n_epochs} Epochs'
    fig.suptitle(title, size=16, y=1.02)
    axes = axes.flatten()
    
    # Colors for different metrics
    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))
    
    # Plot each metric in subplots
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot.items()):
        means, conf_intervals = compute_statistics(all_runs_metrics, metric_key)
        if means and conf_intervals:  # Only plot if we have data
            epochs = range(len(means))
            plot_metric_with_ci(axes[idx], epochs, means, conf_intervals, metric_name, colors[idx])
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(f'{metric_name} Over Epochs')
            axes[idx].grid(True, alpha=0.3)
    
    # Hide empty subplots if any
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(base_dir, f'{exp_type}_all_metrics_subplots.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    # Create normalized plot with all metrics
    plt.figure(figsize=(15, 10))
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot.items()):
        means, conf_intervals = compute_statistics(all_runs_metrics, metric_key)
        if means and conf_intervals:
            epochs = range(len(means))
            
            # Normalize the values
            means_norm = (means - np.min(means)) / (np.max(means) - np.min(means))
            lower_ci = [(ci[0] - np.min(means)) / (np.max(means) - np.min(means)) for ci in conf_intervals]
            upper_ci = [(ci[1] - np.min(means)) / (np.max(means) - np.min(means)) for ci in conf_intervals]
            
            plt.plot(epochs, means_norm, label=metric_name, color=colors[idx])
            plt.fill_between(epochs, lower_ci, upper_ci, color=colors[idx], alpha=0.2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Value')
    plt.title(f'IRL Metrics Over {num_runs} runs of {n_epochs} Epochs (Normalized)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_path_norm = os.path.join(base_dir, f'{exp_type}_all_metrics_normalized.png')
    plt.savefig(save_path_norm, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    # Create the dual axis plot
    create_dual_axis_plot(base_dir, exp_type)
    
    # Save numerical results
    results = {}
    for metric_key, metric_name in metrics_to_plot.items():
        means, conf_intervals = compute_statistics(all_runs_metrics, metric_key)
        results[metric_key] = {
            'means': means,
            'confidence_intervals': conf_intervals
        }
    
    results_path = os.path.join(base_dir, f'{exp_type}_statistics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Statistics saved to: {results_path}")
