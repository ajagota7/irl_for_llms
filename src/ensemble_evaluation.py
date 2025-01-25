import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.irl_utilities import get_evaluation, get_true_reward_model, get_reward_score
from src.ensemble_rewards import create_ensemble_model
import datasets

def collect_ensemble_results(base_dir="/content/drive/MyDrive/irl_experiments", 
                           exp_type="exp_70m",
                           ensemble_methods=["average", "weighted", "median"]):
    """Just collect the results without plotting"""
    # Load test samples
    dataset_ntoxic = datasets.load_dataset("skrishna/jaredjoss-jigsaw-long-2000_70M_non_toxic")
    dataset_toxic = datasets.load_dataset("skrishna/jaredjoss-jigsaw-long-2000_70M_toxic")
    test_samples = ([sample["output"] for sample in dataset_toxic["train"]] + 
                   [sample["output"] for sample in dataset_ntoxic["train"]])

    # Get number of epochs
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith(exp_type)]
    first_run = os.path.join(base_dir, run_dirs[0])
    n_epochs = len([f for f in os.listdir(os.path.join(first_run, "metrics")) 
                   if f.startswith("metrics_epoch")])

    # Initialize results
    results = {method: [] for method in ensemble_methods}
    individual_results = []

    # Get true reward model
    true_rm = "s-nlp/roberta_toxicity_classifier"
    true_model, true_tokenizer = get_true_reward_model(true_rm)

    print(f"Evaluating {len(ensemble_methods)} ensemble methods over {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        print(f"\nProcessing epoch {epoch}")
        epoch_results = {'epoch': epoch}
        
        # Evaluate ensembles
        for method in ensemble_methods:
            print(f"  Evaluating {method} ensemble...")
            ensemble = create_ensemble_model(base_dir, exp_type, epoch, method)
            metrics = get_evaluation(test_samples, true_rm, [ensemble.models[0], ensemble.tokenizer])
            results[method].append(metrics)

        # Evaluate individual models
        print("  Evaluating individual models...")
        individual_metrics = []
        for run_dir in run_dirs:
            metrics_file = os.path.join(base_dir, run_dir, "metrics", f"metrics_epoch_{epoch}.json")
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    individual_metrics.append(metrics)
        
        if individual_metrics:
            # Calculate average metrics for this epoch
            epoch_avg = {}
            for key in individual_metrics[0].keys():
                if key != 'epoch':  # Skip the epoch number
                    values = [m[key] for m in individual_metrics]
                    epoch_avg[key] = np.mean(values)
            individual_results.append(epoch_avg)

    # Save results
    all_results = {
        'ensemble': results,
        'individual': individual_results
    }
    
    save_path = os.path.join(base_dir, f'{exp_type}_ensemble_results.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def plot_ensemble_results(results, exp_type, base_dir="/content/drive/MyDrive/irl_experiments"):
    """Plot results from collected data"""
    metrics_to_plot = ['correlation', 'accuracy', 'f1', 'kendall_tau', 'spearman']
    ensemble_methods = list(results['ensemble'].keys())
    n_epochs = len(results['ensemble'][ensemble_methods[0]])

    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4*len(metrics_to_plot)))
    fig.suptitle('Ensemble vs Individual Model Performance')

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Plot ensemble methods
        for method in ensemble_methods:
            values = [epoch[metric] for epoch in results['ensemble'][method]]
            ax.plot(values, label=f'{method} ensemble', marker='o')
        
        # Plot individual average
        ind_values = [epoch[metric] for epoch in results['individual']]
        ax.plot(ind_values, label='Individual average', linestyle='--', color='gray')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(base_dir, f'{exp_type}_ensemble_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def evaluate_ensembles_over_epochs(base_dir="/content/drive/MyDrive/irl_experiments", 
                                 exp_type="exp_70m",
                                 ensemble_methods=["average", "weighted", "median"]):
    """Run both collection and plotting"""
    results = collect_ensemble_results(base_dir, exp_type, ensemble_methods)
    plot_ensemble_results(results, exp_type, base_dir)
    return results
