import os
import json
import torch
from datetime import datetime
from omegaconf import OmegaConf

def setup_save_directories(cfg, experiments_base_dir="/content/drive/MyDrive"):
    """Setup save directories with configurable base path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Debug prints
    print("\nDEBUG SAVING INFO:")
    print(f"Base dir received: {experiments_base_dir}")
    
    # Create experiment type name
    exp_type = f"exp_70m_loss_{cfg.experiment.training.loss_type}"
    print(f"Experiment type: {exp_type}")
    
    # Create directory paths
    run_dir = os.path.join(experiments_base_dir, f"{exp_type}_run_{timestamp}")
    models_dir = os.path.join(run_dir, "models")
    metrics_dir = os.path.join(run_dir, "metrics")
    
    print(f"Full run directory path: {run_dir}")
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    return run_dir, models_dir, metrics_dir

def save_model_and_metrics(model, metrics, epoch, models_dir, metrics_dir):
    """Save model and metrics for a given epoch."""
    # Save model with both RM and GT metrics in filename
    model_path = os.path.join(
        models_dir, 
        f"model_epoch_{epoch}_rm_acc_{metrics['train_metrics']['accuracy_rm']:.3f}_"
        f"gt_acc_{metrics['train_metrics']['accuracy_gt']:.3f}.pt"
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")
    
    # Save detailed metrics
    metrics_path = os.path.join(metrics_dir, f"metrics_epoch_{epoch}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

def save_config(config, run_dir):
    """Save experiment config."""
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(OmegaConf.to_container(config), f, indent=2)
    print(f"Saved config to: {config_path}")
