import os
import json
import torch
from datetime import datetime
from google.colab import drive
from omegaconf import OmegaConf

def setup_save_directories(cfg, base_dir="/content/drive/MyDrive/irl_experiments"):
    """Setup the basic directory structure for saving experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract experiment type from learn_rm and learn_rm_size
    exp_type = f"exp_{cfg.experiment.learn_rm_size}" if cfg.experiment.learn_rm_size else "exp_70m"
    
    # Create main directories with experiment type in name
    run_dir = os.path.join(base_dir, f"{exp_type}_run_{timestamp}")
    models_dir = os.path.join(run_dir, "models")
    metrics_dir = os.path.join(run_dir, "metrics")
    
    # Create all directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    return run_dir, models_dir, metrics_dir

def save_model_and_metrics(model, metrics, epoch, models_dir, metrics_dir):
    """Save model and metrics for a given epoch."""
    # Save model
    model_path = os.path.join(
        models_dir, 
        f"model_epoch_{epoch}_corr_{metrics['correlation']:.3f}_acc_{metrics['accuracy']:.3f}.pt"
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(metrics_dir, f"metrics_epoch_{epoch}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
def save_config(config, run_dir):
    """Save experiment config."""
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(OmegaConf.to_container(config), f, indent=2)
