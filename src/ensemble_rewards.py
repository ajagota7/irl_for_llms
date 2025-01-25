import torch
import os
from typing import List, Dict
import numpy as np
from src.irl_utilities import RewardModel, get_reward_score
from transformers import AutoTokenizer

class EnsembleRewardModel:
    def __init__(self, base_dir: str, exp_type: str, epoch: int, ensemble_method: str = "average"):
        """
        Initialize ensemble reward model.
        
        Args:
            base_dir: Directory containing all run folders
            exp_type: Experiment type (e.g., 'exp_70m')
            epoch: Which epoch to ensemble models from
            ensemble_method: One of ["average", "weighted", "median", "vote"]
        """
        self.base_dir = base_dir
        self.exp_type = exp_type
        self.epoch = epoch
        self.ensemble_method = ensemble_method
        self.models = []
        self.weights = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{exp_type[4:].lower()}")
        self._load_models()
        
    def _load_models(self):
        """Load all models from the specified epoch across runs."""
        for dirname in os.listdir(self.base_dir):
            if dirname.startswith(self.exp_type):
                run_dir = os.path.join(self.base_dir, dirname)
                models_dir = os.path.join(run_dir, "models")
                
                if not os.path.exists(models_dir):
                    continue
                    
                # Find model file for specified epoch
                for filename in os.listdir(models_dir):
                    if f"model_epoch_{self.epoch}" in filename:
                        model_path = os.path.join(models_dir, filename)
                        model = self._load_single_model(model_path)
                        self.models.append(model)
        
        print(f"Loaded {len(self.models)} models for epoch {self.epoch}")
        
    def _load_single_model(self, model_path: str):
        """Load a single model from path."""
        model = RewardModel(f"EleutherAI/pythia-{self.exp_type[4:].lower()}", self.tokenizer.eos_token_id)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
        
    def compute_weights(self, validation_data: List[str], true_rewards: List[float]):
        """Compute weights for each model based on validation performance."""
        performances = []
        for model in self.models:
            rewards = []
            for text in validation_data:
                reward = self._get_reward(model, text)
                rewards.append(reward)
            # Use correlation with true rewards as performance metric
            performance = np.corrcoef(true_rewards, rewards)[0,1]
            performances.append(performance)
            
        # Normalize performances to get weights
        performances = np.array(performances)
        self.weights = torch.softmax(torch.tensor(performances), dim=0)
        
    def _get_reward(self, model, text: str) -> float:
        """Get reward for a single text from a single model."""
        with torch.no_grad():
            reward = get_reward_score(model, text, self.tokenizer)
        return reward.cpu().item()
    
    def get_ensemble_reward(self, text: str) -> float:
        """Get ensemble reward for a single text."""
        rewards = [self._get_reward(model, text) for model in self.models]
        
        if self.ensemble_method == "average":
            return np.mean(rewards)
            
        elif self.ensemble_method == "weighted":
            if self.weights is None:
                raise ValueError("Must call compute_weights before using weighted ensemble")
            return np.sum([w * r for w, r in zip(self.weights, rewards)])
            
        elif self.ensemble_method == "median":
            return np.median(rewards)
            
        elif self.ensemble_method == "vote":
            # Convert rewards to binary decisions based on threshold
            votes = [r > 0 for r in rewards]
            return float(sum(votes) > len(votes)/2)
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

def create_ensemble_model(base_dir: str, exp_type: str, epoch: int, 
                        ensemble_method: str = "average", 
                        validation_data: List[str] = None,
                        true_rewards: List[float] = None) -> EnsembleRewardModel:
    """
    Create an ensemble model from all available models at specified epoch.
    
    Args:
        base_dir: Directory containing all run folders
        exp_type: Experiment type (e.g., 'exp_70m')
        epoch: Which epoch to ensemble models from
        ensemble_method: One of ["average", "weighted", "median", "vote"]
        validation_data: Optional data for computing weights if using weighted ensemble
        true_rewards: Optional true rewards for validation data if using weighted ensemble
        
    Returns:
        EnsembleRewardModel instance
    """
    ensemble = EnsembleRewardModel(base_dir, exp_type, epoch, ensemble_method)
    
    if ensemble_method == "weighted" and validation_data is not None and true_rewards is not None:
        ensemble.compute_weights(validation_data, true_rewards)
        
    return ensemble

def evaluate_ensemble(ensemble: EnsembleRewardModel, test_data: List[str],
                    true_rewards: List[float]) -> Dict[str, float]:
    """
    Evaluate ensemble model performance.
    
    Args:
        ensemble: EnsembleRewardModel instance
        test_data: List of test texts
        true_rewards: True rewards for test texts
        
    Returns:
        Dictionary of evaluation metrics
    """
    pred_rewards = [ensemble.get_ensemble_reward(text) for text in test_data]
    
    # Compute various metrics
    metrics = {
        'correlation': np.corrcoef(true_rewards, pred_rewards)[0,1],
        'accuracy': np.mean([(p > 0) == (t > 0) for p, t in zip(pred_rewards, true_rewards)]),
        'mse': np.mean((np.array(true_rewards) - np.array(pred_rewards))**2)
    }
    
    return metrics
