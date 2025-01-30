from transformers import AutoTokenizer, GPTNeoXForCausalLM, RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch import nn
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RewardModel(nn.Module):
    def __init__(self, checkpoint_path, eos_token_id):
        super().__init__()
        # Load model with CPU offloading for large models
        model = GPTNeoXForCausalLM.from_pretrained(
            checkpoint_path,
            device_map='auto',
            torch_dtype=torch.float32  # Use float32 instead of float16
        )
        self.model = model
        self.v_head = nn.Linear(model.gpt_neox.embed_in.embedding_dim, 2, bias=False)
        self.v_head = self.v_head.to(device)
        self.eos_token_id = eos_token_id
        
    def forward(self, input_ids):
        # Get the hidden states from the base model
        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_hidden = hidden_states[:, -1, :].to(device)
        
        # Apply the value head
        value = self.v_head(last_hidden)
        return value.float()  # Ensure output is float32

def get_initial_model(checkpoint_path):
    """
    Returns initial reward model. 
    """
    reward_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model = RewardModel(checkpoint_path, reward_tokenizer.eos_token_id)
    reward_model = reward_model.to(device)
    return reward_model.requires_grad_(), reward_tokenizer

def load_saved_model(checkpoint_path, learn_rm):
    """
    Loads a saved model state from the given checkpoint path.
    """
    reward_tokenizer = AutoTokenizer.from_pretrained(learn_rm)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model = RewardModel(learn_rm, reward_tokenizer.eos_token_id)
    reward_model.load_state_dict(torch.load(checkpoint_path))
    reward_model = reward_model.to(device)
    return reward_model.requires_grad_(), reward_tokenizer

def get_true_reward_model(true_rm="s-nlp/roberta_toxicity_classifier"):
    """
    Returns true reward model. 
    """
    reward_tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier').to(device)
    return reward_model, reward_tokenizer

def get_irl_loss(loss_type="max_margin"):
    """
    Returns loss function for optimization. 
    """
    if loss_type == "max_margin":
        def loss(reward_n_t, reward_t):
            def custom_operator(input_vector):
                # Create new tensor with gradients
                result = input_vector.clone()
                # Apply operations
                result = torch.where(result > 0, -result, -2 * result)
                return result
            
            reward_diff = reward_n_t - reward_t
            reward_diff_dir = custom_operator(reward_diff)
            return torch.mean(reward_diff_dir)
            
    elif loss_type == "squared":
        def loss(reward_n_t: torch.Tensor, reward_t: torch.Tensor) -> torch.Tensor:
            reward_diff = reward_n_t - reward_t
            return torch.mean(torch.square(reward_diff))
            
    elif loss_type == "logistic":
        def loss(reward_n_t: torch.Tensor, reward_t: torch.Tensor) -> torch.Tensor:
            reward_diff = reward_n_t - reward_t
            return -torch.mean(torch.log(torch.sigmoid(reward_diff)))
            
    elif loss_type == "hinge":
        def loss(reward_n_t: torch.Tensor, reward_t: torch.Tensor) -> torch.Tensor:
            reward_diff = reward_n_t - reward_t
            return torch.mean(torch.nn.functional.relu(1 - reward_diff))
    
    return loss

def get_optimizer(model, optim_type="adam", lr=0.00001, momentum=0.9):
    """
    Select and return suitable optimizer with training hyper-params. 
    """
    if optim_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        pass

def lp_norm(x, y, p):
    """
    Returns the lp norm between two lists.
    """
    x = np.array(x)
    y = np.array(y)
    diff = np.abs(x - y)
    lp_norm_value = np.sum(diff**p)**(1/p)
    return lp_norm_value

def get_evaluation(samples_data, true_rm, learned_rm, batch_size=32):
    """
    Returns all metrics comparing against both reward model predictions and ground truth labels.
    """
    true_rm, true_rm_tokenizer = get_true_reward_model(true_rm)
    learned_rm, learned_rm_tokenizer = learned_rm[0], learned_rm[1]
    
    samples = samples_data['samples']
    ground_truth_labels = samples_data['ground_truth_labels']
    ground_truth_scores = np.array([1.0 if label == 1 else -1.0 for label in ground_truth_labels])
    
    true_rewards = []
    learned_rewards = []
    
    # Process in batches
    for i in range(0, len(samples), batch_size):
        batch = samples[i:min(i+batch_size, len(samples))]
        
        # Process true rewards
        inputs = true_rm_tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            true_batch_rewards = true_rm(**inputs).logits
        true_rewards.extend((-1 * true_batch_rewards[:, -1]).cpu().numpy())
        
        # Process learned rewards
        inputs = learned_rm_tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            learned_batch_rewards = learned_rm(inputs.input_ids)
        learned_rewards.extend((-1 * learned_batch_rewards[:, -1]).cpu().numpy())
        
        # Clear GPU cache periodically
        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    true_rewards = np.array(true_rewards)
    learned_rewards = np.array(learned_rewards)
    
    # Convert rewards to binary predictions for accuracy/f1
    learned_predictions = np.where(learned_rewards > 0, 0, 1)
    true_rm_predictions = np.where(true_rewards > 0, 0, 1)
    
    # Compute metrics against reward model predictions
    rm_metrics = {
        'pearson_correlation_rm': np.corrcoef(true_rewards, learned_rewards)[0,1],
        'euclidean_distance_rm': lp_norm(true_rewards, learned_rewards, 2),
        'kendall_tau_rm': stats.kendalltau(true_rewards, learned_rewards)[0],
        'spearman_rm': stats.spearmanr(true_rewards, learned_rewards)[0],
        'cosine_similarity_rm': 1 - cosine(true_rewards, learned_rewards),
        'accuracy_rm': accuracy_score(true_rm_predictions, learned_predictions),
        'f1_rm': f1_score(true_rm_predictions, learned_predictions)
    }
    
    # Compute all metrics against ground truth
    gt_metrics = {
        'pearson_correlation_gt': np.corrcoef(ground_truth_scores, learned_rewards)[0,1],
        'euclidean_distance_gt': lp_norm(ground_truth_scores, learned_rewards, 2),
        'kendall_tau_gt': stats.kendalltau(ground_truth_scores, learned_rewards)[0],
        'spearman_gt': stats.spearmanr(ground_truth_scores, learned_rewards)[0],
        'cosine_similarity_gt': 1 - cosine(ground_truth_scores, learned_rewards),
        'accuracy_gt': accuracy_score(ground_truth_labels, learned_predictions),
        'f1_gt': f1_score(ground_truth_labels, learned_predictions)
    }
    
    # Combine all metrics
    metrics = {**rm_metrics, **gt_metrics}
    
    return metrics

def get_reward_score(reward_model, input_text, tokenizer, batch_size=32):
    """Takes reward model and input and returns reward score with batch processing."""
    if isinstance(input_text, str):
        input_text = [input_text]
        
    all_outputs = []
    
    for i in range(0, len(input_text), batch_size):
        batch = input_text[i:min(i+batch_size, len(input_text))]
        input_ids = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        output = reward_model(input_ids.input_ids)
        output = -1 * output[:, -1]  # Get the second logit and negate it
        all_outputs.append(output.float())  # Ensure float32
            
        # Clear memory
        del input_ids
        torch.cuda.empty_cache()
    
    # Concatenate all outputs
    final_output = torch.cat(all_outputs, dim=0)
    return final_output.float()  # Ensure float32

def data_loader(list1, list2, batch_size=32):
    """
    Creates batches with dynamic batch sizing based on available GPU memory
    """
    assert len(list1) == len(list2), "Both lists should have the same length"
    
    # Get current GPU memory usage
    if torch.cuda.is_available():
        gpu = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(gpu).total_memory
        reserved_memory = torch.cuda.memory_reserved(gpu)
        available_memory = total_memory - reserved_memory
        
        # Dynamically adjust batch size based on available memory
        # Using 80% of available memory as a safe limit
        memory_per_sample = 1024 * 1024 * 10  # Estimate: 10MB per sample
        max_possible_batch = int((available_memory * 0.8) / memory_per_sample)
        
        # Cap the batch size at a reasonable maximum
        batch_size = min(max_possible_batch, 128)
        print(f"Using dynamic batch size: {batch_size}")
    
    for i in range(0, len(list1), batch_size):
        batch1 = list1[i:min(i+batch_size, len(list1))]
        batch2 = list2[i:min(i+batch_size, len(list2))]
        
        yield batch1, batch2
