from transformers import AutoTokenizer, GPTNeoXForCausalLM, RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch import nn
import pickle
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, f1_score

# New imports for feature extraction
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
import spacy
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    def __init__(self, feature_type, model_checkpoint, device='cuda'):
        self.feature_type = feature_type
        self.device = device
        self.model = AutoModel.from_pretrained(model_checkpoint).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        # Initialize additional models based on feature type
        if 'semantic' in feature_type:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
        if 'structural' in feature_type:
            self.nlp = spacy.load('en_core_web_sm')
            
    def extract_token_features(self, text):
        """Extract token-level embeddings and n-gram statistics"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state
        
        # Add n-gram statistics
        ngrams = self._compute_ngrams(text)
        
        return {
            'token_embeddings': token_embeddings,
            'ngram_stats': ngrams
        }
    
    def extract_semantic_features(self, text):
        """Extract semantic features using sentence transformers and topic modeling"""
        # Get sentence embeddings
        sentence_embedding = self.semantic_model.encode(text)
        
        # Add topic scores
        topic_scores = self._compute_topic_scores(text)
        
        return {
            'sentence_embedding': sentence_embedding,
            'topic_scores': topic_scores
        }
    
    def extract_structural_features(self, text):
        """Extract syntactic and structural features using spaCy"""
        doc = self.nlp(text)
        
        # Get POS tags and dependency relations
        pos_features = [token.pos_ for token in doc]
        dep_features = [token.dep_ for token in doc]
        
        # Compute structural metrics
        complexity_score = self._compute_complexity(doc)
        
        return {
            'pos_features': pos_features,
            'dep_features': dep_features,
            'complexity': complexity_score
        }
    
    def _compute_ngrams(self, text, n=3):
        """Helper function to compute n-gram statistics"""
        tokens = text.split()
        ngrams = []
        for i in range(len(tokens)-n+1):
            ngrams.append(' '.join(tokens[i:i+n]))
        return Counter(ngrams)
    
    def _compute_topic_scores(self, text):
        """Helper function for topic modeling"""
        # Implement topic modeling logic here
        pass
    
    def _compute_complexity(self, doc):
        """Helper function to compute structural complexity metrics"""
        # Implement complexity metrics here
        pass
    
    def get_features(self, text):
        """Main interface to get all requested feature types"""
        features = {}
        
        if 'token' in self.feature_type:
            features.update(self.extract_token_features(text))
        if 'semantic' in self.feature_type:
            features.update(self.extract_semantic_features(text))
        if 'structural' in self.feature_type:
            features.update(self.extract_structural_features(text))
            
        return features


'''
Old Reward Model Structure

class RewardModel(nn.Module):
    def __init__(self, checkpoint_path, eos_token_id):
        super().__init__()
        model = GPTNeoXForCausalLM.from_pretrained(checkpoint_path)
        self.model = model
        self.v_head = nn.Linear(model.gpt_neox.embed_in.embedding_dim, 2, bias=False)
        self.eos_token_id = eos_token_id
    def forward(self, input_ids):
        returns = self.model(input_ids, output_hidden_states=True).hidden_states[-1][:, -1, :]
        returns_2 = self.v_head(returns)
        return returns_2

'''

# GPU optimized RM
class RewardModel(nn.Module):
    def __init__(self, checkpoint_path, eos_token_id):
        super().__init__()
        # Load model with CPU offloading for large models
        model = GPTNeoXForCausalLM.from_pretrained(
            checkpoint_path,
            device_map='auto',  # Automatically handle device placement
            torch_dtype=torch.float16  # Use half precision
        )
        self.model = model
        self.v_head = nn.Linear(model.gpt_neox.embed_in.embedding_dim, 2, bias=False)
        self.eos_token_id = eos_token_id
        
    @torch.cuda.amp.autocast()  # Enable automatic mixed precision
    def forward(self, input_ids):
        with torch.no_grad():  # Disable gradient computation when not training
            hidden_states = self.model(input_ids, output_hidden_states=True).hidden_states[-1]
            last_hidden = hidden_states[:, -1, :]
            del hidden_states  # Explicitly free memory
            torch.cuda.empty_cache()
            returns = self.v_head(last_hidden)
            return returns




def get_dataset_train(policy_name = "skrishna/pythia-70m-non-toxic"):
    """
       policy_name =  ["skrishna/pythia-70m-non-toxic", "EleutherAI/pythia-70m", "random"]
    """
    if policy_name == "skrishna/pythia-70m-non-toxic":
        dataset_samples = pickle.load(open("datasets/non_toxic_train.pkl", "rb"))
        return dataset_samples
    elif policy_name == "EleutherAI/pythia-70m":
        dataset_samples = pickle.load(open("datasets/toxic_train.pkl", "rb"))
        return dataset_samples
    pass

def get_initial_model(learn_rm):
    """
    Returns initial reward model. 
    """
    reward_tokenizer = AutoTokenizer.from_pretrained(learn_rm)
    reward_model = RewardModel(learn_rm, reward_tokenizer.eos_token_id)
    return reward_model.requires_grad_(), reward_tokenizer

def load_saved_model(checkpoint_path, learn_rm):
    """
    Loads a saved model state from the given checkpoint path.
    """
    reward_tokenizer = AutoTokenizer.from_pretrained(learn_rm)
    reward_model = RewardModel(learn_rm, reward_tokenizer.eos_token_id)
    reward_model.load_state_dict(torch.load(checkpoint_path))
    reward_model.to(device)
    return reward_model.requires_grad_(), reward_tokenizer

def get_true_reward_model(true_rm = "s-nlp/roberta_toxicity_classifier"):
    """
    Returns true reward model. 
    """
    if true_rm == "s-nlp/roberta_toxicity_classifier":
        # load tokenizer and model weights
        reward_tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        reward_model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier').to(device)

        return reward_model, reward_tokenizer

def get_irl_loss(loss_type = "max_margin"):
    """
    Returns loss function for optimization. 
    """
    if loss_type == "max_margin":
        def loss(reward_n_t, reward_t):
            def custom_operator(input_vector):
                # Create masks for positive and negative values
                positive_mask = input_vector > 0
                negative_mask = input_vector < 0
                # Multiply positive values by -1 and negative values by -2
                input_vector[positive_mask] *= -1
                input_vector[negative_mask] *= -2
                return input_vector
            reward_diff = reward_n_t - reward_t
            reward_diff_dir = custom_operator(reward_diff)
            return reward_diff_dir
    elif loss_type == "squared":
        def loss(reward_n_t: torch.Tensor, reward_t: torch.Tensor) -> torch.Tensor:
            reward_diff = reward_n_t - reward_t
            reward_diff_sq = torch.square(reward_diff)
            return reward_diff_sq
    elif loss_type == "logistic":
        def loss(reward_n_t: torch.Tensor, reward_t: torch.Tensor) -> torch.Tensor:
            reward_diff = reward_n_t - reward_t
            reward_diff_sigmoid = torch.sigmoid(reward_diff)
            reward_diff_logistic = -torch.log(reward_diff_sigmoid)
            return reward_diff_logistic
    elif loss_type == "hinge":
        def loss(reward_n_t: torch.Tensor, reward_t: torch.Tensor) -> torch.Tensor:
            reward_diff = reward_n_t - reward_t
            reward_diff_hinge = torch.nn.functional.relu(1 - reward_diff)
            return reward_diff_hinge
    
    return loss

def get_optimizer(model, optim_type = "adam", lr = 0.00001, momentum = 0.9):
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
    # Ensure both x and y are numpy arrays for element-wise operations
    x = np.array(x)
    y = np.array(y)
    
    # Calculate the lp norm
    diff = np.abs(x - y)
    lp_norm_value = np.sum(diff**p)**(1/p)
    return lp_norm_value

def calculate_accuracy_f1(true_rewards, learned_rewards):
    true_rewards_labels = [0 if reward > 0 else 1 for reward in true_rewards]
    learned_rewards_labels = [0 if reward > 0 else 1 for reward in learned_rewards]
    return accuracy_score(true_rewards_labels, learned_rewards_labels), f1_score(true_rewards_labels, learned_rewards_labels)

# get evaluation with rm and gt labels
def get_evaluation(samples_data, true_rm, learned_rm, batch_size=32):
    """
    Returns all metrics comparing against both reward model predictions and ground truth labels.
    
    Args:
        samples_data: Dictionary containing:
            - 'samples': List of text samples to evaluate
            - 'ground_truth_labels': List of corresponding ground truth labels (1 for toxic, 0 for non-toxic)
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
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = reward_model(input_ids.input_ids)
            output = -1 * output[:, -1]
            all_outputs.append(output.cpu())
            
        # Clear memory
        del input_ids
        torch.cuda.empty_cache()
    
    return torch.cat(all_outputs, dim=0)

def data_loader(list1, list2, batch_size=32):  # Increased from 2 to 32
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
        
        # Process all samples in the batch at once
        if isinstance(batch1[0], str):  # If dealing with raw text
            yield batch1, batch2
        else:  # If dealing with tokenized data
            yield [row[-1] for row in batch1], [row[-1] for row in batch2]


# # Add to irl_utilities.py
# def analyze_feature_contributions(model, dataset, feature_config):
#     """Analyze how different features contribute to reward predictions"""
#     feature_importance = defaultdict(list)
    
#     for sample in dataset:
#         # Get base prediction
#         base_pred = model(sample)
        
#         # Analyze each feature type
#         features = model.feature_extractor.get_features(sample)
#         for feature_type in feature_config['type']:
#             if feature_type in features:
#                 # Zero out this feature type
#                 modified_features = features.copy()
#                 modified_features[feature_type] *= 0
                
#                 # Get prediction without this feature
#                 mod_pred = model.forward_with_features(modified_features)
                
#                 # Calculate importance as prediction difference
#                 importance = torch.norm(base_pred - mod_pred)
#                 feature_importance[feature_type].append(importance.item())
    
#     return feature_importance