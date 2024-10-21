from transformers import AutoTokenizer, GPTNeoXForCausalLM, RobertaTokenizer, RobertaForSequenceClassification
import datasets
import torch
from torch import nn
import pickle
import logging
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RewardModel(nn.Module):
    def __init__(self, checkpoint_path, eos_token_id):
        super().__init__()
        # model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        model = GPTNeoXForCausalLM.from_pretrained(checkpoint_path)
        self.model = model
        self.v_head = nn.Linear(model.gpt_neox.embed_in.embedding_dim, 2, bias=False)  # TODO make not magic number
        self.eos_token_id = eos_token_id
    def forward(self, input_ids):
        returns = self.model(input_ids, output_hidden_states=True).hidden_states[-1][:, -1, :]#[0] #, 1)
        # print("Returns : ", returns.shape, returns)
        returns_2 = self.v_head(returns)
        # Applying softmax to returns_2 along the last dimension
        # returns_2 = torch.nn.functional.softmax(returns_2, dim=-1).squeeze(-1)
        # print("Returns : ", returns.shape, returns_2)
        return returns_2
    
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
    # if learn_rm == "EleutherAI/pythia-70m":
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

def get_evaluation(test_data, true_rm , learned_rm, binary=False):
    """
    Returns correlation between true and learned reward models. 
    """
    true_rm, true_rm_tokenizer = get_true_reward_model(true_rm)
    learned_rm, learned_rm_tokenizer = learned_rm[0], learned_rm[1]
    true_rewards = []
    learned_rewards = []
    for sample in test_data:
        input_text = sample
        true_rewards.append(get_reward_score(true_rm, input_text, true_rm_tokenizer, true_reward = True).detach().cpu().item())
        learned_rewards.append(get_reward_score(learned_rm, input_text, learned_rm_tokenizer).detach().cpu().item())

    # convert to numpy array
    # learned_rewards = [-x for x in learned_rewards]
    true_rewards = np.array(true_rewards)
    learned_rewards = np.array(learned_rewards)

    if binary:
        # convert rewards to 0s and 1s
        true_rewards = np.where(true_rewards > 0, 0, 1)
        learned_rewards = np.where(learned_rewards > 0, 0, 1)

    # compute pearson correlation
    # print("True rewards: ", true_rewards)
    # print("Learned rewards: ", learned_rewards)    

    pearson_correlation = np.corrcoef(true_rewards, learned_rewards)[0,1]
    euc_norm = lp_norm(true_rewards, learned_rewards, 2)
    kendall_tau = stats.kendalltau(true_rewards, learned_rewards)[0]
    spearman = stats.spearmanr(true_rewards, learned_rewards)[0]
    cosine_sim = 1 - cosine(true_rewards, learned_rewards)
    mse = mean_squared_error(true_rewards, learned_rewards)
    accuracy, f1 = calculate_accuracy_f1(true_rewards, learned_rewards)
                                    
    print("Correlation between true and learned reward models: ", pearson_correlation)
    print("Euclidean distance between true and learned reward models: ", euc_norm)
    print("Kendall_tau between true and learned reward models: ", kendall_tau)
    print("Spearman between true and learned reward models: ", spearman)
    print("MSE between true and learned reward models: ", mse)
    print("Accuracy between true and learned reward models: ", accuracy)
    print("F1 Score between true and learned reward models: ", f1)

    evals = {
        'pearson_correlation': pearson_correlation,
        'euclidean_distance': euc_norm,
        'kendall_tau': kendall_tau,
        'spearman': spearman,
        'cosine_similarity': cosine_sim,
        'mean_squared_error': mse,
        'accuracy': accuracy,
        'f1': f1
    }

    return evals

def get_policy_outputs(model, sample):
    """
     -- NTBU -- 
    """
    pass

def get_reward_score(reward_model, input_text, tokenizer, true_reward=False):
    """
    Takes reward model and input and returns reward score. 
    """
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    # potential GPU/CPU mismatch fix
    reward_model = reward_model.to(device)
    
    # print("REWARD MODEL: " , reward_model(input_ids))
    if true_reward:
        output = reward_model(input_ids).logits
    else:
        output = reward_model(input_ids)

    output = -1*output[:, -1]
    return output

def data_loader(list1, list2, batch_size=2):
    assert len(list1) == len(list2), "Both lists should have the same length"
    
    for i in range(0, len(list1), batch_size):
        batch1 = [row[-1] for row in list1[i:min(i+batch_size, len(list1))]]
        batch2 = [row[-1] for row in list2[i:min(i+batch_size, len(list2))]]
        yield batch1, batch2

def get_generation(input_text, model, tokenizer):
    print(f"SAMPLE::\n {input_text}")
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    model = model.to(device)
    output = model.generate(input_ids, max_length=100, temperature=0)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"SAMPLE::\n {input_text} \n Output : {output_text}")
    return output_text



def irl_function(num_epochs=100, current_directory="./models", sub_model_name="reward_model", model_size="70M"):
    """
    Step 1 : Generate dataset with random policy 
    Step 2 : Initiate model with K million parameters
    Step 3 : Implement the loss function
    Step 4 : Evaluate reward model on toxicity.  
    """
    # Obtain dataset
    data_name = "allenai-real-toxicity-prompts"
    dataset_ntoxic = datasets.load_dataset(f"skrishna/{data_name}_{model_size}_non_toxic")
    data_set_ntoxic = [sample["output"] for sample in dataset_ntoxic["train"]]
    dataset_toxic = datasets.load_dataset(f"skrishna/{data_name}_{model_size}_toxic")
    data_set_toxic = [sample["output"] for sample in dataset_toxic["train"]]

    # Obtain training functions with hyper-params
    init_model, tokenizer = get_initial_model(model_size)
    loss = get_irl_loss()
    optimizer = get_optimizer(init_model)
    
    epoch_losses = []  # Store average loss per epoch

    # Training loop
    for num_epoch in tqdm(range(num_epochs)):
        epoch_loss = []  # Store loss for each batch
        print(f"Processing Epoch {num_epoch}..")
        for sample_n_toxic, sample_toxic in data_loader(data_set_ntoxic, data_set_toxic):
            policy_outputs_nt = get_reward_score(init_model, sample_n_toxic, tokenizer)
            policy_outputs_t = get_reward_score(init_model, sample_toxic, tokenizer)
            loss_value = loss(policy_outputs_nt, policy_outputs_t)
            epoch_loss.append(torch.mean(loss_value).item())  # Convert to Python number and store
            loss_value.backward()
            optimizer.step()
        
        # Calculate and store the average loss for this epoch
        average_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_losses.append(average_loss)
        print(f"Epoch {num_epoch} Average Loss: {average_loss}")

    # Plotting
    plt.plot(range(num_epochs), epoch_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt_path = f"training_loss_over_epochs_{data_name}_{model_size}.png"
    plt.savefig(plt_path)
    
    # save trained reward model
    current_directory_repo = current_directory + f"/{sub_model_name}"
    torch.save(init_model.state_dict(), current_directory_repo)
    
    # init_model.save_pretrained(f"skrishna/{data_name}_{model_size}_irl")
    # tokenizer.save_pretrained(f"skrishna/{data_name}_{model_size}_irl")



if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--rm", type=str, default="jaredjoss/pythia-70m-roberta-rlhf-model")
    argparse.add_argument("--pm", type=str, default="EleutherAI/pythia-70m")
    argparse.add_argument("--dataset", type=str, default="google/jigsaw_toxicity_pred")
    argparse.add_argument("--ne", type=int, default=10)
    argparse.add_argument("--sfx", type=str, default="irl")
    args = argparse.parse_args()
    # train_dataset = get_dataset_train(args.rm)
    irl_function(num_epochs=args.ne)
    #irl_function(args)
