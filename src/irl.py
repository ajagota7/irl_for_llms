import datasets
import torch
from torch import nn
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from irl_utilities import get_initial_model, get_irl_loss, get_optimizer, get_evaluation, get_reward_score, data_loader, load_saved_model
from create_dataset_irl import generate_irl_demonstrations
from save_utils import setup_save_directories, save_model_and_metrics, save_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METRICS = [
    'accuracy',
    'pearson_correlation',
    'kendall_tau',
    'spearman',
    'f1',
    'euclidean_distance',
    'cosine_similarity'
]

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # Debug prints
    print("\nDEBUG CONFIG:")
    if hasattr(cfg.experiment, 'experiments_base_dir'):
        print(f"Base dir from config: {cfg.experiment.experiments_base_dir}")
    print(f"Environment variable: {os.getenv('EXPERIMENT_DIR')}")
    
    # Set up save directories with configurable base dir
    experiments_base_dir = os.getenv('EXPERIMENT_DIR', "/content/drive/MyDrive/irl_experiments")
    print(f"Using base dir: {experiments_base_dir}")
    
    run_dir, models_dir, metrics_dir = setup_save_directories(cfg, experiments_base_dir)
    save_config(cfg, run_dir)
    
    # Processing the config
    cfg.experiment.learn_rm = cfg.experiment.learn_rm + cfg.experiment.learn_rm_size
    cfg.experiment.learn_rm_path = cfg.experiment.learn_rm_path + cfg.experiment.learn_rm.replace("/", "-") + "_" + \
                                  cfg.experiment.true_rm.replace("/", "-") + f"lr_{cfg.experiment.training.lr}" + \
                                  f"_ss_{cfg.experiment.training.sample_size}"
    
    cfg.experiment.candidate_policies[-1] = cfg.experiment.learn_rm
    cfg.experiment.non_toxic_rm = cfg.experiment.learn_rm
    print("Config: " + OmegaConf.to_yaml(cfg))

    # Create the IRL training dataset if needed
    if cfg.experiment.create_dataset == 'true':
        datasets_irl = generate_irl_demonstrations(
            cfg.experiment.evaluation_dataset,
            cfg.experiment.training.sample_size,
            cfg.experiment.candidate_policies,
            cfg.experiment.learn_rm_size
        )

    # Load and prepare training data with labels
    print("\nPreparing training dataset...")
    dataset = datasets.load_dataset(cfg.experiment.evaluation_dataset)
    train_data = {
        'samples': [sample["original_output"] for sample in dataset["train"]],
        'ground_truth_labels': [1 if sample["toxic"] else 0 for sample in dataset["train"]]
    }
    
    # Load and prepare test data with labels
    print("Preparing test dataset...")
    dataset_toxic = datasets.load_dataset(cfg.experiment.candidate_policy_paths[1])
    dataset_ntoxic = datasets.load_dataset(cfg.experiment.candidate_policy_paths[0])
    data_set_toxic = [sample["output"] for sample in dataset_toxic["train"]]
    data_set_ntoxic = [sample["output"] for sample in dataset_ntoxic["train"]]
    
    test_data = {
        'samples': data_set_toxic + data_set_ntoxic,
        'ground_truth_labels': [1] * len(data_set_toxic) + [0] * len(data_set_ntoxic)
    }

    # Model Initialization
    if cfg.experiment.from_checkpoint != False:
        init_model, tokenizer = load_saved_model(f"src/{cfg.experiment.from_checkpoint}", cfg.experiment.learn_rm)
    else:
        init_model, tokenizer = get_initial_model(cfg.experiment.learn_rm)

    loss = get_irl_loss(cfg.experiment.training.loss_type)
    optimizer = get_optimizer(init_model, cfg.experiment.training.optimizer, cfg.experiment.training.lr)

    # IRL Loop
    epoch_losses = []
    train_metrics_history = []
    test_metrics_history = []

    for epoch in range(cfg.experiment.training.n_epochs):
        epoch_loss = []
        print(f"\nProcessing Epoch {epoch}..")
        
        # Get evaluation metrics for both train and test
        print("Evaluating on training set...")
        train_evals = get_evaluation(train_data, cfg.experiment.true_rm, [init_model, tokenizer])
        print("Evaluating on test set...")
        test_evals = get_evaluation(test_data, cfg.experiment.true_rm, [init_model, tokenizer])
        
        train_metrics_history.append(train_evals)
        test_metrics_history.append(test_evals)
        
        # Training loop
        print("Starting training iteration...")
        for sample_n_toxic, sample_toxic in data_loader(data_set_ntoxic, data_set_toxic):
            optimizer.zero_grad()
            
            # Forward pass
            policy_outputs_nt = get_reward_score(init_model, sample_n_toxic, tokenizer)
            policy_outputs_t = get_reward_score(init_model, sample_toxic, tokenizer)
            loss_value = loss(policy_outputs_nt, policy_outputs_t)

            # Backward pass
            loss_value.backward()
            optimizer.step()
            
            epoch_loss.append(loss_value.item())
        
        # Calculate average loss
        average_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_losses.append(average_loss)
        
        # Save model and all metrics for this epoch
        combined_metrics = {
            'epoch': epoch,
            'average_loss': average_loss,
            'train_metrics': train_evals,
            'test_metrics': test_evals
        }
        
        save_model_and_metrics(init_model, combined_metrics, epoch, models_dir, metrics_dir)
        
        # Print comprehensive metrics for this epoch
        print(f"\nEpoch {epoch} Complete - Summary:")
        print(f"Training Loss: {average_loss:.4f}")
        
        print("\nTraining Metrics:")
        for metric in METRICS:
            rm_key = f"{metric}_rm"
            gt_key = f"{metric}_gt"
            print(f"  {metric:20} - RM: {train_evals[rm_key]:.4f}, GT: {train_evals[gt_key]:.4f}")
            
        print("\nTest Metrics:")
        for metric in METRICS:
            rm_key = f"{metric}_rm"
            gt_key = f"{metric}_gt"
            print(f"  {metric:20} - RM: {test_evals[rm_key]:.4f}, GT: {test_evals[gt_key]:.4f}")
            
        # Clear memory at end of epoch
        torch.cuda.empty_cache()

    # Save final plots
    plot_metrics(epoch_losses, train_metrics_history, test_metrics_history, run_dir)

def plot_metrics(losses, train_metrics, test_metrics, run_dir):
    """Create and save plots of training metrics."""
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'loss_plot.png'))
    plt.close()

    # Plot each metric comparing train vs test for both RM and GT
    for metric in METRICS:
        plt.figure(figsize=(12, 6))
        
        rm_key = f"{metric}_rm"
        gt_key = f"{metric}_gt"
        
        # Get values
        train_rm_values = [m[rm_key] for m in train_metrics]
        test_rm_values = [m[rm_key] for m in test_metrics]
        train_gt_values = [m[gt_key] for m in train_metrics]
        test_gt_values = [m[gt_key] for m in test_metrics]
        
        # Plot RM values
        plt.plot(train_rm_values, label=f'Train (RM)', linestyle='-', color='blue')
        plt.plot(test_rm_values, label=f'Test (RM)', linestyle='--', color='blue')
        
        # Plot GT values
        plt.plot(train_gt_values, label=f'Train (GT)', linestyle='-', color='red')
        plt.plot(test_gt_values, label=f'Test (GT)', linestyle='--', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} Over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'{metric}_plot.png'))
        plt.close()

if __name__ == "__main__":
    main()
