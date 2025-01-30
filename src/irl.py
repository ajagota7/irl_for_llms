# import datasets
# import json
# import torch
# from torch import nn
# import matplotlib.pyplot as plt
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import os

# from torch.cuda.amp import autocast, GradScaler
# from irl_utilities import get_initial_model, get_irl_loss, get_optimizer, get_evaluation, get_reward_score, data_loader, load_saved_model
# from create_dataset_irl import generate_irl_demonstrations
# from save_utils import setup_save_directories, save_model_and_metrics, save_config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(cfg : DictConfig) -> None:
#     # Debug prints
#     print("\nDEBUG CONFIG:")
#     if hasattr(cfg.experiment, 'experiments_base_dir'):
#         print(f"Base dir from config: {cfg.experiment.experiments_base_dir}")
#     print(f"Environment variable: {os.getenv('EXPERIMENT_DIR')}")
    
#     # Set up save directories with configurable base dir
#     experiments_base_dir = os.getenv('EXPERIMENT_DIR', "/content/drive/MyDrive/irl_experiments")
#     print(f"Using base dir: {experiments_base_dir}")
    
#     run_dir, models_dir, metrics_dir = setup_save_directories(cfg, experiments_base_dir)
    
#     # Save config
#     save_config(cfg, run_dir)
    
#     # processing the config
#     cfg.experiment.learn_rm = cfg.experiment.learn_rm + cfg.experiment.learn_rm_size
#     cfg.experiment.learn_rm_path = cfg.experiment.learn_rm_path + cfg.experiment.learn_rm.replace("/", "-") + "_" + cfg.experiment.true_rm.replace("/", "-") + f"lr_{cfg.experiment.training.lr}" + f"_ss_{cfg.experiment.training.sample_size}"
    
#     cfg.experiment.candidate_policies[-1] = cfg.experiment.learn_rm
#     cfg.experiment.non_toxic_rm = cfg.experiment.learn_rm
#     print("Config: " + OmegaConf.to_yaml(cfg))

#     # Create the IRL training dataset
#     if cfg.experiment.create_dataset == 'true':
#         datasets_irl = generate_irl_demonstrations(cfg.experiment.evaluation_dataset, cfg.experiment.training.sample_size, cfg.experiment.candidate_policies, cfg.experiment.learn_rm_size)
#     dataset_ntoxic = datasets.load_dataset(cfg.experiment.candidate_policy_paths[0])
#     data_set_ntoxic = [sample["output"] for sample in dataset_ntoxic["train"]]
#     dataset_toxic = datasets.load_dataset(cfg.experiment.candidate_policy_paths[1])
#     data_set_toxic = [sample["output"] for sample in dataset_toxic["train"]]
#     test_samples = [sample["output"] for sample in dataset_toxic["train"]] + [sample["output"] for sample in dataset_ntoxic["train"]]

#     # Initialization
#     if cfg.experiment.from_checkpoint != False:
#         init_model, tokenizer = load_saved_model(f"src/{cfg.experiment.from_checkpoint}", cfg.experiment.learn_rm)
#     else:
#         init_model, tokenizer = get_initial_model(cfg.experiment.learn_rm)

#     loss = get_irl_loss(cfg.experiment.training.loss_type)
#     optimizer = get_optimizer(init_model, cfg.experiment.training.optimizer, cfg.experiment.training.lr)

#     # IRL Loop
#     epoch_losses = []
#     correlations = []
#     accuracies = []
#     f1s = []

#     # Add gradient scaler for mixed precision training
#     scaler = GradScaler()

#     for epoch in range(cfg.experiment.training.n_epochs):
#         epoch_loss = []
#         print(f"Processing Epoch {epoch}..")
        
#         # Get evaluation metrics
#         evals = get_evaluation(test_samples, cfg.experiment.true_rm, [init_model, tokenizer])
        
#         # Store metrics
#         metrics = {
#             'epoch': epoch,
#             'correlation': evals['pearson_correlation'],
#             'accuracy': evals['accuracy'],
#             'f1': evals['f1'],
#             'euclidean_distance': evals['euclidean_distance'],
#             'kendall_tau': evals['kendall_tau'],
#             'spearman': evals['spearman'],
#             'cosine_similarity': evals['cosine_similarity']
#         }
        
#         # Training loop
#         for sample_n_toxic, sample_toxic in data_loader(data_set_ntoxic, data_set_toxic):
#             policy_outputs_nt = get_reward_score(init_model, sample_n_toxic, tokenizer)
#             policy_outputs_t = get_reward_score(init_model, sample_toxic, tokenizer)
#             loss_value = loss(policy_outputs_nt, policy_outputs_t)
#             epoch_loss.append(torch.mean(loss_value).item())
#             loss_value.backward()
#             optimizer.step()
        
#         # Calculate average loss
#         average_loss = sum(epoch_loss) / len(epoch_loss)
#         metrics['average_loss'] = average_loss
#         print(f"Epoch {epoch} Average Loss: {average_loss}")
        
#         # Save model and metrics for this epoch
#         save_model_and_metrics(init_model, metrics, epoch, models_dir, metrics_dir)
        
#         # Store metrics for plotting
#         epoch_losses.append(average_loss)
#         correlations.append(metrics['correlation'])
#         accuracies.append(metrics['accuracy'])
#         f1s.append(metrics['f1'])

#     # Create and save final plots
#     plot_metrics(epoch_losses, correlations, accuracies, f1s, run_dir)
    
# def plot_metrics(losses, correlations, accuracies, f1s, run_dir):
#     """Create and save plots of training metrics."""
#     # Plot losses
#     plt.figure(figsize=(10, 6))
#     plt.plot(losses, label='Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Average Loss')
#     plt.title('Training Loss Over Epochs')
#     plt.legend()
#     plt.savefig(os.path.join(run_dir, 'loss_plot.png'))
#     plt.close()

#     # Plot other metrics
#     plt.figure(figsize=(10, 6))
#     plt.plot(correlations, label='Correlation')
#     plt.plot(accuracies, label='Accuracy')
#     plt.plot(f1s, label='F1 Score')
#     plt.xlabel('Epoch')
#     plt.ylabel('Score')
#     plt.title('Metrics Over Epochs')
#     plt.legend()
#     plt.savefig(os.path.join(run_dir, 'metrics_plot.png'))
#     plt.close()

# if __name__ == "__main__":
#     main()



import datasets
import torch
from torch import nn
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from torch.cuda.amp import autocast, GradScaler

from irl_utilities import get_initial_model, get_irl_loss, get_optimizer, get_evaluation, get_reward_score, data_loader, load_saved_model
from create_dataset_irl import generate_irl_demonstrations
from save_utils import setup_save_directories, save_model_and_metrics, save_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define all metrics we want to track
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
    
    # Set up directories and save config
    run_dir, models_dir, metrics_dir = setup_save_directories(cfg, experiments_base_dir)
    save_config(cfg, run_dir)
    
    # Processing the config
    cfg.experiment.learn_rm = cfg.experiment.learn_rm + cfg.experiment.learn_rm_size
    cfg.experiment.learn_rm_path = cfg.experiment.learn_rm_path + cfg.experiment.learn_rm.replace("/", "-") + "_" + cfg.experiment.true_rm.replace("/", "-") + f"lr_{cfg.experiment.training.lr}" + f"_ss_{cfg.experiment.training.sample_size}"
    
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

    # Load training datasets
    dataset = datasets.load_dataset(cfg.experiment.evaluation_dataset)
    train_samples = [sample["original_output"] for sample in dataset["train"]]
    
    # Load test datasets from candidate policies
    dataset_ntoxic = datasets.load_dataset(cfg.experiment.candidate_policy_paths[0])
    data_set_ntoxic = [sample["output"] for sample in dataset_ntoxic["train"]]
    dataset_toxic = datasets.load_dataset(cfg.experiment.candidate_policy_paths[1])
    data_set_toxic = [sample["output"] for sample in dataset_toxic["train"]]
    test_samples = data_set_toxic + data_set_ntoxic

    # Initialization
    if cfg.experiment.from_checkpoint != False:
        init_model, tokenizer = load_saved_model(f"src/{cfg.experiment.from_checkpoint}", cfg.experiment.learn_rm)
    else:
        init_model, tokenizer = get_initial_model(cfg.experiment.learn_rm)

    loss = get_irl_loss(cfg.experiment.training.loss_type)
    optimizer = get_optimizer(init_model, cfg.experiment.training.optimizer, cfg.experiment.training.lr)
    scaler = GradScaler()  # For mixed precision training

    # IRL Loop
    epoch_losses = []
    train_metrics_history = []
    test_metrics_history = []

    for epoch in range(cfg.experiment.training.n_epochs):
        epoch_loss = []
        print(f"\nProcessing Epoch {epoch}..")
        
        # Get evaluation metrics for both train and test
        print("Evaluating on training set...")
        train_evals = get_evaluation(train_samples, cfg.experiment.true_rm, [init_model, tokenizer])
        print("Evaluating on test set...")
        test_evals = get_evaluation(test_samples, cfg.experiment.true_rm, [init_model, tokenizer])
        
        # Store metrics
        train_metrics = {
            'epoch': epoch,
            'set': 'train',
            **train_evals
        }
        
        test_metrics = {
            'epoch': epoch,
            'set': 'test',
            **test_evals
        }
        
        train_metrics_history.append(train_metrics)
        test_metrics_history.append(test_metrics)
        
        # Training loop with mixed precision
        print("Starting training iteration...")
        for sample_n_toxic, sample_toxic in data_loader(data_set_ntoxic, data_set_toxic):
            optimizer.zero_grad()
            
            with autocast():
                policy_outputs_nt = get_reward_score(init_model, sample_n_toxic, tokenizer)
                policy_outputs_t = get_reward_score(init_model, sample_toxic, tokenizer)
                loss_value = loss(policy_outputs_nt, policy_outputs_t)
            
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss.append(torch.mean(loss_value).item())
        
        # Calculate average loss
        average_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_losses.append(average_loss)
        
        # Save model and all metrics for this epoch
        combined_metrics = {
            'epoch': epoch,
            'average_loss': average_loss,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        save_model_and_metrics(init_model, combined_metrics, epoch, models_dir, metrics_dir)
        
        # Print comprehensive metrics for this epoch
        print(f"\nEpoch {epoch} Complete - Summary:")
        print(f"Training Loss: {average_loss:.4f}")
        
        print("\nTraining Metrics:")
        for metric in METRICS:
            print(f"  {metric}: {train_metrics[metric]:.4f}")
            
        print("\nTest Metrics:")
        for metric in METRICS:
            print(f"  {metric}: {test_metrics[metric]:.4f}")
            
        # Clear memory at end of epoch
        torch.cuda.empty_cache()

    # Save final plots
    plot_metrics(epoch_losses, train_metrics_history, test_metrics_history, run_dir)

def plot_metrics(losses, train_metrics, test_metrics, run_dir):
    """Create and save plots of training metrics."""
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'plots', 'loss_plot.png'))
    plt.close()

    # Plot each metric comparing train vs test
    for metric in METRICS:
        plt.figure(figsize=(10, 6))
        
        train_values = [m[metric] for m in train_metrics]
        test_values = [m[metric] for m in test_metrics]
        
        plt.plot(train_values, label=f'Train {metric}')
        plt.plot(test_values, label=f'Test {metric}')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(run_dir, 'plots', f'{metric}_plot.png'))
        plt.close()

if __name__ == "__main__":
    main()