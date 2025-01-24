import datasets
import json
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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # Set up save directories
    run_dir, models_dir, metrics_dir = setup_save_directories(cfg)
    
    # Save config
    save_config(cfg, run_dir)
    
    # processing the config
    cfg.experiment.learn_rm = cfg.experiment.learn_rm + cfg.experiment.learn_rm_size
    cfg.experiment.learn_rm_path = cfg.experiment.learn_rm_path + cfg.experiment.learn_rm.replace("/", "-") + "_" + cfg.experiment.true_rm.replace("/", "-") + f"lr_{cfg.experiment.training.lr}" + f"_ss_{cfg.experiment.training.sample_size}"
    
    cfg.experiment.candidate_policies[-1] = cfg.experiment.learn_rm
    cfg.experiment.non_toxic_rm = cfg.experiment.learn_rm
    print("Config: " + OmegaConf.to_yaml(cfg))

    # Create the IRL training dataset
    if cfg.experiment.create_dataset == 'true':
        datasets_irl = generate_irl_demonstrations(cfg.experiment.evaluation_dataset, cfg.experiment.training.sample_size, cfg.experiment.candidate_policies, cfg.experiment.learn_rm_size)
    dataset_ntoxic = datasets.load_dataset(cfg.experiment.candidate_policy_paths[0])
    data_set_ntoxic = [sample["output"] for sample in dataset_ntoxic["train"]]
    dataset_toxic = datasets.load_dataset(cfg.experiment.candidate_policy_paths[1])
    data_set_toxic = [sample["output"] for sample in dataset_toxic["train"]]
    test_samples = [sample["output"] for sample in dataset_toxic["train"]] + [sample["output"] for sample in dataset_ntoxic["train"]]

    # Initialization
    if cfg.experiment.from_checkpoint != False:
        init_model, tokenizer = load_saved_model(f"src/{cfg.experiment.from_checkpoint}", cfg.experiment.learn_rm)
    else:
        init_model, tokenizer = get_initial_model(cfg.experiment.learn_rm)

    loss = get_irl_loss(cfg.experiment.training.loss_type)
    optimizer = get_optimizer(init_model, cfg.experiment.training.optimizer, cfg.experiment.training.lr)

    # IRL Loop
    epoch_losses = []
    correlations = []
    accuracies = []
    f1s = []

    for epoch in range(cfg.experiment.training.n_epochs):
        epoch_loss = []
        print(f"Processing Epoch {epoch}..")
        
        # Get evaluation metrics
        evals = get_evaluation(test_samples, cfg.experiment.true_rm, [init_model, tokenizer])
        
        # Store metrics
        metrics = {
            'epoch': epoch,
            'correlation': evals['pearson_correlation'],
            'accuracy': evals['accuracy'],
            'f1': evals['f1'],
            'euclidean_distance': evals['euclidean_distance'],
            'kendall_tau': evals['kendall_tau'],
            'spearman': evals['spearman'],
            'cosine_similarity': evals['cosine_similarity']
        }
        
        # Training loop
        for sample_n_toxic, sample_toxic in data_loader(data_set_ntoxic, data_set_toxic):
            policy_outputs_nt = get_reward_score(init_model, sample_n_toxic, tokenizer)
            policy_outputs_t = get_reward_score(init_model, sample_toxic, tokenizer)
            loss_value = loss(policy_outputs_nt, policy_outputs_t)
            epoch_loss.append(torch.mean(loss_value).item())
            loss_value.backward()
            optimizer.step()
        
        # Calculate average loss
        average_loss = sum(epoch_loss) / len(epoch_loss)
        metrics['average_loss'] = average_loss
        print(f"Epoch {epoch} Average Loss: {average_loss}")
        
        # Save model and metrics for this epoch
        save_model_and_metrics(init_model, metrics, epoch, models_dir, metrics_dir)
        
        # Store metrics for plotting
        epoch_losses.append(average_loss)
        correlations.append(metrics['correlation'])
        accuracies.append(metrics['accuracy'])
        f1s.append(metrics['f1'])

    # Create and save final plots
    plot_metrics(epoch_losses, correlations, accuracies, f1s, run_dir)
    
def plot_metrics(losses, correlations, accuracies, f1s, run_dir):
    """Create and save plots of training metrics."""
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'loss_plot.png'))
    plt.close()

    # Plot other metrics
    plt.figure(figsize=(10, 6))
    plt.plot(correlations, label='Correlation')
    plt.plot(accuracies, label='Accuracy')
    plt.plot(f1s, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Metrics Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'metrics_plot.png'))
    plt.close()

if __name__ == "__main__":
    main()
