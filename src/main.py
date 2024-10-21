import datasets
import json
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from irl_utilities import get_initial_model, get_irl_loss, get_optimizer, get_evaluation, get_reward_score, data_loader, load_saved_model
from create_dataset_irl import generate_irl_demonstrations

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
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
    correlation = 0
    accuracy = 0

    for epoch in range(cfg.experiment.training.n_epochs):
        epoch_loss = []  # Store loss for each batch
        print(f"Processing Epoch {epoch}..")
        evals = get_evaluation(test_samples, cfg.experiment.true_rm, [init_model, tokenizer])
        correlations.append(evals['pearson_correlation'])
        accuracies.append(evals['accuracy'])
        f1s.append(evals['f1'])
        accuracy = evals['accuracy']
        correlation = evals['pearson_correlation']
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
        print(f"Epoch {epoch} Average Loss: {average_loss}")

        if cfg.experiment.correlation_threshold != False and correlation > cfg.experiment.correlation_threshold:
            torch.save(init_model.state_dict(), f"src/{cfg.experiment.learn_rm_path}_{epoch}_epochs_oriCorr_{correlation:.2f}_acc_{accuracy}.pt")

    # save trained reward model and results
    data_plt_name = cfg.experiment.data_name.replace("/", "-")
    torch.save(init_model.state_dict(), f"src/{cfg.experiment.learn_rm_path}_{epoch}_epochs_oriCorr_{correlation:.2f}_acc_{accuracy}.pt")
    results = {'epoch_losses': epoch_losses, 'correlations': correlations, 'accuracies': accuracies, 'f1s': f1s}
    with open(f"src/results/results_{cfg.experiment.training.n_epochs}_epochs_lr_{cfg.experiment.training.lr}_{data_plt_name}_{cfg.experiment.learn_rm_size}_ss_{cfg.experiment.training.sample_size}_oriCorr_{correlation:.2f}_acc_{accuracy}.json", 'w') as f:
        json.dump(results, f)
    
    # Plotting
    plt.plot(range(cfg.experiment.training.n_epochs), epoch_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt_path = f"src/plots/training_loss_over_{cfg.experiment.training.n_epochs}_epochs_lr_{cfg.experiment.training.lr}_{data_plt_name}_{cfg.experiment.learn_rm_size}.png"
    plt.savefig(plt_path)
    plt.close()

    plt.plot(range(cfg.experiment.training.n_epochs), correlations, label='Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.title('Correlation Over Epochs')
    plt.legend()
    plt_path = f"src/plots/correlation_over_{cfg.experiment.training.n_epochs}_epochs_lr_{cfg.experiment.training.lr}_{data_plt_name}_{cfg.experiment.learn_rm_size}.png"
    plt.savefig(plt_path)

    
if __name__ == "__main__":
    main()