import torch
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from datasets import Dataset, DatasetDict
from huggingface_hub import notebook_login, HfFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_generation(input_text, model, tokenizer):
    print(f"SAMPLE::\n {input_text}")
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    model = model.to(device)
    output = model.generate(input_ids, max_length=100, temperature=0)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"SAMPLE::\n {input_text} \n Output : {output_text}")
    return output_text

def upload_dataset_to_hf_hub(train_data, test_data, repo_name, private=False):
    """
    Uploads a list of dictionaries as a dataset to the Hugging Face Hub.

    Parameters:
    - data (list of dict): The dataset to upload, where each item in the list is a dictionary representing a data point.
    - repo_name (str): The name of the repository on the Hugging Face Hub where the dataset will be uploaded.
    - private (bool, optional): If True, the dataset will be private. Defaults to False.
    """
    train_data = {key: [dic[key] for dic in train_data] for key in train_data[0]}
    test_data = {key: [dic[key] for dic in test_data] for key in test_data[0]}
    
    # Convert list of dictionaries to a Hugging Face Dataset
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)
    
    # Convert to a DatasetDict if needed (for example, for train/test splits)
    # For simplicity, assuming the entire dataset is treated as 'train'
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # Prompt for Hugging Face authentication if not already logged in
    if not HfFolder.get_token():
        print("You need to be logged into the Hugging Face Hub.")
        notebook_login()
    else:
        print("Already logged in to the Hugging Face Hub.")
    
    # Push to the Hub
    dataset_dict.push_to_hub(repo_name, private=private)
    print(f"Dataset uploaded to Hugging Face Hub successfully: https://huggingface.co/datasets/{repo_name}")

def generate_irl_demonstrations(dataset_name, num_samples_train, model_checkpoints, model_size, debug=True):
    dataset_prompts = datasets.load_dataset(dataset_name)
    num_samples_train = num_samples_train
    num_samples = 2 * num_samples_train
    if debug:
        num_samples_train = 100
        num_samples = 150
    data_name = dataset_name.replace("/", "-")
    dataset_destinations = [f"skrishna/{data_name}_{model_size}_non_toxic", f"skrishna/{data_name}_{model_size}_toxic"]
    for ind, model_chkp in enumerate(model_checkpoints):
        if model_chkp != "random":
            model = GPTNeoXForCausalLM.from_pretrained(model_chkp)
            tokenizer = AutoTokenizer.from_pretrained(model_chkp)
            train_samples = []
            for sample_id in tqdm(range(num_samples_train)):
                input_sample = dataset_prompts["train"][sample_id]["prompt"]
                output_sample = get_generation(input_sample, model, tokenizer)
                train_samples.append({"prompt" : input_sample, "output": output_sample})
            test_samples = []
            for sample_id in tqdm(range(num_samples_train, num_samples)):
                input_sample = dataset_prompts["train"][sample_id]["prompt"]
                output_sample = get_generation(input_sample, model, tokenizer)
                test_samples.append({"prompt" : input_sample, "output": output_sample})
            upload_dataset_to_hf_hub(train_samples, test_samples, dataset_destinations[ind])
    return dataset_destinations        
    pass

if __name__ == "__main__":
    dataset_toxicity = datasets.load_dataset("jaredjoss/jigsaw-long-2000")
    model_size = "410M"
    generate_irl_demonstrations("jaredjoss/jigsaw-long-2000", 500, ["jaredjoss/pythia-410m-roberta-lr_8e7-kl_01-steps_12000-rlhf-model", "EleutherAI/pythia-410m", "random"], model_size, debug=False)
    







