from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import datasets

from src.configs import model_configs
from src.configs import data_configs


def get_dataset(dataset_name, split):
    dataset_path = data_configs.dataset_to_path[dataset_name]
    if dataset_name == "c4-10k":
        split = "train" # has no test split
    dataset = datasets.load_from_disk(dataset_path)[split]
    return dataset

def get_tokenizer(model_name):
    model_path = model_configs.model_name_to_path[model_name]
    if "lns-" in model_name:
        return AutoTokenizer.from_pretrained(model_path[:-6]+"t5-base")

    return AutoTokenizer.from_pretrained(model_path)

def get_model(model_name, device):
    if device is None:
        device = "auto"
    
    model_path = model_configs.model_name_to_path[model_name]
    return AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
