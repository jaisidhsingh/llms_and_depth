from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm
from typing import *
import datasets
import torch
import tyro
import os

from src.utils import get_tokenizer, get_model, get_dataset
from src.configs import DATA_ROOT, data_configs
from src.tracer import ActivationTracer


@dataclass
class Args:
    model_name: str = "lns-llama-1b"
    dataset_name: str = "gsm8k-socratic"
    split: str = "test"
    device: str = "cuda"

@torch.no_grad()
def prepare_data(args):
    tokenizer = get_tokenizer(args.model_name)
    dataset = get_dataset(args.dataset_name, args.split)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(sample):
        if "gsm8k" in args.dataset_name:
            text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        else:
            text = sample['text']
        
        return tokenizer(text, truncation=False)

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
    max_length = max([len(item["input_ids"]) for item in tokenized_dataset])

    def tokenize_with_global_padding(sample):
        if "gsm8k" in args.dataset_name:
            text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        else:
            text = sample['text']
        
        return tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )

    final_dataset = dataset.map(tokenize_with_global_padding, remove_columns=dataset.column_names)
    inter_folder = data_configs.dataset_to_path[args.dataset_name].split("/")[-2]
    save_path = os.path.join(DATA_ROOT, inter_folder, f"{args.dataset_name}-{args.split}-full-tokenized-{args.model_name}")
    final_dataset.save_to_disk(save_path)

    new_dataset = datasets.load_from_disk(save_path)
    print(all( [len(item["input_ids"]) == len(new_dataset[0]["input_ids"]) for item in new_dataset] ))

@torch.no_grad()
def main(args):
    tokenizer = get_tokenizer(args.model_name)
    model = get_model(args.model_name, args.device)

    inter_folder = data_configs.dataset_to_path[args.dataset_name].split("/")[-2]
    save_path = os.path.join(DATA_ROOT, inter_folder, f"{args.dataset_name}-{args.split}-full-tokenized-{args.model_name}")
    dataset = datasets.load_from_disk(save_path)

    print(model.model.layers[0].layer_index) 

if __name__ == "__main__":
    args = tyro.cli(Args, default=vars(Args))
    main(args)
