from transformers import AutoConfig
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm
from typing import *
from copy import deepcopy
import datasets
import torch
import tyro
import os

from src.utils import *
from src.configs import DATA_ROOT, data_configs, model_configs
from src.tracer import HookedModel
from src.modelling.modelling_lns_llama_1b import LlamaForCausalLM


@dataclass
class Args:
    model_name: str = "recurrent-1b"
    dataset_name: str = "gsm8k-main"
    split: str = "test"
    device: str = "cuda"
    batch_size: int = 4
    num_workers: int = 8
    num_recurrent_steps: int = 8


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
    autocast = torch.amp.autocast

    inter_folder = data_configs.dataset_to_path[args.dataset_name].split("/")[-2]
    save_path = os.path.join(DATA_ROOT, inter_folder, f"{args.dataset_name}-{args.split}-full-tokenized-{args.model_name}")
    dataset = datasets.load_from_disk(save_path)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=embed_dataset_collate_fn)
    old_cache = None

    kwargs = {}
    if "recurrent" in args.model_name:
        kwargs["num_recurrent_steps"] = args.num_recurrent_steps
    
    hooked_model = HookedModel(model, args.model_name, reduction="token_mean", kwargs=kwargs)

    bar = tqdm(total=len(loader))
    for idx, batch in enumerate(loader):
        batch = cast_batch_to_device(batch, args.device)
        batch["attention_mask"] = None

        with autocast(args.device):
            outputs = hooked_model(batch)

        if old_cache is None:
            old_cache = deepcopy(hooked_model.cache)
        else:
            old_cache.add_cache_keywise(hooked_model.cache)
        
        hooked_model.cache.clear()
        old_cache.print_shapes()
        
        bar.update(1)
        
    old_cache.print_shapes()
    model_path = model_configs.model_name_to_path[args.model_name]
    cache_save_path = os.path.join(model_path, save_path.split("/")[-1]+"_mean.pt")
    torch.save(old_cache.store, cache_save_path)
    print("All done!")


if __name__ == "__main__":
    args = tyro.cli(Args, default=vars(Args))
    main(args)
