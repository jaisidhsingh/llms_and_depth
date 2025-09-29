import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from copy import deepcopy
from dotenv import dotenv_values

import datasets
from torch.nn import Identity
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# find our variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_VARS = dotenv_values(dotenv_path=PROJECT_ROOT / ".env")
assert os.path.exists(ENV_VARS['RESULTS_FOLDER']) and os.path.exists(ENV_VARS['CONFIGS_FOLDER']), "Specified folders do not exist"

# map short-hand names to path
DATASET_NAME_TO_PATH = {
    "gsm8k-main": f"{ENV_VARS['DATASETS_FOLDER']}/gsm8k-saved/gsm8k-main",
    "gsm8k-socratic": f"{ENV_VARS['DATASETS_FOLDER']}/gsm8k-saved/gsm8k-socratic",
    "c4-10k": f"{ENV_VARS['DATASETS_FOLDER']}/c4-10k-saved/data",
}
MODEL_NAME_TO_PATH = {
    "llama3.2-1b": f"{ENV_VARS['PRETRAINED_MODELS_FOLDER']}/Llama-3.2-1B",
    "qwen3-0.6b": f"{ENV_VARS['PRETRAINED_MODELS_FOLDER']}/Qwen3-0.6B-Base",
    "qwen3-0.6b-it": f"{ENV_VARS['PRETRAINED_MODELS_FOLDER']}/Qwen3-0.6B",
    "gemma3-1b-pt": f"{ENV_VARS['PRETRAINED_MODELS_FOLDER']}/gemma-3-1b-pt",
    "gemma3-1b-it": f"{ENV_VARS['PRETRAINED_MODELS_FOLDER']}/gemma-3-1b-it",
    "gemma2-2b": f"{ENV_VARS['PRETRAINED_MODELS_FOLDER']}/gemma-2-2b",
}
DATASET_TO_HF_ID = {
    "gsm8k-main": "openai/gsm8k",
    "gsm8k-socratic": "openai/gsm8k",
}
MODEL_TO_HF_ID = {
    "llama3.2-1b": "meta-llama/Llama-3.2-1B"
}

# model_name: num_attn_linear_layers + num_mlp_linear_layers
MODEL_TO_LAYER_DIV = {
    "llama3.2-1b": 4 + 3,
    "qwen3-0.6b": 4 + 3,
    "qwen3-0.6b-it": 4 + 3,
    "gemma3-1b-pt": 4 + 3,
    "gemma3-1b-it": 4 + 3,
    "gemma2-2b": 4 + 3,
}

# data utils
def map_gsm8k_inputs(sample):
    answer = sample['answer'].replace("####", "Final answer value = ")
    return {"input_text" : f"QUESTION: {sample['question']}  ANSWER: {answer}"}

def map_c4_inputs(sample):
    return sample

def get_dataset(dataset_name, split, on_colab=False):
    dataset_path = DATASET_NAME_TO_PATH[dataset_name]
    if dataset_name == "c4-10k":
        split = "train" # has no test split

    if not on_colab:
        dataset = datasets.load_from_disk(dataset_path)[split]
        map_func = map_gsm8k_inputs if "gsm8k" in dataset_name else map_c4_inputs
        dataset = dataset.map(map_func, remove_columns=dataset.column_names)
    else:
        dataset_id = DATASET_TO_HF_ID[dataset_name]
        if "gsm8k" in dataset_name:
            data_type = dataset_name.split("-")[1]
            dataset = datasets.load_dataset(dataset_id, data_type, split=split)
        else:
            dataset = datasets.load_dataset(dataset_id, split=split)

        map_func = map_gsm8k_inputs if "gsm8k" in dataset_name else map_c4_inputs
        dataset = dataset.map(map_func, remove_columns=dataset.column_names)

    return dataset

def get_tokenizer(model_name, on_colab=False):
    if not on_colab:
        model_path = MODEL_NAME_TO_PATH[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model_id = MODEL_TO_HF_ID[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def embed_dataset_collate_fn(batch):
    input_ids = torch.cat([torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0) for item in batch], dim=0)
    attention_mask = torch.cat([torch.tensor(item["attention_mask"], dtype=torch.long).unsqueeze(0) for item in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def cast_batch_to_device(batch, device):
    for k, v in batch.items():
        batch[k] = v.to(device)
    return batch


# model utils
def get_model(model_name, device, get_init_model=False, on_colab=False):
    if not get_init_model:
        if not on_colab:
            model_path = MODEL_NAME_TO_PATH[model_name]
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True, torch_dtype=torch.float32,
                attn_implementation="eager" if "gemma" in model_name else "sdpa"
            )
        else:
            model_id = MODEL_TO_HF_ID[model_name]
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, trust_remote_code=True, torch_dtype=torch.float32)
    else:
        if not on_colab:
            model_path = MODEL_NAME_TO_PATH[model_name]
            config = AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
        else:
            model_id = MODEL_TO_HF_ID[model_name]
            config = AutoConfig.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)

    model = model.to(device)
    model.eval()
    return model

def remove_layers_after(layer_index, model_to_modify):
    num_layers = model_to_modify.config.num_hidden_layers
    for i in range(layer_index+1, num_layers):
        model_to_modify.model.layers[i] = Identity()

    print(f"Layers {layer_index+1} to {num_layers} removed inplace.")

def remove_layer_at(layer_index, model_to_modify):
    model_to_modify.model.layers[layer_index] = Identity()


# other utils
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def print_intro(args):
    print("#"*50)
    print(f"# Model: {args.model_name}")
    print(f"# Dataset: {args.dataset_name}, split: {args.split}")
    print(f"# Using device: {args.device}")
    print(f"# Computing metrics: {args.metrics}")
    print(f"# Batch size: {args.batch_size}")
    print(f"# Num workers: {args.num_workers}")
    print(f"# Random seed: {args.random_seed}")
    print("#"*50)
    print(" ")

def collect_from_cache(cache):
    alls = []
    layer_indices = [int(item.split("_")[-1]) for item in list(cache.keys())]
    layer_indices.sort()

    for i in layer_indices:
        alls.append(cache[f"layer_{i}"])

    out = np.stack(alls)
    return out

def create_experiment_name(args):
    return f"{args.model_name}_{args.dataset_name}_{args.metrics}_seed-{args.random_seed}"

def save_result(result, save_name, args):
    save_folder = os.path.join(args.results_folder, args.model_name, args.dataset_name)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, save_name)
    torch.save(result, save_path)
    print("Result saved at", save_path)

def split_args(args):
    metrics = set(args.metrics.split(","))
    all_eval_metrics = set(["drop_perf", "drop_ppl"])

    eval_metrics = metrics.intersection(all_eval_metrics)
    non_eval_metrics = metrics - eval_metrics

    non_eval_args, eval_args = None, None

    if len(list(non_eval_metrics)) > 0:
        non_eval_args = deepcopy(args)
        non_eval_args.metrics = ",".join(list(non_eval_metrics))

    if len(list(eval_metrics)) > 0:
        eval_args = deepcopy(args)
        eval_args.metrics = ",".join(list(eval_metrics))

    return non_eval_args, eval_args

def save_metrics(metrics, args):
    name = create_experiment_name(args)
    result_save_path = os.path.join(args.results_folder, f"{name}.pt")
    torch.save(metrics, result_save_path)

    config_save_path = os.path.join(args.config_folder, f"{name}.json")
    with open(config_save_path, "w") as f:
        json.dump(vars(args), f)

    print(f"Results saved at {result_save_path}")
    print(f"Config to reproduce results saved at {config_save_path}")

def collect_metrics(metrics_list):
    tmp = metrics_list[0]
    rows = tmp.keys()
    for item in metrics_list[1:]:
        for k in rows:
            for kk, vv in item[k].items():
                tmp[k][kk] = vv
    return tmp
