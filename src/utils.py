import os
import torch
import numpy as np
from pathlib import Path
from dotenv import dotenv_values

import datasets
from torch.nn import Identity
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import lm_eval
from lm_eval.models.huggingface import HFLM


# find our variables
project_root = Path(__file__).resolve().parent.parent
env_vars = dotenv_values(dotenv_path=project_root / ".env")

# map short-hand names to path
DATASET_NAME_TO_PATH = {
    "gsm8k-main": f"{env_vars['DATASETS_FOLDER']}/gsm8k-saved/gsm8k-main",
    "gsm8k-socratic": f"{env_vars['DATASETS_FOLDER']}/gsm8k-saved/gsm8k-socratic",
    "c4-10k": f"{env_vars['DATASETS_FOLDER']}/c4-10k-saved/data",
}
MODEL_NAME_TO_PATH = {
    "llama-1b": f"{env_vars['PRETRAINED_MODELS_FOLDER']}/Llama-3.2-1B"
}


# data utils
def get_dataset(dataset_name, split):
    dataset_path = DATASET_NAME_TO_PATH[dataset_name]
    if dataset_name == "c4-10k":
        split = "train" # has no test split
    dataset = datasets.load_from_disk(dataset_path)[split]
    return dataset

def get_tokenizer(model_name):
    model_path = MODEL_NAME_TO_PATH[model_name]
    return AutoTokenizer.from_pretrained(model_path)

def embed_dataset_collate_fn(batch):
    input_ids = torch.cat([torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0) for item in batch], dim=0)
    attention_mask = torch.cat([torch.tensor(item["attention_mask"], dtype=torch.long).unsqueeze(0) for item in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def cast_batch_to_device(batch, device):
    for k, v in batch.items():
        batch[k] = v.to(device)
    return batch


# model utils
def get_model(model_name, device, get_init_model=False):
    model_path = MODEL_NAME_TO_PATH[model_name]
    if not get_init_model:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
        model = AutoModelForCausalLM.from_config(config)

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

def collect_from_cache(cache):
    alls = []
    layer_indices = [int(item.split("_")[-1]) for item in list(cache.keys())]
    layer_indices.sort()

    for i in layer_indices:
        alls.append(cache[f"layer_{i}"])

    out = np.stack(alls)
    return out

def save_result(result, save_name, args):
    save_folder = os.path.join(args.results_folder, args.model_name, args.dataset_name)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, save_name)
    torch.save(result, save_path)
    print("Result saved at", save_path)


# eval utils
def evaluate_model(model, args, benchmark):
    lm = HFLM(model, device=args.device, batch_size=args.batch_size)
    print(f"Wrapped model in HFLM as desired by `lm_eval`. Note: only single process evaluation on device=`{lm._device}`")

    task_manager = lm_eval.task_manager()

    fewshot_as_multiturn = False
    apply_chat_template = False

    if "llama" in benchmark:
        fewshot_as_multiturn = True

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[benchmark],
        task_manager=task_manager,
        fewshot_as_multiturn=fewshot_as_multiturn,
        apply_chat_template=apply_chat_template
    )
    return results
