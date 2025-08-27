from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import datasets
import torch
import os

import lm_eval
from lm_eval.models.huggingface import HFLM

from src.configs import model_configs
from src.configs import data_configs
from src.modelling.modelling_lns_llama_1b import LlamaForCausalLM

"""
Data utils
"""

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

def embed_dataset_collate_fn(batch):
    input_ids = torch.cat([torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0) for item in batch], dim=0)
    attention_mask = torch.cat([torch.tensor(item["attention_mask"], dtype=torch.long).unsqueeze(0) for item in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

"""
Model utils
"""

def get_model(model_name, device):
    if device is None:
        device = "auto"
    
    model_path = model_configs.model_name_to_path[model_name]
    if "lns-" in model_name:
        config = AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
        model = LlamaForCausalLM(config)
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), weights_only=True), strict=False)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model.to(device)
        model.eval()
        return model

    return AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True)

def cast_batch_to_device(batch, device):
    keys = ["input_ids", "attention_mask"]
    if "labels" in batch.keys():
        keys.append("labels")
    
    for k in keys:
        batch[k] = batch[k].to(device)
    
    return batch

"""
Result saving utils
"""

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
    save_path = os.path.join(save_folder, save_name)
    torch.save(result, save_path)
    print("Result saved at", save_path)

def remove_layers_after(layer_index, model_to_modify):
    num_layers = model_to_modify.config.num_hidden_layers
    num_layers_to_remove = num_layers - layer_index - 1
    for i in range(num_layers_to_remove):
        # del model_to_modify.model.layers[i]
        model_to_modify.model.layers.pop(-1)
        # print(i)
    
    print(model_to_modify.model.layers)

def evaluate_model(model, args, benchmark):
    lm = HFLM(model, device=args.device, batch_size=args.batch_size)
    print(f"Wrapped model in HFLM as desired by `lm_eval`. Note: only single process evaluation on device=`{lm._device}`")

    task_manager = lm_eval.task_manager()

    fewshot_as_multiturn = False
    apply_chat_template = False

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[benchmark],
        task_manager=task_manager,
        fewshot_as_multiturn=fewshot_as_multiturn,
        apply_chat_template=apply_chat_template
    )
    return results
