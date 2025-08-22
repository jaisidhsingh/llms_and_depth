from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import datasets
import torch
import os

from src.configs import model_configs
from src.configs import data_configs
from src.modelling.modelling_lns_llama_1b import LlamaForCausalLM


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

def embed_dataset_collate_fn(batch):
    input_ids = torch.cat([torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0) for item in batch], dim=0)
    attention_mask = torch.cat([torch.tensor(item["attention_mask"], dtype=torch.long).unsqueeze(0) for item in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def cast_batch_to_device(batch, device):
    keys = ["input_ids", "attention_mask"]
    if "labels" in batch.keys():
        keys.append("labels")
    
    for k in keys:
        batch[k] = batch[k].to(device)
    
    return batch

@torch.no_grad()
def shannon_entropy(x):
    e = torch.linalg.eigvals(x).real
    t = x.trace()
    e_ = e / t
    return -1 * (e_ * torch.log(e_)).sum().item()

@torch.no_grad()
def self_cos_sim(x, setting="token_norm", device="cuda"):
    x = torch.from_numpy(x).float().to(device).transpose(0, 1)
    x /= x.norm(dim=-1, keepdim=True)
    # if `token_norm`, x.shape = (num_data_samples, num_layers, dim)
    sim = torch.einsum("bnd,bmd->bnm", x, x)
    return sim.mean(0).cpu().numpy()

@torch.no_grad()
def self_eigenspectrum(x, setting="token_norm", device="cuda"):
    x = torch.from_numpy(x).float().to(device)
    x = torch.linalg.eigvals(x).real
    return x.cpu().numpy()

def collect_from_cache(cache):
    alls = []
    layer_indices = [int(item.split("_")[-1]) for item in list(cache.keys())]
    layer_indices.sort()

    for i in layer_indices:
        alls.append(cache[f"layer_{i}"])
    
    out = np.stack(alls)
    return out

@torch.no_grad()
def make_layerwise_gram_matrix(x, device="cuda"):
    # shape: (num_layers, num_samples, dim)
    x = torch.from_numpy(x).float().to(device)
    x = torch.einsum("nbd,ncd->nbc", x, x)
    return x
