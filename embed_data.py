from transformers import AutoConfig
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm
from typing import *
from pylab import plt
import numpy as np
from copy import deepcopy
import datasets
import torch
import tyro
import os

from src.utils import *
from src.configs import *
from src.math_utils import *
from src.plotting_utils import *
from src.tracer import HookedModel


@dataclass
class Args:
    model_name: str = "llama-1b"
    dataset_name: str = "gsm8k-main"
    split: str = "test"
    device: str = "cuda"
    batch_size: int = 16
    num_workers: int = 8
    num_recurrent_steps: int = 8
    hook_type: str = "reduced"
    decorrelate: bool = False
    to_hook: str = "layer_output"
    reduction: str = "token_norm"


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
    model = get_model(args.model_name, args.device, get_init_model=True)
    autocast = torch.amp.autocast

    inter_folder = data_configs.dataset_to_path[args.dataset_name].split("/")[-2]
    dataset_save_path = os.path.join(DATA_ROOT, inter_folder, f"{args.dataset_name}-{args.split}-full-tokenized-{args.model_name}")
    dataset = datasets.load_from_disk(dataset_save_path)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=embed_dataset_collate_fn)
    main_cache = None

    kwargs = {
        "to_hook": args.to_hook,
        "hook_type": args.hook_type,
        "decorrelate": args.decorrelate
    }
    
    hooked_model = HookedModel(model, args.model_name, reduction=args.reduction, kwargs=kwargs)
    unpadded = 0

    bar = tqdm(total=len(loader))
    for idx, batch in enumerate(loader):
        batch = cast_batch_to_device(batch, args.device)
        # batch["attention_mask"] = None
        
        # get an estimate of how much we've padded
        if idx == 0:
            unpadded = batch["attention_mask"].clone().float().sum(dim=1).mean(dim=0)
        
        with autocast(args.device):
            outputs = hooked_model(batch)

        if main_cache is None:
            main_cache = deepcopy(hooked_model.cache)
        else:
            main_cache.add_cache_keywise(hooked_model.cache)
        
        hooked_model.cache.clear()
        bar.update(1)
        
    main_cache.print_shapes()

    x = collect_from_cache(main_cache.store)
    # l = 1
    # y = x - x[l]
    # y = self_cos_sim(y[l+1:])
    x = self_cos_sim(x)
    plot_desc = f"{args.model_name}-{args.dataset_name}_init_cossims"
    # if not args.decorrelate:
    #     plot_desc = plot_desc.replace("decorr_", f"decorr-l{l}_")
    plot_cossims(x, plot_desc)
    
    # unpadded = unpadded.cpu().item()
    # im = []
    # for i in range(model.config.num_hidden_layers):
    #     x = old_cache.store[f"layer_{i}"].mean(0)
    #     im.append(np.expand_dims(x, axis=0))
    
    # im = np.concatenate(im, axis=0)
    # im = im[:, :int(unpadded)]
    # plt.figure(figsize=(6, 12))
    # plt.imshow(im.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    # plt.xticks([i for i in range(model.config.num_hidden_layers)])
    # plt.ylabel("token positions (padding removed)")
    # plt.xlabel("layers")
    # plt.title(f"Input-Output similarity across depth\n{args.model_name}---{args.dataset_name}")
    # plt.colorbar()
    # plt.savefig(f"./new_plots/{args.model_name}_{args.dataset_name}_inp-out-cossims.png", bbox_inches="tight")

    # save_path = os.path.join(RESULTS_ROOT, args.model_name, f"{args.dataset_name}-{args.split}-full-tokenized-{args.model_name}")
    # model_path = model_configs.model_name_to_path[args.model_name]
    # cache_save_path = os.path.join(model_path, save_path.split("/")[-1]+"_mean.pt")
    # torch.save(old_cache.store, cache_save_path)
    print("All done!")

if __name__ == "__main__":
    args = tyro.cli(Args, default=vars(Args))
    main(args)
