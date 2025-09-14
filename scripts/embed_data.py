import os
import tyro
import torch
import datasets
import numpy as np
from typing import *
from pylab import plt
from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
from transformers import AutoConfig
from torch.utils.data import DataLoader

from src.tracer import Tracer
from src.utils import get_model, get_tokenizer, get_dataset


@torch.no_grad()
def main(args):
    tokenizer = get_tokenizer(args.model_name)
    model = get_model(args.model_name, args.device, get_init_model=True)

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

        # get an estimate of how much we've padded
        if idx == 0:
            unpadded = batch["attention_mask"].clone().float().sum(dim=1).mean(dim=0)

        with torch.autocast(args.device):
            outputs = hooked_model(batch)
            del outputs

        if main_cache is None:
            main_cache = deepcopy(hooked_model.cache)
        else:
            main_cache.add_cache_keywise(hooked_model.cache)

        hooked_model.cache.clear()
        bar.update(1)

    main_cache.print_shapes()

    x = collect_from_cache(main_cache.store)
    print(x.shape)

    # x = self_cos_sim(x)
    # plot_desc = f"{args.model_name}-{args.dataset_name}_init_cossims"
    # plot_cossims(x, plot_desc)
    # print("All done!")

if __name__ == "__main__":
    args = tyro.cli(Args, default=vars(Args))
    main(args)
