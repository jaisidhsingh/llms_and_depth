import torch
import numpy as np
from dataclasses import dataclass
import tyro

from src.utils import *
from src.configs import *
from src.math_utils import *
from src.plotting_utils import *


@dataclass
class Args:
    model_name: str = "lns-llama-1b"
    dataset_name: str = "gsm8k-main"
    split: str = "test"
    device: str = "cuda"
    reduction: str = "mean"
    plot_metric: str = "cossims"


def main(args):
    inter_folder = data_configs.dataset_to_path[args.dataset_name].split("/")[-2]
    save_path = os.path.join(DATA_ROOT, inter_folder, f"{args.dataset_name}-{args.split}-full-tokenized-{args.model_name}")
    
    model_path = model_configs.model_name_to_path[args.model_name]
    if args.reduction == "norm":
        cache_save_path = os.path.join(model_path, save_path.split("/")[-1]+".pt")
    elif args.reduction == "mean":
        cache_save_path = os.path.join(model_path, save_path.split("/")[-1]+"_mean.pt")

    cache = torch.load(cache_save_path, weights_only=False, map_location="cpu")
    x = collect_from_cache(cache)

    if args.plot_metric == "cossims":
        x = self_cos_sim(x)
        plot_cossims(x, f"{args.model_name}_{args.plot_metric}")
    
    elif args.plot_metric == "entropy":
        x = make_layerwise_gram_matrix(x)
        ents = []
        for subx in x:
            ents.append(shannon_entropy(subx))

        plot_entropy(ents, f"{args.model_name}_{args.plot_metric}")


if __name__ == "__main__":
    args = tyro.cli(Args, default=vars(Args))
    main(args)

