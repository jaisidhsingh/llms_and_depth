import os
import tyro
from tqdm import tqdm

from src.utils import split_args
from src.cli import Args


def compute_non_eval_metrics(args):
    pass

def compute_eval_metrics(args):
    pass

def save_metrics(metrics, args):
    pass

def plot_metrics(metrics, args):
    pass

def collect_metrics(metrics_list):
    tmp = metrics_list[0]
    rows = tmp.keys()
    for item in metrics_list[1:]:
        for k in rows:
            for kk, vv in item[k].items():
                tmp[k][kk] = vv
    return tmp


def main(args):
    non_eval_args, eval_args = split_args(args)
    non_eval_metrics = compute_non_eval_metrics(non_eval_args)

    # make sure we have something saved and plotted at every step
    save_metrics(non_eval_metrics, args)
    plot_metrics(non_eval_metrics, args)

    eval_metrics = compute_eval_metrics(eval_args)

    # now we have everything so we over-write what we saved and plotted earlier
    metrics = collect_metrics([non_eval_metrics, eval_metrics])
    save_metrics(metrics, args)
    plot_metrics(metrics, args)

    # whew
    print("All done!")


if __name__ == "__main__":
    args = tyro.cli(Args, default=vars(Args()))
    main(args)
