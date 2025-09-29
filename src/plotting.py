import torch
import numpy as np
from pylab import plt
import plotly.graph_objects as go
import plotly.io as pio
import os

from src.tracer import LayerWiseMetricCache
from src.utils import create_experiment_name


def normalize_values(x):
    min_, max_ = x.min(axis=0), x.max(axis=0)
    return x - min_ / (max_ - min_)

def plot_metrics(metrics, args):
    layers = list(metrics.keys())
    x = [i+1 for i in range(len(layers))]
    metric_names = args.metrics.split(",")
    data = {m : [] for m in metric_names}

    for m in metric_names:
        for l in layers:
            data[m].append(metrics[l][m].item())

        data[m] = np.array(data[m]).astype(np.float32)

    plt.plot(x, data[m], label=m)
    fig, axes = plt.subplots(1, len(metric_names), figsize=(15, 6))
    for idx, m in enumerate(metric_names):
        axes[idx].plot(x, data[m], label=m)
        axes[idx].set_xlabel("Depth")
        axes[idx].set_ylabel(f"{m} metric value")
        axes[idx].set_xticks([u+1 for u in range(len(x))])
        axes[idx].legend()
        axes[idx].set_title(m)

    name = create_experiment_name(args)
    save_path = os.path.join(args.results_folder, f"{name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Metrics plotted at {save_path}")


@torch.no_grad()
def plot_across_runs():
    folder = "/lustre/home/jsingh/projects/llms_and_depth/results_with_alpha"
    files = os.listdir(folder)
    model_names = [f.split("_")[0] for f in files]
    dataset_name = files[0].split("_")[1]
    seed = int(files[0].split("seed-")[-1].replace(".pt", ""))

    ww_metrics = ["mean_alphas", "weighted_alphas", "percent_untrained"]
    data_metrics = ["input_output_cossim", "batch_entropy"]

    data = {}
    for name, f in zip(model_names, files):
        per_run_data = torch.load(os.path.join(folder, f))
        layer_inds = [int(k.split("_")[-1]) for k in per_run_data.keys()]
        relative_depth = [ (idx + 1) / len(layer_inds) for idx in layer_inds]

        subdata = {"relative_depth": relative_depth}

        for m in ww_metrics:
            subdata[m] = []
            for k in per_run_data.keys():
                subdata[m].append( per_run_data[k][m].cpu().item() )
        
        for m in data_metrics:
            subdata[m] = []
            for k in per_run_data.keys():
                subdata[m].append( per_run_data[k][m].cpu().item() )
        
        data[name] = subdata
    
    torch.save(data, "/lustre/home/jsingh/projects/llms_and_depth/results_with_alpha/cleaned_data.pt")
    
    # ww_metrics first
    fig, axes = plt.subplots(1, len(ww_metrics), figsize=(20, 6))
    for i, m in enumerate(ww_metrics):
        for name in model_names:
            x = data[name]["relative_depth"]
            y = data[name][m]
            
            axes[i].plot(x, y, label=name)
            if m != "percent_untrained":
                axes[i].axhline(6.0, linestyle="--", c="black")
            axes[i].set_xlabel("relative depth")
            axes[i].set_ylabel(m)
            axes[i].legend()
            axes[i].set_title(f"{m} --- {dataset_name}")

    plt.suptitle(f"{dataset_name} -- {seed}") 
    plt.savefig("../ww_metrics.pdf", bbox_inches="tight", dpi=300)


    # data_metrics first
    fig, axes = plt.subplots(1, len(data_metrics), figsize=(20,6))
    for i, m in enumerate(data_metrics):
        for name in model_names:
            x = data[name]["relative_depth"]
            y = data[name][m]
            
            axes[i].plot(x, y, label=name)
            # axes[i].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            axes[i].set_xlabel("relative depth")
            axes[i].set_ylabel(m)
            axes[i].legend()
            axes[i].set_title(f"{m} --- {dataset_name}")
    
    plt.suptitle(f"{dataset_name} -- {seed}") 
    plt.savefig("../data_metrics.pdf", bbox_inches="tight", dpi=300)

# @torch.no_grad()
# def plot_ppls():
#     pass