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
        axes[idx].legend()
        axes[idx].set_title(m)

    name = create_experiment_name(args)
    save_path = os.path.join(args.results_folder, f"{name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Metrics plotted at {save_path}")


class InteractivePlotter:
    def __init__(self, cache: LayerWiseMetricCache, out_dir="plots/interactive"):
        self.cache = cache
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        # Pro-paper color scheme (Plotly default, but can be customized)
        self.color_scheme = pio.templates["plotly_white"]

    def plot_metrics(self):
        # Assume cache.data is {layer_idx: {metric_name: tensor}}
        # Finalize cache if not already done
        if not self.cache.is_finalised:
            self.cache.finalize()

        # Gather all metric names
        metric_names = set()
        for layer_dict in self.cache.data.values():
            metric_names.update(layer_dict.keys())

        # For each metric, collect values for each layer
        for metric in metric_names:
            layer_indices = []
            metric_values = []
            for layer_idx in sorted(self.cache.data.keys()):
                layer_dict = self.cache.data[layer_idx]
                if metric in layer_dict:
                    # If tensor, reduce to scalar (mean), else use as is
                    val = layer_dict[metric]
                    if isinstance(val, torch.Tensor):
                        # If 1D or more, take mean
                        if val.numel() > 1:
                            val = val.mean().item()
                        else:
                            val = val.item()
                    metric_values.append(val)
                    layer_indices.append(layer_idx)

            # Create interactive plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=layer_indices,
                y=metric_values,
                mode="lines+markers",
                marker=dict(color="#636EFA"),  # Plotly blue
                line=dict(color="#636EFA"),
                name=metric
            ))
            fig.update_layout(
                title=f"{metric} across transformer layers",
                xaxis_title="Layer index",
                yaxis_title=metric,
                template="plotly_white"
            )
            # Save to HTML
            html_path = os.path.join(self.out_dir, f"{metric}_layerwise.html")
            fig.write_html(html_path)
