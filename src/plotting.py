import torch
import numpy as np
from pylab import plt
import plotly.graph_objects as go
import plotly.io as pio
import os

from src.tracer import LayerWiseMetricCache


def plot_cossims(x, save_name):
    plt.imshow(x, cmap="plasma", vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel("Layer index")
    plt.ylabel("Layer index")
    plt.title(save_name)
    plt.savefig(f"./plots/{save_name}.png")

def plot_eigenspectrum(x, save_name):
    x = x.tolist()
    x.sort()
    plt.plot([i for i in range(len(x))], x)
    plt.xlabel("Number of eigenvalues")
    plt.ylabel("Eigenvalue")
    plt.title(save_name)
    plt.savefig(f"./plots/{save_name}.png")

def plot_entropy(x, save_name):
    plt.plot([i for i in range(len(x))], x)
    plt.xlabel("Layer index")
    plt.ylabel("Entropy")
    plt.title(save_name)
    plt.savefig(f"./plots/{save_name}.png")


class Plotter:
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
