from pylab import plt
import numpy as np
import torch


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
