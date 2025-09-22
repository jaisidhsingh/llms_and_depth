import torch


@torch.no_grad()
def make_layerwise_gram_matrix(x, device):
    # needs x.shape: (num_layers, num_samples, dim)
    x = x.float().to(device)
    x = torch.einsum("nbd,ncd->nbc", x, x)
    return x

@torch.no_grad()
def shannon_entropy(x, device):
    # for prompt/dataset entropy
    x = x.float().to(device)
    e = torch.linalg.eigvals(x).real / x.trace()
    return -1 * (e * torch.log(e)).sum()

@torch.no_grad()
def self_cos_sim(x, device):
    # needs x.shape = (num_data_samples, num_layers, dim)
    x = x.float().to(device).transpose(0, 1)
    x /= x.norm(dim=-1, keepdim=True)
    sim = torch.einsum("bnd,bmd->bnm", x, x)
    return sim.mean(0).cpu().numpy()

@torch.no_grad()
def self_eigenspectrum(x, device):
    x = x.float().to(device)
    x = torch.linalg.eigvals(x).real
    return x.cpu().numpy()
