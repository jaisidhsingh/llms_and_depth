import torch


@torch.no_grad()
def make_layerwise_gram_matrix(x, device="cuda"):
    # shape: (num_layers, num_samples, dim)
    x = torch.from_numpy(x).float().to(device)
    x = torch.einsum("nbd,ncd->nbc", x, x)
    return x

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
