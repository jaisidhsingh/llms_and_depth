import torch
import torch.nn.functional as F
from copy import deepcopy
import src.metrics as metrics
from src.utils import MODEL_TO_LAYER_DIV
import weightwatcher as ww


class LayerWiseMetricCache:
    # stores metrics computed for every layer in a LM
    # row: layer_{layer_index}
    # col: metric_name
    def __init__(self, cache={}):
        self.data = cache
        self.is_finalised = False

    @torch.no_grad()
    def push(self, row, col, val):
        if not row in self.data.keys():
            self.data[row] = {}
            self.data[row][col] = []

        elif not col in self.data[row].keys():
                self.data[row][col] = []

        self.data[row][col].append(val.clone())

    @torch.no_grad()
    def push_from_cache(self, cache):
        for crow, ccol in cache.data.items():
            for k, v in ccol.items():
                self.push(crow, k, deepcopy(v[0]))

    @torch.no_grad()
    def finalize(self, reduce=True):
        for row, col in self.data.items():
            for k, v in col.items():
                self.data[row][k] = torch.cat(v, dim=0)
                if reduce:
                    self.data[row][k] = self.data[row][k].mean(dim=0)
        self.is_finalised = True

    def clear(self):
        self.data.clear()

    def print_shapes(self):
        if self.is_finalised:
            for k in self.data.keys():
                for kk, v in self.data[k].items():
                    print(f"{k} --- {kk} --- {v.shape}")
        else:
            raise Exception("One needs to do `cache.finalise()` before one can do `cache.print_shapes()`.")


class Tracer:
    def __init__(self, model, model_name, config):
        self.model_name = model_name
        self.model = model
        self.config = config
        self.cache = LayerWiseMetricCache()
        self.active_hooks = {}
        self.add_hooks_to_model()

    def get_traced_model_name(self):
        name = f"{self.config.dataset_name}-{self.config.split}_{self.config.model_name}_{self.config.to_hook}-{self.config.reduction}"
        return name

    def all_hooks(self, layer_index, metric):
        # separate hook for each metric
        # hooks can't be used for layer dropping

        @torch.no_grad()
        def input_output_cosine_sim(module, input_tensor, output_tensor):
            input_t = F.normalize(input_tensor[0], dim=-1)
            output_t = F.normalize(output_tensor[0], dim=-1)
            sim = (input_t * output_t).sum(dim=-1) # shape: [B, L]
            sim = sim.mean(dim=[0,1]).unsqueeze(0) # shape: [1]
            self.cache.push(f"layer_{layer_index}", "input_output_cossim", sim.cpu())

        @torch.no_grad()
        def batch_entropy(module, input_tensor, output_tensor):
            bs = output_tensor[0].shape[0]
            if bs > 1:
                token_mean_output = output_tensor[0].mean(dim=1) # shape: [B, D]
            else:
                token_mean_output = output_tensor[0].squeeze(dim=0) # shape [L, D]
            gram = token_mean_output @ token_mean_output.T
            ent = metrics.shannon_entropy(gram, device=self.config.device)
            self.cache.push(f"layer_{layer_index}", "batch_entropy", ent.unsqueeze(0))

        @torch.no_grad()
        def attn_rank(module, input_tensor, output_tensor):
            assert len(output_tensor) > 1, "Attention weights are not returned by the attention layer, cannot compute `attention_rank`"
            heads_attn_rank = torch.linalg.matrix_rank(output_tensor[1]).float()
            batch_attn_rank = heads_attn_rank.mean(dim=-1)
            self.cache.push(f"layer_{layer_index}", "attn_rank", batch_attn_rank.cpu())

        metric_to_hook_map = {
            "input_output_cossim": input_output_cosine_sim,
            "batch_entropy": batch_entropy,
            "attn_rank": attn_rank
        }
        return metric_to_hook_map[metric]

    def add_hooks_to_model(self):
        num_layers = self.model.config.num_hidden_layers
        metrics = self.config.metrics.split(",")
        metrics = [item for item in metrics if item != "ww_alpha"]

        for i in range(num_layers):
            row = f"layer_{i}"
            if row not in self.active_hooks.keys():
                self.active_hooks[row] = {}

            for metric in metrics:
                if metric not in self.active_hooks[row].keys():
                    self.active_hooks[row][metric] = {}

                if metric != "attn_rank":
                    self.active_hooks[row][metric] = self.model.model.layers[i].register_forward_hook(
                        self.all_hooks(i, metric)
                    )
                else:
                    self.active_hooks[row]["attn_rank"] = self.model.model.layers[i].self_attn.register_forward_hook(
                        self.all_hooks(i, "attn_rank")
                    )

    def remove_all_hooks(self):
        for k, v in self.active_hooks.items():
            for kk in v.keys():
                self.active_hooks[k][kk].remove()

        self.active_hooks.clear()
    
    def chunk_all_tensor_metrics(self, per_tensor_metrics, tensors_per_block):
        total = len(per_tensor_metrics)
        chunks = []

        for start in range(0, total, tensors_per_block):
            chunk = []
        
            for j in range(tensors_per_block):
                chunk.append(per_tensor_metrics[start + j].item())
        
            chunks.append(chunk)
        chunks = torch.tensor(chunks, dtype=torch.float32, device="cpu")
        assert chunks.ndim == 2, f"Chunked WeightWatcher metric tensor should have no. of dimension = 2, found {chunks.ndim}"
        return chunks

    def attach_weightwatcher(self):
        """
        WeightWatcher computes alpha (ESD-PL fit) for each weight tensor in the model.
        That means >= 3 linear layers for attention layers and >= 2 linear layers for FFN layers.
        We need to aggregate the per-tensor alpha values into a per-transformer-block alpha value.
        """
        watcher = ww.WeightWatcher(model=self.model)
        details = watcher.analyze()

        alphas = details['alpha']
        num_eigenvals = details['num_evals']

        tensors_per_block = MODEL_TO_LAYER_DIV[self.model_name]
        alphas_chunked = self.chunk_all_tensor_metrics(alphas, tensors_per_block)
        num_eigenvals_chunked = self.chunk_all_tensor_metrics(num_eigenvals, tensors_per_block)

        mean_alphas = alphas_chunked.mean(dim=-1)
        weighted_alphas = (alphas_chunked * num_eigenvals_chunked).sum(dim=-1) / num_eigenvals_chunked.sum(dim=-1)
        percent_untrained = (alphas_chunked > 6.0).sum(dim=-1) / alphas_chunked.shape[-1]

        ww_metrics = {
            "mean_alphas": mean_alphas,
            "weighted_alphas": weighted_alphas,
            "percent_untrained": percent_untrained
        }

        num_layers = self.model.config.num_hidden_layers

        for k, v in ww_metrics.items():
            for i in range(num_layers):
                value = v[i]
                assert value.ndim <= 1, "For some reason, per layer WeightWatcher metric value is more than one dimensional."
                
                if value.ndim < 1:
                    value = value.unsqueeze(0)

                self.cache.push(f"layer_{i}", k, value)


    def __call__(self, batch):
        return self.model(**batch, output_attentions="attn_rank" in self.config.metrics)
