import torch
import torch.nn.functional as F

import src.metrics as metrics


class LayerWiseMetricCache:
    # stores metrics computed for every layer in a LM
    # row: layer_{layer_index}
    # col: metric_name
    def __init__(self, cache={}):
        self.data = cache
        self.is_finalised = False

    def __idx__(self, key):
        return self.data[key]

    @torch.no_grad()
    def push(self, row, col, val):
        if not row in self.data.keys():
            self.data[row] = {}
            self.data[row][col] = []

        elif not col in self.data[row].keys():
                self.data[row][col] = []

        self.data[row][col].append(val)

    @torch.no_grad()
    def push_from_cache(self, cache):
        for crow, ccol in cache.data.items():
            for k, v in ccol.items():
                if not v[0].device == "cpu":
                    v[0] = v[0].cpu()
                self.data[crow][k].append(v[0])

    @torch.no_grad()
    def finalize(self, reduce=True):
        for row, col in self.data.items():
            for k, v in col.items():
                self.data[row][k] = torch.cat(v, dim=0)
                if reduce:
                    self.data[row][k] = self.data[row][k].mean(dim=0)
        self.is_finalised = True

    def clear(self):
        self.data = {}

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

    def llama_hooks(self, layer_index, metric):
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
            token_mean_output = output_tensor[0].mean(dim=1) # shape: [B, D]
            gram = token_mean_output @ token_mean_output.T
            ent = metrics.shannon_entropy(gram)
            self.cache.push(f"layer_{layer_index}", "batch_entropy", ent)

        @torch.no_grad()
        def attn_rank(module, input_tensor, output_tensor):
            assert len(output_tensor) > 1, "Attention weights are not returned by the attention layer, cannot compute `attention_rank`"
            batch_attn_rank = torch.linalg.matrix_rank(output_tensor[1]).mean(dim=-1)
            self.cache.push(f"layer_{layer_index}", "attn_rank", batch_attn_rank.cpu())

        metric_to_hook_map = {
            "input_output_cossim": input_output_cosine_sim,
            "batch_entropy": batch_entropy,
            "attn_rank": attn_rank
        }
        return metric_to_hook_map[metric]

    def add_hooks_to_model(self):
        if "llama-" in self.model_name:
            num_layers = self.model.config.num_hidden_layers
            metrics = self.config.metrics.split(",")

            for i in range(num_layers):
                row = f"layer_{i}"
                if row not in self.active_hooks.keys():
                    self.active_hooks[row] = {}

                for metric in metrics:
                    if metric not in self.active_hooks[row].keys():
                        self.active_hooks[row][metric] = {}

                    if metric != "attn_rank":
                        self.active_hooks[row][metric] = self.model.model.layers[i].register_forward_hook(
                            self.llama_hooks(i, metric)
                        )
                    else:
                        self.active_hooks[row]["attn_rank"] = self.model.model.layers[i].self_attn.register_forward_hook(
                            self.llama_hooks(i, "attn_rank")
                        )

    def remove_all_hooks(self):
        for k, v in self.active_hooks.items():
            for kk in v.keys():
                self.active_hooks[k][kk].remove()
                del self.active_hooks[k][kk]

    def __call__(self, batch):
        return self.model(**batch)
