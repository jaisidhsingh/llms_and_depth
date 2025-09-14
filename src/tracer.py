import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass

from src.metrics import *


class LayerWiseMetricCache:
    def __init__(self, cache={}):
        self.data = cache
        self.is_finalised = False

    def __idx__(self, key):
        return self.data[key]

    @torch.no_grad()
    def push(self, row, col, val):
        if not row in self.data.keys():
            self.store[row] = {}
            self.data[row][col] = []

        elif not col in self.data[row].keys():
                self.data[row][col] = []

        self.store[row][col].append(val)

    @torch.no_grad()
    def push_from_cache(self, cache):
        for k, v in cache.data.items():
            if not v[0].device == "cpu":
                v[0] = v[0].cpu()
            self.data[key].append(v[0])

    def finalize(self):
        for row, col in self.data.items():
            for k, v in col.items():
                self.data[row][k] = torch.cat(v, dim=0)
        self.is_finalised = True

    def clear(self):
        self.data = {}

    def print_shapes(self):
        if self.is_finalised:
            for k, v in self.data.items():
                print(k, "---", v.shape)
        else:
            raise Exception("One needs to do `cache.finalise()` before one can do `cache.print_shapes()`.")


class Tracer:
    def __init__(self, model, model_name, config):
        self.model_name = model_name
        self.model = model
        self.config = config
        self.cache = LayerWiseMetricCache()
        self.add_hooks_to_model()

    def get_traced_model_name(self):
        name = f"{self.config.dataset_name}-{self.config.split}_{sel.config.model_name}_{self.config.to_hook}-{self.config.reduction}"
        return name

    def llama_hooks(self, layer_index, metric):
        # separate hook for each metric
        # hooks can't be used for layer dropping of course

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
            ent = shannon_entropy(gram)
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
                for metric in metrics:
                    if metric != "attn_rank":
                        self.model.model.layers[i].register_forward_hook(self.llama_hooks(i, metric))
                    else:
                        self.model.model.layers[i].self_attn.register_forward_hook(self.llama_hooks(i, "attn_rank"))

    def __call__(self, batch):
        return self.model(**batch)


# class HookedModel:
#     def __init__(self, model, model_name, reduction="token_mean", kwargs={}):
#         self.model = model
#         self.model_name = model_name
#         self.num_layers = None

#         assert "to_hook" in kwargs.keys(), "You haven't specified which activations to trace."
#         self.kwargs = kwargs
#         print(f"Hooked activations will be decorrelated when cached: {kwargs['decorrelate']}")

#         self.cache = ActivationCache()

#         assert reduction in ["token_mean", "none", "sample_mean", "token_norm"], "Unsupported reduction type chosen."
#         self.reduction = reduction
#         self.add_hooks_to_model()


#     def get_hooked_model_name(self, args):
#         name = f"{args.dataset_name}-{args.split}_{self.model_name}_{self.kwargs['to_hook']}-{self.reduction}"
#         return name


#     def hook_llama(self, layer_index, hook_subject="output"):
#         def reduced_hook(module, input_tensor, output_tensor):
#             # current shape: (B, L, D), `self.reduction` along an axis (either `L` or `B`) to the cache
#             if hook_subject == "output" and not self.kwargs["decorrelate"]:
#                 value = output_tensor[0].cpu().numpy()

#             elif hook_subject == "input":
#                 value = input_tensor[0].cpu().numpy()

#             elif hook_subject == "output" and self.kwargs["decorrelate"]:
#                 value = (output_tensor[0] - input_tensor[0]).cpu().numpy()

#             if self.reduction == "token_mean":
#                 value = value.mean(axis=1)
#             elif self.reduction == "sample_mean":
#                 value = value.mean(axis=0)
#             elif self.reduction == "token_norm":
#                 value = np.linalg.norm(value, ord=2, axis=1)

#             self.cache.push(f"layer_{layer_index}", value)

#         def tokenwise_hook(module, input_tensor, output_tensor):
#             input_t = F.normalize(input_tensor[0], dim=-1)
#             output_t = F.normalize(output_tensor[0], dim=-1)
#             sim = (input_t * output_t).sum(dim=-1)
#             sim = sim.mean(dim=0).unsqueeze(0)
#             self.cache.push(f"layer_{layer_index}", sim.cpu().numpy())

#         hook_name_map = {
#             "reduced": reduced_hook,
#             "tokenwise": tokenwise_hook
#         }

#         return hook_name_map[self.kwargs["hook_type"]]


#     def add_hooks_to_model(self):
#         if "llama" in self.model_name:
#             self.num_layers = self.model.config.num_hidden_layers

#             for i in range(self.num_layers):
#                 if self.kwargs["to_hook"] == "layer_output":
#                     self.model.model.layers[i].register_forward_hook(self.hook_llama(i, hook_subject="output"))

#                 elif self.kwargs["to_hook"] == "attn":
#                     self.model.model.layers[i].self_attn.register_forward_hook(self.hook_llama(i, hook_subject="output"))

#                 elif self.kwargs["to_hook"] == "pre_mlp":
#                     self.model.model.layers[i].mlp.register_forward_hook(self.hook_llama(i, hook_subject="input"))

#                 elif self.kwargs["to_hook"] == "mlp":
#                     self.model.model.layers[i].mlp.register_forward_hook(self.hook_llama(i, hook_subject="output"))


#     def __call__(self, batch):
#         return self.model(**batch)
