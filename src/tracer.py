from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class TracerConfig:
    pass


class ActivationCache:
    def __init__(self, cache={}):
        self.store = cache

    def __idx__(self, key):
        return self.store[key]

    def push(self, key, val):
        self.store[key] = val
    
    def pop(self, key):
        return self.store.pop(key)
    
    def add_cache_keywise(self, cache, dim=0):
        for k, v in cache.store.items():
            current_v = self.store[k]
            self.store[k] = np.concatenate([current_v, v], axis=dim)
    
    def clear(self):
        self.store = {}
    
    def print_shapes(self):
        for k, v in self.store.items():
            print(k, "---", v.shape)

class HookedModel:
    def __init__(self, model, model_name, reduction="token_mean", kwargs={}):
        self.model = model
        self.model_name = model_name
        self.num_layers = None
        self.kwargs = kwargs
        self.cache = ActivationCache()

        assert reduction in ["token_mean", "none", "sample_mean", "token_norm"], "Unsupported reduction type chosen!"
        self.reduction = reduction
        self.add_hooks_to_model()
    
    def hook_llama(self, layer_index):
        def hook(module, input, output):
            # current shape: (B, L, D), `self.reduction` along an axis (either `L` or `B`) to the cache
            value = output[0].cpu().numpy()

            if self.reduction == "token_mean":
                value = value.mean(axis=1)
            elif self.reduction == "sample_mean":
                value = value.mean(axis=0)
            elif self.reduction == "token_norm":
                value = np.linalg.norm(value, ord=2, axis=1)
            
            self.cache.push(f"layer_{layer_index}", value)
        return hook
    
    def hook_recurrent(self, layer_index, is_recurrent_step=False):
        pass
    
    def add_hooks_to_model(self):
        if "llama" in self.model_name:
            self.num_layers = self.model.config.num_hidden_layers

            for i in range(self.num_layers):
                self.model.model.layers[i].register_forward_hook(self.hook_llama(i))
        
        elif "recurrent" in self.model_name:
            self.num_total_layers = self.model.config.n_layers_in_prelude \
                + self.model.config.n_layers_in_recurrent_block * self.kwargs["num_recurrent_steps"] \
                + self.model.config.n_layers_in_coda

            n_p = self.model.config.n_layers_in_prelude
            n_r = self.model.config.n_layers_in_recurrent_block
            n_c = self.model.config.n_layers_in_coda
 
            for i in range(self.model.config.n_layers_in_prelude):
                self.model.transformer.prelude[i].register_forward_hook(self.hook_transformer_layer(i))
            
            for j in range(self.model.config.n_layers_in_recurrent_block):
                self.model.transformer.core_block[j].register_forward_hook(self.hook_transformer_layer(n_p + j))
            
            for k in range(self.model.config.n_layers_in_coda):
                self.model.transformer.coda[k].register_forward_hook(self.hook_transformer_layer(n_p + n_r + k))

    def __call__(self, batch):
        return self.model(**batch) 
