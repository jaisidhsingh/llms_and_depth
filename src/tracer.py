from dataclasses import dataclass
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
            print(current_v.shape, v.shape)
            self.store[k] = torch.cat([current_v, v], dim=dim)
    
    def clear(self):
        self.store = {}
    
    def print_shapes(self):
        for k, v in self.store.items():
            print(k, "---", v.shape)

class HookedModel:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.num_layers = None
        self.cache = ActivationCache()
        self.add_hooks_to_model(model, model_name)
    
    def hook_transformer_layer(self, layer_index):
        def hook(module, input, output):
            self.cache.push(f"layer_{layer_index}", output[0])
        return hook
    
    def add_hooks_to_model(self, model, model_name):
        if "llama" in model_name:
            self.num_layers = self.model.config.num_hidden_layers
            for i in range(self.num_layers):
                self.model.model.layers[i].register_forward_hook(self.hook_transformer_layer(i))
        
        else:
            pass

    def __call__(self, batch):
        return self.model(**batch) 