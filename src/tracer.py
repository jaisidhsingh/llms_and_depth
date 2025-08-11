from dataclasses import dataclass
import torch


@dataclass
class TracerConfig:
    pass


class ActivationCache:
    def __init__(self, cache={}):
        self.store = cache
    
    def push(self, key, val):
        self.store[key] = val
    
    def pop(self, key):
        return self.store.pop(key)
    
    def add_cache(self, cache, dim=0):
        for k, v in cache.items():
            current_v = self.store[k]
            self.store[k] = torch.cat([current_v, v], dim=dim)


class ActivationTracer:
    def __init__(self, model):
        self.model = model
        self.cache = ActivationCache()
