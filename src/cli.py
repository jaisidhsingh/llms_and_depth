from abc import ABC
from dataclasses import dataclass

from src.utils import ENV_VARS


class BaseArgs(ABC):
    # allow indexing & attribute access
    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class Args(BaseArgs):
    # global settings
    random_seed: int = 0
    device: str = "cuda"
    results_folder = ENV_VARS["RESULTS_FOLDER"]

    # model settings
    model_name: str = "llama-1b"
    to_hook: str = "layer_output"
    hook_type: str = "reduced"
    reduction: str = "token_norm"
    decorrelate: bool = False

    # data settings
    dataset_name: str = "gsm8k-main"
    split: str = "test"
    batch_size: int = 16
    num_workers: int = 8
