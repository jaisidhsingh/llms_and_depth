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
    random_seed: int = 123
    device: str = "cuda"
    results_folder = ENV_VARS["RESULTS_FOLDER"]
    config_folder = ENV_VARS["CONFIGS_FOLDER"]
    on_colab: bool = False

    # model settings
    model_name: str = "llama3.2-1b"
    # all metrics = (input_output_cossim, batch_entropy, ww_alpha, drop_ppl)
    metrics: str = "input_output_cossim,batch_entropy,ww_alpha"

    # data settings
    dataset_name: str = "gsm8k-main"
    split: str = "test"
    batch_size: int = 16
    num_workers: int = 8
