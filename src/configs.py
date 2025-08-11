from types import SimpleNamespace


MODELS_ROOT = "/fast/jsingh/hf_cache/models"
DATA_ROOT = "/fast/jsingh/data"

model_configs = SimpleNamespace(**{})
model_configs.model_name_to_path = {
    "llama-1b": f"{MODELS_ROOT}/Llama-3.2-1B",
    "lns-llama-1b": f"{MODELS_ROOT}/LNS_1B",
    "recurrent-1b": f"{MODELS_ROOT}/huginn-0125"
}

data_configs = SimpleNamespace(**{})
data_configs.dataset_to_path = {
    "gsm8k-main": f"{DATA_ROOT}/gsm8k-saved/gsm8k-main",
    "gsm8k-socratic": f"{DATA_ROOT}/gsm8k-saved/gsm8k-socratic",
    "c4-10k": f"{DATA_ROOT}/c4-10k-saved"
}