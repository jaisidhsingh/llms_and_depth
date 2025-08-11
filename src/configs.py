from types import SimpleNamespace


ROOT = "/fast/jsingh/hf_cache/models"

model_configs = SimpleNamespace(**{})


model_configs.model_name_to_path = {
    "llama-1b": f"{ROOT}/Llama-3.2-1B",
    "lns-llama-1b": f"{ROOT}/LNS_1B",
    "recurrent-1b": f"{ROOT}/huginn-0125"
}