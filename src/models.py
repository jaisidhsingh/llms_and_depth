from transformers import AutoModelForCausalLM, AutoTokenizer
from src.configs import model_configs


def get_tokenizer(model_name):
    model_path = model_configs.model_name_to_path[model_name]
    return AutoTokenizer.from_pretrained(model_path)

def get_model(model_name, device):
    if device is None:
        device = "auto"
    
    model_path = model_configs.model_name_to_path[model_name]
    return AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
