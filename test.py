import tyro
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import evaluate_model, remove_layers_after


@dataclass
class TestArgs:
    device = "mps"
    batch_size = 4

def local_macbook_test(args):

    model_path = "/Users/jaisidhsingh/Code/pretrained_llms/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(args.device)
    model.eval()
    print("Model loaded.")
    print(len(model.model.layers))

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    layer_to_eval = 8 # format: 1 to num_hidden_layers, NOT AN INDEX
    remove_layers_after(layer_to_eval-1, model_to_modify=model)

    results = evaluate_model(model, args, benchmark="gsm8k_cot_zeroshot")
    print(results)


if __name__ == "__main__":
    args = tyro.cli(TestArgs, default=vars(TestArgs()))
    local_macbook_test(args)