import tyro
import lm_eval
from dataclasses import dataclass
from transformers import AutoModelForCausalLM

from src.utils import get_model, evaluate_model, remove_layers_after


@dataclass
class TestArgs:
    device: str = "cuda"
    batch_size: int = 4
    model_name: str = "llama-1b"
    benchmark: str = "gsm8k_cot_llama" 

def local_test(args):

    # model_path = "/Users/jaisidhsingh/Code/pretrained_llms/Llama-3.2-1B"
    model = get_model(args.model_name, args.device)
    print("Model loaded.")
    N = len(model.model.layers)

    layer_to_eval = N-1 # format: 1 to num_hidden_layers, NOT AN INDEX
    remove_layers_after(layer_to_eval-1, model_to_modify=model)

    results = evaluate_model(model, args, benchmark=args.benchmark)
    print(results)

def lm_eval_tests(args):
    tm = lm_eval.task_manager()
    out = tm.load_task_or_group([args.benchmark])
    print(out[args.benchmark].DATASET_PATH)
    print(out[args.benchmark].DATASET_NAME)


if __name__ == "__main__":
    args = tyro.cli(TestArgs, default=vars(TestArgs()))
    local_test(args)
    # lm_eval_tests(args)
