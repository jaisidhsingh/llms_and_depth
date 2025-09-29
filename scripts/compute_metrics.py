import tyro
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import DataCollatorWithPadding

from src.utils import *
from src.cli import Args
from src.tracer import Tracer
from src.plotting import plot_metrics


@torch.no_grad()
def compute_non_eval_metrics(args):
    seed_everything(args.random_seed)

    dataset = get_dataset(args.dataset_name, "test", on_colab=args.on_colab)
    tokenizer = get_tokenizer(args.model_name, on_colab=args.on_colab)

    def tokenize_fn(sample):
        tokenized_sample = tokenizer(sample["input_text"], truncation=True)
        return {"input_ids": tokenized_sample.input_ids, "attention_mask": tokenized_sample.attention_mask}

    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model(args.model_name, args.device, on_colab=args.on_colab)

    traced_model = Tracer(model, args.model_name, args)
    print("Setup cache for tracking layer-wise metrics.")

    bar = tqdm(total=len(loader))
    for idx, batch in enumerate(loader):
        batch = cast_batch_to_device(batch, args.device)
        outputs = traced_model(batch)
        
        bar.update(1)
    bar.close()
        
    traced_model.remove_all_hooks()

    if "ww_alpha" in args.metrics:
        print("Computing WeightWatcher metrics per layer.")
        traced_model.attach_weightwatcher()

    traced_model.cache.finalize()
    return traced_model.cache.data


@torch.no_grad()
def compute_eval_metrics(args):
    seed_everything(args.random_seed)

    dataset = get_dataset(args.dataset_name, "test", on_colab=args.on_colab)
    tokenizer = get_tokenizer(args.model_name, on_colab=args.on_colab)

    def tokenize_fn(sample):
        tokenized_sample = tokenizer(sample["input_text"], truncation=True)
        return {"input_ids": tokenized_sample.input_ids, "attention_mask": tokenized_sample.attention_mask}

    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model(args.model_name, args.device, on_colab=args.on_colab)
    model_depth = model.config.num_hidden_layers
    metrics = {}

    assert args.metrics == 'drop_ppl', "Only perplexity with layer skipping is supported currently."
    def drop_ppl_hook(module, inputs, outputs):
        return (inputs[0], *outputs[1:])

    for layer_index in range(model_depth):
        metrics[f'layer_{layer_index}'] = {}
        hook = model.model.layers[layer_index].register_forward_hook(drop_ppl_hook)
        print(f"Hooked model at layer {layer_index+1}")

        running_ppl = 0.0
        bar = tqdm(total=len(loader))
        bar.set_description(f"Skipping layer {layer_index+1}")

        for idx, batch in enumerate(loader):
            batch = cast_batch_to_device(batch, args.device)
            labels = batch["input_ids"]
            
            outputs = model(**batch)

            logits = outputs.logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            vocab = logits.shape[-1]
            
            loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))
            ppl = loss.exp()
            running_ppl += ppl.item()

            bar.update(1)
            bar.set_postfix({"ppl": round(running_ppl / (idx+1), 3)})

        bar.close()
        metrics[f'layer_{layer_index}']['drop_ppl'] = running_ppl / len(loader)
        hook.remove()
        print(f"Done for model layer {layer_index+1}")
        print(" ")
    
    return metrics


@torch.no_grad()
def compute_base_ppl(args):
    seed_everything(args.random_seed)

    dataset = get_dataset(args.dataset_name, "test", on_colab=args.on_colab)
    tokenizer = get_tokenizer(args.model_name, on_colab=args.on_colab)

    def tokenize_fn(sample):
        tokenized_sample = tokenizer(sample["input_text"], truncation=True)
        return {"input_ids": tokenized_sample.input_ids, "attention_mask": tokenized_sample.attention_mask}

    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model(args.model_name, args.device, on_colab=args.on_colab)

    running_ppl = 0.0
    bar = tqdm(total=len(loader))
    bar.set_description(args.model_name)

    for idx, batch in enumerate(loader):
        batch = cast_batch_to_device(batch, args.device)
        labels = batch["input_ids"]
        outputs = model(**batch)

        logits = outputs.logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        vocab = logits.shape[-1]
        
        loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))
        ppl = loss.exp()
        running_ppl += ppl.item()

        bar.update(1)
        bar.set_postfix({"ppl": round(running_ppl / (idx+1), 3)})

    bar.close()
    print(round(running_ppl / len(loader), 3))
    print(" ")


def compute_grad_norm(args):
    seed_everything(args.random_seed)

    dataset = get_dataset(args.dataset_name, "test", on_colab=args.on_colab)
    tokenizer = get_tokenizer(args.model_name, on_colab=args.on_colab)

    def tokenize_fn(sample):
        tokenized_sample = tokenizer(sample["input_text"], truncation=True)
        return {"input_ids": tokenized_sample.input_ids, "attention_mask": tokenized_sample.attention_mask}

    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model(args.model_name, args.device, on_colab=args.on_colab)
    model.train()

    running_ppl = 0.0
    bar = tqdm(total=len(loader))
    bar.set_description(args.model_name)

    metrics = {f"layer_{i}": {"mean_grad_norm": 0} for i in range(model.config.num_hidden_layers)}

    for idx, batch in enumerate(loader):
        batch = cast_batch_to_device(batch, args.device)
        labels = batch["input_ids"]

        model.zero_grad()
        outputs = model(**batch)
        
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        vocab = logits.shape[-1]
        
        loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))
        ppl = loss.exp()
        running_ppl += ppl.item()
        loss.backward()

        for lidx, layer in enumerate(model.model.layers):
            total_norm_sq = 0
            pc = 0
            for param in layer.parameters():
                assert param.grad is not None, "Grad is None."
                # total_norm_sq += param.grad.norm().item() ** 2
                total_norm_sq += param.grad.norm().item()
                pc += 1

            # metrics[f"layer_{lidx}"]['grad_norm'] += total_norm_sq ** 0.5
            metrics[f"layer_{lidx}"]['mean_grad_norm'] += total_norm_sq / pc

        bar.update(1)
    bar.close()
    
    for k, v in metrics.items():
        metrics[k]['mean_grad_norm'] = v['mean_grad_norm'] / len(loader)
    return metrics


def main(args):
    print_intro(args)

    non_eval_args, eval_args = split_args(args)
    non_eval_metrics, eval_metrics = None, None

    # if we are indeed computing non-eval-based metrics
    if non_eval_args is not None:
        print("Computing metrics which don't require evaluation.")
        # FLAG
        non_eval_metrics = compute_grad_norm(non_eval_args)
        # non_eval_metrics = compute_non_eval_metrics(non_eval_args)
        save_metrics(non_eval_metrics, non_eval_args)
        return

    # if we're indeed computing eval-based metrics
    if eval_args is not None:
        print("Computing metrics which require evaluation.")
        # FLAG
        compute_base_ppl(eval_args)
        return
    
        eval_metrics = compute_eval_metrics(eval_args)
        save_metrics(eval_metrics, eval_args)
    
    if eval_metrics is not None and non_eval_metrics is not None:
        metrics = collect_metrics([non_eval_metrics, eval_metrics])
        save_metrics(metrics, args)

    print("All done! \n\n")

if __name__ == "__main__":
    args = tyro.cli(Args, default=vars(Args()))
    main(args)
