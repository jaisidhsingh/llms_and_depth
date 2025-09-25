import tyro
from tqdm import tqdm
import weightwatcher as ww
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

from src.cli import Args
from src.utils import *
from src.plotting import plot_metrics
from src.tracer import Tracer, LayerWiseMetricCache



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

    bar = tqdm(total=len(loader))

    traced_model = Tracer(model, args.model_name, args)
    print("setup cache for tracking layer-wise metrics.")

    for idx, batch in enumerate(loader):
        batch = cast_batch_to_device(batch, args.device)

        with torch.autocast(args.device):
           outputs = traced_model(batch)

        bar.update(1)
        if idx == 0:
            break


    traced_model.remove_all_hooks()

    if ",ww_alpha" in args.metrics:
        traced_model.attach_weightwatcher()

    traced_model.cache.finalize()
    traced_model.cache.print_shapes()

    return traced_model.cache.data

def compute_eval_metrics(args):
    return {}


def main(args):
    # only non-eval metrics can be computed using the `Tracer` backend
    # with > 1 gpus, there is the option to parallelise metric computation
    # currently, repo is built on single gpu setup
    #
    # eval metrics are designed to be computed across multiple jobs.
    non_eval_args, eval_args = split_args(args)

    # if we are indeed computing non-eval-based metrics
    if non_eval_args is not None:
        print("Computing metrics which don't require evaluation...")
        non_eval_metrics = compute_non_eval_metrics(non_eval_args)

        # make sure we have something saved and plotted at every step
        save_metrics(non_eval_metrics, non_eval_args)
        plot_metrics(non_eval_metrics, non_eval_args)

    # if we're indeed computing eval-based metrics
    if eval_args is not None:
        print("Computing metrics which require evaluation...")
        # this typically takes longer if we're computing the "layer_skip_extrinsic" metric
        eval_metrics = compute_eval_metrics(eval_args)

        # now we have everything so we over-write what we saved and plotted earlier
        metrics = collect_metrics([non_eval_metrics, eval_metrics])
        save_metrics(metrics, args)
        plot_metrics(metrics, args)

    # whew
    print("All done!")


if __name__ == "__main__":
    args = tyro.cli(Args, default=vars(Args()))
    main(args)
