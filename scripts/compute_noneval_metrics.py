import tyro
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.cli import Args
from src.utils import *
from src.tracer import Tracer, LayerWiseMetricCache



def compute_non_eval_metrics(args):
    tokenizer = get_tokenizer(args.model_name, on_colab=args.on_colab)
    print(f"loaded tokenizer for {args.model_name}.")
    return

    model = get_model(args.model_name, args.device, on_colab=args.on_colab)
    print(f"loaded model {args.model_name}.")
    print(model)

    dataset = get_dataset(args.dataset_name, "test", on_colab=args.on_colab)
    print(f"{args.dataset_name} of {len(dataset)} samples loaded.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    bar = tqdm(total=len(loader))

    traced_model = Tracer(model, args.model_name, args)
    main_cache = LayerWiseMetricCache()
    print("setup cache for tracking layer-wise metrics.")

    for idx, batch in loader:
        batch = cast_batch_to_device(batch, args.device)

        with torch.autocast(args.device):
           outputs = traced_model(batch)
           del outputs

        main_cache.push_from_cache(traced_model.cache)
        traced_model.cache.clear()

        bar.update(1)

    traced_model.remove_all_hooks()
    main_cache.finalize()
    main_cache.print_shapes()

    return main_cache.data


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
        non_eval_metrics = compute_non_eval_metrics(non_eval_args)

    #     # make sure we have something saved and plotted at every step
    #     save_metrics(non_eval_metrics, non_eval_args)
    #     plot_metrics(non_eval_metrics, non_eval_args)

    # # if we're indeed computing eval-based metrics
    # if eval_args is not None:
    #     # this typically takes longer if we're computing the "layer_skip_extrinsic" metric
    #     eval_metrics = compute_eval_metrics(eval_args)

    #     # now we have everything so we over-write what we saved and plotted earlier
    #     metrics = collect_metrics([non_eval_metrics, eval_metrics])
    #     save_metrics(metrics, args)
    #     plot_metrics(metrics, args)

    # whew
    print("All done!")


if __name__ == "__main__":
    args = tyro.cli(Args, default=vars(Args()))
    main(args)
