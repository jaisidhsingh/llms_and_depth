import tyro
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.cli import Args
from src.utils import *
from src.tracer import Tracer, LayerWiseMetricCache


def get_metrics_from_cache(cache):
    pass

def compute_non_eval_metrics(args):
    tokenizer = get_tokenizer(args.model_name)
    model = get_model(args.model_name, args.device)
    dataset = get_dataset(args.dataset_name, "test")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    bar = tqdm(total=len(loader))

    traced_model = Tracer(model, args.model_name, args)
    main_cache = LayerWiseMetricCache()

    for idx, batch in loader:
        batch = cast_batch_to_device(batch, args.device)

        with torch.autocast(args.device):
           outputs = traced_model(batch)
           del outputs

        main_cache.push_from_cache(traced_model.cache)
        traced_model.cache.clear()

        bar.update(1)

    main_cache.finalize()
    return get_metrics_from_cache(main_cache)


def compute_eval_metrics(args):
    pass

def save_metrics(metrics, args):
    pass

def plot_metrics(metrics, args):
    pass

def collect_metrics(metrics_list):
    tmp = metrics_list[0]
    rows = tmp.keys()
    for item in metrics_list[1:]:
        for k in rows:
            for kk, vv in item[k].items():
                tmp[k][kk] = vv
    return tmp


def main(args):
    # every metric can't be computed using the same `Tracer` backend
    non_eval_args, eval_args = split_args(args)

    # if we are indeed computing non-eval-based metrics
    if non_eval_args is not None:
        # only non-eval metrics can be computed using the `Tracer` backend
        # with > 1 gpus, we can parallelise metric computation
        # currently, repo is built on single gpu setup
        non_eval_metrics = compute_non_eval_metrics(non_eval_args)

        # make sure we have something saved and plotted at every step
        save_metrics(non_eval_metrics, args)
        plot_metrics(non_eval_metrics, args)

    # if we're indeed computing eval-based metrics
    if eval_args is not None:
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
