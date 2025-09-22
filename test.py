from transformers import DataCollatorWithPadding
from tqdm import tqdm
import tyro
from statistics import mean, stdev
from torch.utils.data import DataLoader

from src.utils import *
from src.cli import Args
from src.tracer import *


def main(args):
    seed_everything(args.random_seed)

    dataset = get_dataset(args.dataset_name, "test", on_colab=args.on_colab)
    tokenizer = get_tokenizer(args.model_name, on_colab=args.on_colab)

    def tokenize_fn(sample):
        tokenized_sample = tokenizer(sample["input_text"], truncation=True)
        return {"input_ids": tokenized_sample.input_ids, "attention_mask": tokenized_sample.attention_mask}

    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    model = get_model(args.model_name, args.device, on_colab=args.on_colab)

    bar = tqdm(total=len(loader))

    traced_model = Tracer(model, args.model_name, args)
    print("setup cache for tracking layer-wise metrics.")

    for idx, batch in enumerate(loader):
        batch = cast_batch_to_device(batch, args.device)

        with torch.autocast(args.device):
           outputs = traced_model(batch)

        bar.update(1)
        if idx == 1:
            break

    traced_model.remove_all_hooks()
    traced_model.cache.finalize(reduce=False)
    traced_model.cache.print_shapes()


if __name__ == "__main__":
    args = tyro.cli(Args, default=vars(Args))
    main(args)
