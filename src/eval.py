import torch.nn.functional as F
import lm_eval
from lm_eval.models.huggingface import HFLM


# eval utils
def extrinsic_eval(model, args, benchmark):
    lm = HFLM(model, device=args.device, batch_size=args.batch_size)
    print(f"Wrapped model in HFLM as desired by `lm_eval`. Note: only single process evaluation on device=`{lm._device}`")

    task_manager = lm_eval.task_manager()

    fewshot_as_multiturn = False
    apply_chat_template = False

    if "llama" in benchmark:
        fewshot_as_multiturn = True

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[benchmark],
        task_manager=task_manager,
        fewshot_as_multiturn=fewshot_as_multiturn,
        apply_chat_template=apply_chat_template
    )
    return results

def intrinsic_eval(logits, labels):
    # chop of the last position in the raw logits
    logits = logits[:, :-1, :]
    # right shift the labels
    labels = labels[:, 1:]

    loss = F.cross_entropy(logits, labels)
    perplexity = loss.exp()
    return loss.item(), perplexity.item()
