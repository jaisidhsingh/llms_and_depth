# the folder `run_experiment` is executable. usage: `sh run_experiment/compute_noneval_metrics.sh`

python3 -m scripts.compute_metrics --model-name llama3.2-1b --batch-size 8 --metrics grad_norm;
python3 -m scripts.compute_metrics --model-name qwen3-0.6b --batch-size 8 --metrics grad_norm;
python3 -m scripts.compute_metrics --model-name qwen3-0.6b-it --batch-size 8 --metrics grad_norm;
python3 -m scripts.compute_metrics --model-name gemma2-2b --batch-size 8 --metrics grad_norm;
python3 -m scripts.compute_metrics --model-name gemma3-1b-pt --batch-size 8 --metrics grad_norm;
python3 -m scripts.compute_metrics --model-name gemma3-1b-it --batch-size 8 --metrics grad_norm;
