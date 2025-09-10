# Analysis of LLMs along their depth
Jaisidh Singh

### Depth-wise representation similarity

- [ ] cosine similarity - only residual
- [ ] cosine similarity - random inputs
- [ ] cosine similarity - rand. init
- [ ] cosine similarity - de-correlated
- [ ] cosine similarity - factual data
- [ ] cosine similarity - gsm8k

### Depth-wise eigenspectrum analysis

- [ ] eigenspectrum difference - non-ln-scaling model
- [ ] eigenspectrum difference - ln-scaling model

### Depth-wise impacts

- [ ] influence on ntp - non-ln-scaling model
- [ ] influence on ntp - ln-scaling model

### Diganta's ideas

- [ ] Test time mixture of depth
- [ ] report performance metrics on GSM-8K

## Questions

- Is representation similarity a good indicator of layer-wise performance (across all architectures?)
- Whether CoT/RLHF/SFT fine-tuning alters internal data-structures in terms of syntax tree?

- Does LNS fix the second half low contribution effects as per RC?
- Same for recurrent models.
- Do representations of different eigenspectra lead to different confidences?
- Take HH dataset and get CoT traces from a big aligned model, then fine-tune Pythia on it.

- Does LNS show similar entropy + layer-wise sim/perf?
- Does LNS affect the discrete de-tokenization stage?

## Forward propagation with residuals
```
x = input_embeds
layer_1_out = x + f1(x) + g1(x + f1(x))
decorrelated_layer_1_out = f1(x) + g1(x + f1(x))

layer_2_out =  x + f1(x) + g1(x + f1(x)) + f2(x + f1(x) + g1(x + f1(x))) + g2(x + f1(x) + g1(x + f1(x)) + f2(x + f1(x) + g1(x + f1(x))))
decorrelated_layer_2_out = f2(x + f1(x) + g1(x + f1(x))) + g2(x + f1(x) + g1(x + f1(x)) + f2(x + f1(x) + g1(x + f1(x))))

decorrelated_layer_i_out = layer_i_out - layer_i_inp
```