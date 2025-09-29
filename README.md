# Analysis of LLMs along their depth
Jaisidh Singh

## TLDR

- LLMs show increased layer "redundancy" along their depth.
- Redundancy (in current literature) mainly points to
    - the impact of skipping a layer in the forward pass
    - how similar layer-wise representations are

- Pre-LN transformers exhibit 
    - reduced grad-norms in deeper layers
    - lower performance drops when skipping deeper layers

- Post-LN transformers exhibit
    - increased grad-norms in deeper layers
    - lower performance drops when skipping early layers

- Mixed-LN transformers:
    - uniform grad-norms across depth
    - uniform performance drops upon skipping layers acros depth

- Redundancy actually seems like how "well-trained" a layer is

## Experiment 1
- Idea: quantify "well-trained" using HTSR Theory (power-law fit of the ESD of eigenvalues of weight matrices).
- Theory: I've written a small proof, that relates the variance of $x_l$ with the Frobenius norm of the weight matrices in layer $l$. 
- Why: confirms our hypothesis & backs our proof
- Result:
    - Pre-LN transformers have more "untrained" weight matrices along depth
    - Mixed-LN transformers have less "untrained" weight matrices along depth

## Experiment 2
- Idea: since Mix-LN checkpoints aren't released, compute grad-norm across depth for Gemma3
- Why: confirms connection of (LayerNorms + grad-norms) to "well-trained"-ness 
- Result:
    - TODO

## Experiment 3
- Idea: compute $\Delta$-perplexity found by skipping layers across depth
- Why: fully rounded redundancy quantification according to current literature
- Result:
    - TODO
