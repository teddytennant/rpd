Reliable Parallel Decoding (RPD) selects which masked positions to commit in one step (https://arxiv.org/abs/2609.36452).

pip install -e .
python -m pytest

This package does not run an LLaDA or Dream forward pass. The caller supplies per-layer probability distributions for currently masked positions (analysis window already sliced, final layer last). ell_0 is recorded on the presets and is not applied inside the selector. When several vocabulary entries share the maximum probability, argmax takes the lowest index.
