"""Reliable Parallel Decoding (RPD), one-step commitment rule.

Paper: https://arxiv.org/abs/2609.36452

The caller supplies, for currently masked positions, per-layer probability
distributions over the analysis window (final layer last). ``select`` is full
RPD. ``select_block`` is RPD-block. Both default to the LLaDA preset.
"""

from rpd.decode import (
    BETA,
    BLOCK_SIZE,
    K_MAX,
    PRESETS,
    THETA_C,
    THETA_H,
    W,
    Commitment,
    Preset,
    argmax_token,
    candidate_mask,
    confidence_drop,
    dream,
    entropy,
    final_confidence,
    llada,
    persistence,
    select,
    select_block,
    stability_score,
)

__all__ = [
    "BETA",
    "BLOCK_SIZE",
    "K_MAX",
    "PRESETS",
    "THETA_C",
    "THETA_H",
    "W",
    "Commitment",
    "Preset",
    "argmax_token",
    "candidate_mask",
    "confidence_drop",
    "dream",
    "entropy",
    "final_confidence",
    "llada",
    "persistence",
    "select",
    "select_block",
    "stability_score",
]
