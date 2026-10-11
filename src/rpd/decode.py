"""One-step Reliable Parallel Decoding selector.

Definitions follow Section 3 and Table 4 of arXiv:2609.36452. Probabilities
are used as given (they are not renormalized). Logarithms are natural, so
entropy is in nats. The convention 0 log 0 = 0 is used.

The prose drop ceilings 0.167 (LLaDA) and 0.175 (Dream) are
(K_max - theta_s) / w. They are not the penalty weight w. Table 4 sets
w to 15 (LLaDA) and 20 (Dream).

ell_0 is the first layer of the paper's analysis window. It is stored on the
preset for documentation only. The selector does not slice layers: the caller
passes the window, with the final layer last.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.special import entr

# Shared Table 4 values.
THETA_H = 0.9
THETA_C = 0.6
K_MAX = 6
BETA = 4.0  # nats
W = 32
BLOCK_SIZE = 32


class Preset(NamedTuple):
    """Model-specific Table 4 settings: theta_s, w, and ell_0."""

    theta_s: float
    w: float
    ell_0: int


llada = Preset(theta_s=3.5, w=15.0, ell_0=17)
dream = Preset(theta_s=2.5, w=20.0, ell_0=15)

PRESETS = {"llada": llada, "dream": dream}


class Commitment(NamedTuple):
    """Positions committed in this step, and the token id y_i at each."""

    positions: jnp.ndarray
    token_ids: jnp.ndarray


def entropy(probs):
    """Shannon entropy in nats along the last axis.

    H(p) = -sum_v p(v) log p(v), with 0 log 0 = 0. A one-hot distribution
    has entropy 0.
    """

    probs = jnp.asarray(probs)
    total = jnp.sum(entr(probs), axis=-1)
    return jnp.where(total == 0, jnp.zeros_like(total), total)


def argmax_token(probs):
    """Greedy token y = argmax_v p(v).

    Ties take the lowest vocabulary index. ``jnp.argmax`` returns the first
    maximum, which is that index.
    """

    return jnp.argmax(jnp.asarray(probs), axis=-1)


def persistence(layer_probs):
    """Prediction persistence K_i.

    Longest suffix of the provided layers whose argmax equals the final-layer
    argmax. The final layer matches itself, so K_i is at least 1.
    ``layer_probs`` is (n_layers, vocab) or (n, n_layers, vocab), final layer last.
    """

    probs, squeeze = _as_batched(layer_probs)
    final_token = argmax_token(probs[:, -1, :])
    layer_token = argmax_token(probs)
    match = layer_token == final_token[:, None]
    # Count trailing matches: first mismatch in the reversed mask, or all layers.
    rev_mismatch = jnp.flip(~match, axis=-1)
    has_mismatch = jnp.any(rev_mismatch, axis=-1)
    first_mismatch = jnp.argmax(rev_mismatch, axis=-1)
    n_layers = probs.shape[1]
    k = jnp.where(has_mismatch, first_mismatch, n_layers)
    return k[0] if squeeze else k


def confidence_drop(layer_probs):
    """Confidence drop r_i over the persistent suffix.

    r_i = max_{layer in the K_i suffix} p_layer(y_i) - p_final(y_i).
    Layers before the suffix do not contribute, even if p(y_i) is larger there.
    """

    probs, squeeze = _as_batched(layer_probs)
    y = argmax_token(probs[:, -1, :])
    k = persistence(probs)
    p_y = jnp.take_along_axis(probs, y[:, None, None], axis=-1)[..., 0]
    n_layers = probs.shape[1]
    layer_index = jnp.arange(n_layers)
    in_suffix = layer_index[None, :] >= (n_layers - k[:, None])
    peak = jnp.max(jnp.where(in_suffix, p_y, -jnp.inf), axis=-1)
    drop = peak - p_y[:, -1]
    return drop[0] if squeeze else drop


def final_confidence(layer_probs):
    """Final-layer confidence c_i = p_final(y_i)."""

    probs, squeeze = _as_batched(layer_probs)
    final = probs[:, -1, :]
    y = argmax_token(final)
    conf = jnp.take_along_axis(final, y[:, None], axis=-1)[:, 0]
    return conf[0] if squeeze else conf


def stability_score(layer_probs, *, w, k_max=K_MAX):
    """Stability score S_i = min(K_i, K_max) - w * r_i."""

    k = persistence(layer_probs)
    drop = confidence_drop(layer_probs)
    capped = jnp.minimum(k, jnp.asarray(k_max, dtype=k.dtype))
    return capped - jnp.asarray(w, dtype=drop.dtype) * drop


def candidate_mask(
    layer_probs,
    *,
    theta_s,
    w,
    theta_h=THETA_H,
    theta_c=THETA_C,
    k_max=K_MAX,
):
    """Boolean candidate mask, Eq. 5.

    A masked position is a candidate when c_i >= theta_h, or when
    theta_c <= c_i < theta_h and S_i >= theta_s. Below theta_c it is never
    a candidate. High confidence bypasses only the stability test.
    """

    probs, squeeze = _as_batched(layer_probs)
    conf = final_confidence(probs)
    score = stability_score(probs, w=w, k_max=k_max)
    theta_h = jnp.asarray(theta_h, dtype=conf.dtype)
    theta_c = jnp.asarray(theta_c, dtype=conf.dtype)
    theta_s = jnp.asarray(theta_s, dtype=score.dtype)
    high = conf >= theta_h
    stable = (conf >= theta_c) & (conf < theta_h) & (score >= theta_s)
    mask = high | stable
    return mask[0] if squeeze else mask


def select(
    positions,
    layer_probs,
    *,
    preset=None,
    theta_h=THETA_H,
    theta_c=THETA_C,
    theta_s=None,
    w=None,
    k_max=K_MAX,
    beta=BETA,
    fallback_window=W,
):
    """One step of full RPD.

    Parameters
    ----------
    positions:
        Canvas indices of currently masked positions, shape (n_masked,).
        Order does not matter; candidates are scanned left to right. Indices
        should be unique.
    layer_probs:
        Distributions for those positions, shape (n_masked, n_layers, vocab).
        The layer axis is the analysis window, final layer last.
    preset:
        ``llada`` (default), ``dream``, a preset name, or a mapping with
        ``theta_s`` and ``w``. Explicit ``theta_s`` or ``w`` overrides the preset.
    beta:
        Cumulative entropy budget in nats. A candidate i is accepted when
        E_i <= beta.
    fallback_window:
        If nothing is accepted, commit the highest-confidence position among
        the first this many masked positions from the left. Ties take the
        leftmost. The fallback may commit a non-candidate.

    Returns
    -------
    Commitment
        Committed canvas indices and their final-layer token ids y_i, in
        left-to-right order. The canvas is not modified.
    """

    positions, layer_probs = _prepare(positions, layer_probs)
    if positions.shape[0] == 0:
        return _empty(positions)

    theta_s, w = _resolve_stability(preset, theta_s, w)
    tokens = argmax_token(layer_probs[:, -1, :])
    conf = final_confidence(layer_probs)
    ent = entropy(layer_probs[:, -1, :])
    cand = candidate_mask(
        layer_probs,
        theta_s=theta_s,
        w=w,
        theta_h=theta_h,
        theta_c=theta_c,
        k_max=k_max,
    )
    accepted = _accepted_indices(cand, ent, beta)
    if not accepted:
        accepted = [_fallback_index(conf, fallback_window)]
    return _gather(positions, tokens, accepted)


def select_block(
    positions,
    layer_probs,
    *,
    preset=None,
    theta_h=THETA_H,
    theta_c=THETA_C,
    theta_s=None,
    w=None,
    k_max=K_MAX,
    block_size=BLOCK_SIZE,
):
    """One step of RPD-block.

    The active block is the leftmost origin-0 span of ``block_size`` canvas
    indices that still contains a masked position. Block k is the half-open
    interval [k * block_size, (k + 1) * block_size). Candidate selection uses
    the same rule as full RPD, but only inside the active block, and there is
    no entropy budget: every candidate in that block is committed. If the
    block has no candidate, the most confident masked position in the block
    is committed (leftmost on a tie). Later blocks are not committed in the
    same step.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    positions, layer_probs = _prepare(positions, layer_probs)
    if positions.shape[0] == 0:
        return _empty(positions)

    theta_s, w = _resolve_stability(preset, theta_s, w)
    tokens = argmax_token(layer_probs[:, -1, :])
    conf = final_confidence(layer_probs)
    cand = candidate_mask(
        layer_probs,
        theta_s=theta_s,
        w=w,
        theta_h=theta_h,
        theta_c=theta_c,
        k_max=k_max,
    )
    block_ids = positions // int(block_size)
    active = int(jnp.min(block_ids))
    in_block = [int(block_ids[i]) == active for i in range(block_ids.shape[0])]
    accepted = [i for i, inside in enumerate(in_block) if inside and bool(cand[i])]
    if not accepted:
        accepted = [_block_fallback(conf, in_block)]
    return _gather(positions, tokens, accepted)


def _as_batched(layer_probs):
    probs = jnp.asarray(layer_probs)
    if probs.ndim == 2:
        return probs[None, ...], True
    if probs.ndim != 3:
        raise ValueError(
            "layer_probs must have shape (n_layers, vocab) or "
            "(n_masked, n_layers, vocab)"
        )
    if probs.shape[1] < 1 or probs.shape[2] < 1:
        raise ValueError("layer window and vocabulary must be non-empty")
    return probs, False


def _prepare(positions, layer_probs):
    positions = jnp.asarray(positions)
    layer_probs = jnp.asarray(layer_probs)
    if positions.ndim != 1:
        raise ValueError("positions must be a 1-d array of canvas indices")
    if layer_probs.ndim != 3:
        raise ValueError(
            "layer_probs must have shape (n_masked, n_layers, vocab), "
            "final layer last"
        )
    if positions.shape[0] != layer_probs.shape[0]:
        raise ValueError(
            "positions and layer_probs must cover the same masked positions, "
            f"got {positions.shape[0]} and {layer_probs.shape[0]}"
        )
    if layer_probs.shape[1] < 1 or layer_probs.shape[2] < 1:
        raise ValueError("layer window and vocabulary must be non-empty")
    if not jnp.issubdtype(positions.dtype, jnp.integer):
        positions = positions.astype(jnp.int32)
    if positions.shape[0] == 0:
        return positions, layer_probs
    order = jnp.argsort(positions)
    return positions[order], layer_probs[order]


def _resolve_stability(preset, theta_s, w):
    if preset is None:
        preset = llada
    elif isinstance(preset, str):
        key = preset.lower()
        try:
            preset = PRESETS[key]
        except KeyError as exc:
            known = ", ".join(sorted(PRESETS))
            raise ValueError(f"unknown preset {preset!r}; expected one of {known}") from exc
    if isinstance(preset, Mapping):
        base_theta_s = preset["theta_s"]
        base_w = preset["w"]
    else:
        base_theta_s = preset.theta_s
        base_w = preset.w
    if theta_s is None:
        theta_s = base_theta_s
    if w is None:
        w = base_w
    return float(theta_s), float(w)


def _accepted_indices(cand, ent, beta):
    """Left-to-right scan. E_i sums final-layer entropy of preceding masked
    positions that are not already accepted this step (Eq. 6). Rejected
    candidates and non-candidates both count. Accept when E_i <= beta.
    """

    n = int(cand.shape[0])
    cand_py = [bool(cand[i]) for i in range(n)]
    ent_py = [float(ent[i]) for i in range(n)]
    beta = float(beta)
    accepted = []
    accepted_set = set()
    for i in range(n):
        if not cand_py[i]:
            continue
        upstream = 0.0
        for j in range(i):
            if j not in accepted_set:
                upstream += ent_py[j]
        if upstream <= beta:
            accepted.append(i)
            accepted_set.add(i)
    return accepted


def _fallback_index(conf, window):
    limit = min(int(window), int(conf.shape[0]))
    if limit < 1:
        raise ValueError("no masked position in the fallback window")
    # argmax returns the first maximum, which is the leftmost after sorting.
    return int(jnp.argmax(conf[:limit]))


def _block_fallback(conf, in_block):
    best = None
    best_c = None
    for i, inside in enumerate(in_block):
        if not inside:
            continue
        value = float(conf[i])
        if best is None or value > best_c:
            best = i
            best_c = value
    if best is None:
        raise ValueError("active block has no masked position")
    return best


def _empty(positions):
    pos_dtype = positions.dtype if jnp.issubdtype(positions.dtype, jnp.integer) else jnp.int32
    return Commitment(
        jnp.zeros((0,), dtype=pos_dtype),
        jnp.zeros((0,), dtype=jnp.int32),
    )


def _gather(positions, tokens, indices):
    idx = jnp.asarray(indices, dtype=jnp.int32)
    return Commitment(positions[idx], tokens[idx])
