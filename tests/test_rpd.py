"""Hand-computed checks for the RPD one-step selector.

Expected numbers are closed forms or literals from the commitment rule.
They are not produced by a second copy of the selector.
"""

import math

import jax.numpy as jnp
import pytest

import rpd
from rpd import (
    BETA,
    BLOCK_SIZE,
    K_MAX,
    THETA_C,
    THETA_H,
    W,
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


def _row(values, layers=1):
    row = jnp.asarray(values, dtype=jnp.float32)
    return jnp.stack([row] * layers, axis=0)


def _batch(per_position):
    return jnp.stack([jnp.asarray(item, dtype=jnp.float32) for item in per_position], axis=0)


def _committed(result):
    return result.positions.tolist(), result.token_ids.tolist()


def test_table4_presets_and_shared_thresholds():
    assert THETA_H == 0.9
    assert THETA_C == 0.6
    assert K_MAX == 6
    assert BETA == 4
    assert W == 32
    assert BLOCK_SIZE == 32
    assert llada.theta_s == 3.5
    assert llada.w == 15
    assert llada.ell_0 == 17
    assert dream.theta_s == 2.5
    assert dream.w == 20
    assert dream.ell_0 == 15
    assert rpd.llada is llada
    assert rpd.dream is dream
    # Prose ceilings are (K_max - theta_s) / w, not the weight itself.
    assert (K_MAX - llada.theta_s) / llada.w == pytest.approx(0.16666666666666666)
    assert (K_MAX - dream.theta_s) / dream.w == pytest.approx(0.175)


def test_entropy_one_hot_is_zero_and_half_is_log_two():
    one_hot = jnp.asarray([1.0, 0.0, 0.0])
    assert float(entropy(one_hot)) == 0.0
    assert math.isfinite(float(entropy(one_hot)))

    half = jnp.asarray([0.5, 0.5])
    assert float(entropy(half)) == pytest.approx(math.log(2))

    # 0 log 0 = 0, so an interior zero does not change the two-point entropy.
    with_zero = jnp.asarray([0.5, 0.0, 0.5])
    assert float(entropy(with_zero)) == pytest.approx(math.log(2))
    assert math.isfinite(float(entropy(with_zero)))


def test_persistence_is_four_when_argmax_holds_on_the_last_four_layers():
    # Layer 0 flips to vocab index 2. The last four layers agree on index 0.
    window = jnp.asarray(
        [
            [0.0, 0.1, 0.9],
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.75, 0.2, 0.05],
        ],
        dtype=jnp.float32,
    )
    assert int(persistence(window)) == 4
    assert int(argmax_token(window[-1])) == 0


def test_confidence_drop_and_stability_score():
    # Suffix probabilities of y are 0.8, 0.7, 0.75. r = 0.8 - 0.75 = 0.05.
    # S = min(3, 6) - 15 * 0.05 = 2.25.
    window = jnp.asarray(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.75, 0.25],
        ],
        dtype=jnp.float32,
    )
    assert int(persistence(window)) == 3
    assert float(confidence_drop(window)) == pytest.approx(0.05)
    assert float(stability_score(window, w=15)) == pytest.approx(2.25)
    assert float(stability_score(window, w=dream.w)) == pytest.approx(2.0)


def test_confidence_drop_ignores_layers_before_the_suffix():
    # Final argmax is 0 with probability 0.4, so K = 1 and r = 0.
    # The earlier layer puts 0.45 on token 0 but its argmax is 1, so it is
    # outside the suffix and must not raise the drop.
    window = jnp.asarray(
        [
            [0.45, 0.55, 0.0],
            [0.4, 0.3, 0.3],
        ],
        dtype=jnp.float32,
    )
    assert int(persistence(window)) == 1
    assert float(confidence_drop(window)) == pytest.approx(0.0)


def test_stability_score_caps_persistence_at_k_max():
    window = _row([1.0, 0.0], layers=8)
    assert int(persistence(window)) == 8
    assert float(stability_score(window, w=15)) == pytest.approx(6.0)


def test_argmax_tie_takes_lowest_vocabulary_index():
    tied = jnp.asarray([0.2, 0.4, 0.4])
    assert int(argmax_token(tied)) == 1

    result = select([3], _batch([_row([0.4, 0.4, 0.2])]))
    assert _committed(result) == ([3], [0])


def test_high_confidence_is_a_candidate_even_if_stability_is_terrible():
    # K = 2, r = 0.05, S = min(2, 6) - 15 * 0.05 = 1.25 < 3.5.
    # c = 0.95 >= 0.9, so it is still a candidate.
    window = jnp.asarray([[1.0, 0.0], [0.95, 0.05]], dtype=jnp.float32)
    assert float(stability_score(window, w=15)) == pytest.approx(1.25)
    assert bool(candidate_mask(window, theta_s=llada.theta_s, w=llada.w))

    # Equality at theta_h also bypasses stability (S = 1 < 3.5).
    at_high = _row([0.9, 0.1])
    assert bool(candidate_mask(at_high, theta_s=llada.theta_s, w=llada.w))


def test_confidence_below_theta_c_is_never_a_candidate():
    # Perfect persistence, S = 6 >= 3.5, but c = 0.5 < 0.6.
    window = _row([0.5, 0.5], layers=6)
    assert float(stability_score(window, w=15)) == pytest.approx(6.0)
    assert not bool(candidate_mask(window, theta_s=llada.theta_s, w=llada.w))


def test_intermediate_confidence_needs_stability():
    stable = _row([0.7, 0.3], layers=4)
    unstable = _row([0.7, 0.3], layers=1)
    assert float(stability_score(stable, w=15)) == pytest.approx(4.0)
    assert float(stability_score(unstable, w=15)) == pytest.approx(1.0)
    assert bool(candidate_mask(stable, theta_s=3.5, w=15))
    assert not bool(candidate_mask(unstable, theta_s=3.5, w=15))

    # S == theta_s is admitted. S just below is not.
    assert bool(candidate_mask(stable, theta_s=4.0, w=15))
    assert not bool(candidate_mask(stable, theta_s=4.1, w=15))

    # c == theta_c is in the stability band.
    at_floor = _row([0.6, 0.4], layers=6)
    assert bool(candidate_mask(at_floor, theta_s=3.5, w=15))
    assert not bool(candidate_mask(_row([0.6, 0.4]), theta_s=3.5, w=15))


def test_select_uses_candidate_rule():
    sure = _row([1.0, 0.0], layers=4)
    high_but_unstable = jnp.asarray([[1.0, 0.0], [0.95, 0.05]], dtype=jnp.float32)
    # Pad the sure token to two layers so the batch shares a layer count.
    sure_two = _row([1.0, 0.0], layers=2)
    both = select([0, 1], _batch([high_but_unstable, sure_two]))
    assert _committed(both) == ([0, 1], [0, 0])

    low = _row([0.5, 0.5], layers=6)
    only_sure = select([0, 1], _batch([low, _row([1.0, 0.0], layers=6)]))
    assert _committed(only_sure) == ([1], [0])

    stable = _row([0.7, 0.3], layers=4)
    admitted = select([4, 8], _batch([stable, sure]))
    assert _committed(admitted) == ([4, 8], [0, 0])

    # Final confidence is 0.7, but the argmax flips until the last layer, so K = 1.
    flipped = jnp.asarray(
        [
            [0.2, 0.8],
            [0.2, 0.8],
            [0.2, 0.8],
            [0.7, 0.3],
        ],
        dtype=jnp.float32,
    )
    assert float(final_confidence(flipped)) == pytest.approx(0.7)
    assert int(persistence(flipped)) == 1
    rejected = select([4, 8], _batch([flipped, sure]))
    assert _committed(rejected) == ([8], [0])


def test_right_candidate_waits_when_upstream_entropy_exceeds_beta():
    # Left one-hot has E = 0, so it commits. The uniform position is not a
    # candidate and has entropy log(55) > 4, so the right candidate waits.
    vocab = 55
    left = _row([1.0] + [0.0] * (vocab - 1))
    middle = _row([1.0 / vocab] * vocab)
    right = _row([0.0, 1.0] + [0.0] * (vocab - 2))
    result = select([0, 1, 2], _batch([left, middle, right]))
    assert _committed(result) == ([0], [0])

    # Input order does not change the left-to-right scan.
    shuffled = select([2, 0, 1], _batch([right, left, middle]))
    assert _committed(shuffled) == ([0], [0])


def test_accepted_position_does_not_count_toward_later_entropy():
    # Left is an intermediate candidate (c = 0.65, K = 4, r = 0, S = 4).
    # Its entropy plus the uniform non-candidate exceeds 4, but the uniform
    # entropy alone is log(21) < 4. Excluding the accepted left position lets
    # the right candidate commit. Counting it would make the right one wait.
    rest = 0.35 / 20
    left_row = [0.65] + [rest] * 20
    left = _row(left_row, layers=4)
    middle = _row([1.0 / 21] * 21, layers=4)
    right = _row([1.0] + [0.0] * 20, layers=4)
    result = select([0, 1, 2], _batch([left, middle, right]))
    assert _committed(result) == ([0, 2], [0, 0])


def test_leading_entropy_rejects_candidates_and_fallback_picks_highest():
    # A non-candidate before every candidate has entropy above beta, so the
    # candidate set is empty after the scan. Fallback then takes the highest
    # confidence in the window, which is the right one-hot, not the leftmost
    # candidate.
    vocab = 55
    leading = _row([1.0 / vocab] * vocab)
    mid = _row([0.95, 0.05] + [0.0] * (vocab - 2))
    right = _row([0.0, 0.0, 0.0, 1.0] + [0.0] * (vocab - 4))
    result = select([0, 1, 2], _batch([leading, mid, right]))
    assert _committed(result) == ([2], [3])


def test_zero_budget_accepts_zero_entropy_candidates():
    # E = 0 <= 0, so both one-hots commit. A strict < comparison would reject
    # them and the tie fallback would keep only the leftmost.
    left = _row([1.0, 0.0])
    right = _row([0.0, 1.0])
    result = select([0, 1], _batch([left, right]), beta=0.0)
    assert _committed(result) == ([0, 1], [0, 1])


def test_empty_candidate_set_falls_back_inside_first_w():
    # 33 masked positions, all below theta_c. Among the first W = 32, the
    # highest confidence is 0.55 at canvas indices 2 and 8. Leftmost wins.
    # Index 32 has confidence 0.59 but sits outside the window.
    base = [0.34, 0.33, 0.33]
    rows = [_row(base) for _ in range(33)]
    rows[2] = _row([0.55, 0.25, 0.20])
    rows[8] = _row([0.55, 0.30, 0.15])
    rows[32] = _row([0.59, 0.21, 0.20])
    result = select(list(range(33)), _batch(rows))
    assert _committed(result) == ([2], [0])


def test_fallback_still_runs_when_fewer_than_w_positions_are_masked():
    rows = [
        _row([0.4, 0.3, 0.3]),
        _row([0.45, 0.35, 0.20]),
        _row([0.42, 0.30, 0.28]),
    ]
    result = select([10, 11, 12], _batch(rows))
    assert _committed(result) == ([11], [0])


def test_default_preset_is_llada_and_dream_override_is_accepted():
    # K = 3, r = 0, S = 3. LLaDA theta_s = 3.5 rejects; Dream theta_s = 2.5 admits.
    mid = _row([0.7, 0.3], layers=3)
    sure = _row([0.0, 1.0], layers=3)
    default = select([5, 9], _batch([mid, sure]))
    assert _committed(default) == ([9], [1])

    as_dream = select([5, 9], _batch([mid, sure]), preset=dream)
    assert _committed(as_dream) == ([5, 9], [0, 1])

    by_name = select([5, 9], _batch([mid, sure]), preset="dream")
    assert _committed(by_name) == ([5, 9], [0, 1])

    by_dict = select(
        [5, 9],
        _batch([mid, sure]),
        preset={"theta_s": 2.5, "w": 20.0, "ell_0": 15},
    )
    assert _committed(by_dict) == ([5, 9], [0, 1])

    # An explicit threshold overrides the preset.
    forced = select([5, 9], _batch([mid, sure]), preset=llada, theta_s=3.0)
    assert _committed(forced) == ([5, 9], [0, 1])


def test_default_weight_is_llada_not_dream():
    # K = 6, r = 0.15, c = 0.7.
    # w = 15: S = 6 - 2.25 = 3.75 >= 3.5, candidate.
    # w = 20: S = 6 - 3.0 = 3.0 < 3.5, not a candidate.
    layers = [
        [0.85, 0.15],
        [0.7, 0.3],
        [0.7, 0.3],
        [0.7, 0.3],
        [0.7, 0.3],
        [0.7, 0.3],
    ]
    window = jnp.asarray(layers, dtype=jnp.float32)
    assert int(persistence(window)) == 6
    assert float(confidence_drop(window)) == pytest.approx(0.15)
    assert float(stability_score(window, w=15)) == pytest.approx(3.75)
    assert float(stability_score(window, w=20)) == pytest.approx(3.0)

    sure = _row([1.0, 0.0], layers=6)
    default = select([0, 1], _batch([window, sure]))
    assert _committed(default) == ([0, 1], [0, 0])
    heavier = select([0, 1], _batch([window, sure]), w=20)
    assert _committed(heavier) == ([1], [0])


def test_rpd_block_does_not_commit_a_later_block():
    # Position 0 keeps block 0 active. Position 31 is in that block and is a
    # candidate. Position 32 is a candidate in the next block and must wait.
    mask = _row([0.4, 0.3, 0.3])
    in_block = _row([1.0, 0.0, 0.0])
    later = _row([0.0, 1.0, 0.0])
    result = select_block([0, 31, 32], _batch([mask, in_block, later]))
    assert _committed(result) == ([31], [0])

    # Same canvas under full RPD: entropy of the mask is below beta, and the
    # accepted candidate does not count, so both candidates commit.
    full = select([0, 31, 32], _batch([mask, in_block, later]))
    assert _committed(full) == ([31, 32], [0, 1])


def test_rpd_block_fallback_stays_in_the_active_block():
    # First block still has a mask, so the later-block candidate is not committed.
    mask = _row([0.4, 0.3, 0.3])
    later = _row([1.0, 0.0, 0.0])
    result = select_block([31, 32], _batch([mask, later]))
    assert _committed(result) == ([31], [0])


def test_rpd_block_commits_every_candidate_and_ignores_entropy():
    vocab = 55
    left = _row([1.0] + [0.0] * (vocab - 1))
    noisy = _row([1.0 / vocab] * vocab)
    right = _row([0.0, 1.0] + [0.0] * (vocab - 2))
    blocked = select_block([1, 2, 3], _batch([left, noisy, right]))
    assert _committed(blocked) == ([1, 3], [0, 1])

    # The same positions under full RPD wait on the right candidate.
    full = select([1, 2, 3], _batch([left, noisy, right]))
    assert _committed(full) == ([1], [0])


def test_rpd_block_does_not_add_a_more_confident_non_candidate():
    # c = 0.89 fails stability (K = 1) and is below theta_h. c = 0.7 passes.
    # The block commits the candidate only, not the higher-confidence non-candidate.
    failed = jnp.asarray(
        [
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.89, 0.11],
        ],
        dtype=jnp.float32,
    )
    stable = _row([0.7, 0.3], layers=4)
    result = select_block([1, 2], _batch([stable, failed]))
    assert _committed(result) == ([1], [0])


def test_rpd_block_tie_in_fallback_is_leftmost():
    left = _row([0.45, 0.35, 0.20])
    right = _row([0.45, 0.30, 0.25])
    result = select_block([4, 9], _batch([left, right]))
    assert _committed(result) == ([4], [0])


def test_rpd_block_advances_when_earlier_blocks_have_no_mask():
    token = _row([0.0, 1.0, 0.0])
    result = select_block([40], _batch([token]))
    assert _committed(result) == ([40], [1])


def test_empty_masked_set_commits_nothing():
    empty = jnp.zeros((0, 1, 2), dtype=jnp.float32)
    assert _committed(select([], empty)) == ([], [])
    assert _committed(select_block([], empty)) == ([], [])


def test_caller_window_is_not_sliced_with_ell_0():
    # ell_0 is 17 for LLaDA, but a 4-layer window the caller already sliced
    # still has K = 4. The selector does not drop leading layers itself.
    window = _row([0.8, 0.2], layers=4)
    assert llada.ell_0 == 17
    assert int(persistence(window)) == 4
    result = select([0], _batch([window]))
    assert _committed(result) == ([0], [0])
