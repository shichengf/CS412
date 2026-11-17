"""Tests for the evaluation metrics in dyncrl/metrics.py."""

import numpy as np

from dyncrl.metrics import mcc, support_precision, weight_correlation


def scrambled_copy(z_true, perm):
    """Permute, sign-flip and rescale the columns of z_true."""
    signs = np.array([1.0, -1.0, -1.0, 1.0])
    scales = np.array([0.5, 2.0, 3.0, 0.1])
    return z_true[:, perm] * signs * scales


def test_mcc_recovers_permutation():
    rng = np.random.default_rng(0)
    z_true = rng.standard_normal((500, 4))
    perm = [2, 0, 3, 1]
    score, match = mcc(z_true, scrambled_copy(z_true, perm))
    assert abs(score - 1.0) < 1e-6
    assert match == {c: t for c, t in enumerate(perm)}


def test_mcc_with_extra_learned_latents():
    rng = np.random.default_rng(0)
    z_true = rng.standard_normal((500, 4))
    perm = [2, 0, 3, 1]
    z_hat = np.concatenate([scrambled_copy(z_true, perm), rng.standard_normal((500, 3))], axis=1)
    score, match = mcc(z_true, z_hat)
    assert score > 0.999
    assert sorted(match.values()) == [0, 1, 2, 3]
    assert all(match[c] == t for c, t in enumerate(perm))


def test_weight_correlation_affine_and_constant_columns():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 50)
    alpha_true = np.stack([t, 1.0 - t, np.full_like(t, 0.3)], axis=1)
    alpha_hat = 2.0 * alpha_true + 0.5
    alpha_hat[:, 2] = rng.standard_normal(len(t))   # ignored, constant in the truth
    assert abs(weight_correlation(alpha_hat, alpha_true) - 1.0) < 1e-6


def test_support_precision_hand_built():
    supports = {0: {0, 1, 2, 3}, 1: {4, 5, 6}}
    match = {0: 0, 1: 1}
    A = np.full((10, 3), 0.01)
    A[[0, 1, 2], 0] = [1.0, 0.9, 0.8]       # concentrated on the true support
    A[:, 1] = 1.0
    A[[4, 5, 6], 1] = 1.1                   # top three in the support, but diffuse
    A[[7, 8, 9], 2] = 1.0                   # concentrated, but latent 2 is unmatched

    assert support_precision(A, supports, [0], match) == 1.0
    assert A[[4, 5, 6], 1].sum() < 0.5 * A[:, 1].sum()
    assert support_precision(A, supports, [1], match) == 0.0
    assert support_precision(A, supports, [2], match) == 0.0

    B = A.copy()
    B[[0, 1, 9], 0] = [1.0, 0.9, 0.8]       # two of the top three in the support
    B[2, 0] = 0.01
    assert abs(support_precision(B, supports, [0], match) - 2 / 3) < 1e-12
    assert abs(support_precision(B, supports, [0, 1, 2], match) - (2 / 3) / 3) < 1e-12
