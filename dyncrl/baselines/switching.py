"""Discrete-switching baseline for the tracking readout.

Temporal methods built on discrete regimes assume the system is in exactly one
regime at a time. This baseline gives such a method its best case on our
latents: every latent vector is assigned to the regime whose mean is nearest,
either outright (hard) or with softmax weights over the distances (soft).
Both return a (T, K) matrix with columns in the order of `regimes`, so the
result can be scored like the tracking weights.
"""

import numpy as np


def _sq_distances(z_path, means, regimes):
    """Squared Euclidean distance from every latent vector to every regime mean."""
    M = np.stack([means[k] for k in regimes])            # (K, n)
    return ((np.asarray(z_path)[..., None, :] - M) ** 2).sum(axis=-1)


def hard_assign(z_path, means, regimes):
    """One-hot assignment to the nearest regime mean.

    z_path is (T, n), or (batch, T, n); with a batch every sequence is assigned
    separately and the result is the fraction of sequences in each regime.
    """
    d = _sq_distances(z_path, means, regimes)
    onehot = np.eye(len(regimes))[d.argmin(axis=-1)]
    return onehot.mean(axis=0) if onehot.ndim == 3 else onehot


def soft_assign(z_path, means, regimes, temperature=1.0):
    """Softmax weights over the negative squared distances, averaged over a batch."""
    logits = -_sq_distances(z_path, means, regimes) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=-1, keepdims=True)
    return w.mean(axis=0) if w.ndim == 3 else w
