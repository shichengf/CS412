"""Readout 1: tracking a mechanism transition.

During a transition the mechanism is a convex combination of the atomic
mechanisms seen in training, W(t) = sum_k alpha_k(t) W^(k). In latent space the
conditional mean then moves between the regime means, so the mixing weights can
be read out by least squares:

    alpha(t) = Proj_simplex( B^+ (z_t - mu_0) ),   B = [mu_1 - mu_0, ..., mu_{K-1} - mu_0]

Nothing is trained here. The functions take latents from the frozen Stage 1
encoder and return a weight trajectory.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MechanismBasis:
    """Baseline mean and the shifts of the other regimes relative to it."""

    regimes: list          # regimes[0] is the baseline
    mu0: np.ndarray        # (n,)
    B: np.ndarray          # (n, K-1)

    @property
    def sigma_min(self):
        """Smallest singular value of B. The recovery error scales with 1 / sigma_min."""
        return float(np.linalg.svd(self.B, compute_uv=False).min())


def regime_means(z, regime):
    """Mean latent of every regime. z: (N, n), regime: (N,)."""
    regime = np.asarray(regime).reshape(-1)
    return {int(k): z[regime == k].mean(axis=0) for k in np.unique(regime)}


def select_separated(means, k):
    """Greedily pick k regimes whose means are far apart (a well-conditioned basis)."""
    keys = sorted(means)
    if k >= len(keys):
        return keys
    dist = {(a, b): float(np.linalg.norm(means[a] - means[b])) for a in keys for b in keys}
    chosen = list(max(((a, b) for a in keys for b in keys if a < b), key=dist.get))
    while len(chosen) < k:
        rest = [a for a in keys if a not in chosen]
        chosen.append(max(rest, key=lambda a: min(dist[(a, b)] for b in chosen)))
    return sorted(chosen)


def build_basis(means, regimes):
    """Differential basis for the given regimes; the first one is the baseline."""
    mu0 = means[regimes[0]]
    B = np.stack([means[k] - mu0 for k in regimes[1:]], axis=1)
    return MechanismBasis(list(regimes), mu0, B)


def project_simplex(raw):
    """Map K-1 unconstrained weights to K weights on the simplex.

    The baseline weight is the remainder 1 - sum(raw); negative entries are
    clipped and the result is renormalised.
    """
    full = np.concatenate([[1.0 - raw.sum()], raw])
    full = np.clip(full, 0.0, None)
    total = full.sum()
    return full / total if total > 1e-10 else np.full_like(full, 1.0 / len(full))


def solve_weights(z_t, basis):
    """Least-squares mixing weights for one latent vector z_t (n,)."""
    raw = np.linalg.pinv(basis.B) @ (z_t - basis.mu0)
    return project_simplex(raw)


def smooth(alpha, window):
    """Moving average over time with half-width `window`. alpha: (T, K)."""
    if window <= 0:
        return alpha
    out = np.empty_like(alpha)
    for t in range(len(alpha)):
        out[t] = alpha[max(0, t - window): t + window + 1].mean(axis=0)
    return out


def calibrate(alpha, lo=0.0, hi=1.0):
    """Rescale each weight so that its extremes along the path map to [lo, hi].

    The raw weights have the right shape but a compressed scale, because the
    regime means are estimated through a nonlinear encoder. Fixing the two
    endpoints removes that distortion.
    """
    out = alpha.copy()
    for k in range(alpha.shape[1]):
        a, b = alpha[:, k].min(), alpha[:, k].max()
        if b - a > 1e-8:
            out[:, k] = (alpha[:, k] - a) / (b - a) * (hi - lo) + lo
    return out


def track(z_path, basis, window=0):
    """Recover the weight trajectory along a transition.

    z_path is (T, n), or (batch, T, n) when several sequences share the same
    path; in that case the latents are averaged over the batch first.
    Returns alpha (T, K) in the order of basis.regimes.
    """
    z_path = np.asarray(z_path)
    if z_path.ndim == 3:
        z_path = z_path.mean(axis=0)
    alpha = np.stack([solve_weights(z_t, basis) for z_t in z_path])
    return smooth(alpha, window)
