"""Synthetic benchmark for tracking: a latent process with K atomic mechanisms.

Latents follow a nonlinear autoregression with two lags. All regimes share the
same base transition matrices; regime k adds a fixed amount to one edge of the
lag-1 matrix. A transition between regimes uses the mixed matrix

    W(t) = W_base + sum_k alpha_k(t) * Delta_k

so a pure regime is the special case of a one-hot alpha.
"""

import numpy as np
from scipy.stats import ortho_group

LAGS = 2
SLOPE = 0.2


def _leaky(a):
    return np.where(a > 0, a, a * SLOPE)


def _well_conditioned(n, rng, threshold):
    while True:
        A = rng.uniform(-1, 1, (n, n))
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        if np.linalg.cond(A) <= threshold:
            return A


def make_system(n=8, n_regimes=5, delta=0.5, noise=0.1, seed=0):
    """Draw the base transition matrices, one changed edge per regime, and the mixing."""
    rng = np.random.default_rng(seed)
    conds = []
    for _ in range(2000):
        A = rng.uniform(-1, 1, (n, n))
        conds.append(np.linalg.cond(A / np.linalg.norm(A, axis=0, keepdims=True)))
    threshold = np.percentile(conds, 25)

    edges = rng.permutation(n * n)[:n_regimes]
    deltas = np.zeros((n_regimes, n, n))
    for k, e in enumerate(edges):
        deltas[k, e // n, e % n] = delta

    return {
        "n": n, "n_regimes": n_regimes, "noise": noise,
        "W1": _well_conditioned(n, rng, threshold),
        "W2": _well_conditioned(n, rng, threshold),
        "deltas": deltas,
        "mixing": [ortho_group.rvs(n, random_state=int(rng.integers(1 << 31))) for _ in range(2)],
    }


def mix(system, y):
    """Nonlinear invertible mixing from latents to observations."""
    x = y
    for A in system["mixing"]:
        x = _leaky(x) @ A
    return x


def simulate(system, alpha, batch, rng):
    """Windows of LAGS + 1 frames generated under mixing weights alpha (K,).

    Returns observations x and latents y, both (batch, LAGS + 1, n).
    """
    n = system["n"]
    past = rng.normal(0, 1, (batch, LAGS, n))
    past = (past - past.mean(0, keepdims=True)) / past.std(0, keepdims=True)
    W1 = system["W1"] + np.tensordot(alpha, system["deltas"], axes=1)
    y_t = rng.normal(0, system["noise"], (batch, n))
    y_t = y_t + _leaky(past[:, -1] @ W1) + _leaky(past[:, -2] @ system["W2"])
    y = np.concatenate([past, _leaky(y_t)[:, None]], axis=1)
    return mix(system, y), y


def pure_regimes(system, per_regime=20000, seed=0):
    """Training data: every window comes from a single atomic mechanism.

    Returns x (N, 3, n), y (N, 3, n) and the regime label c (N,).
    """
    rng = np.random.default_rng(seed)
    K = system["n_regimes"]
    xs, ys, cs = [], [], []
    for k in range(K):
        x, y = simulate(system, np.eye(K)[k], per_regime, rng)
        xs.append(x), ys.append(y), cs.append(np.full(per_regime, k))
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            np.concatenate(cs))


def linear_path(steps, regimes, n_regimes):
    """Weights that move through the given regimes one after another.

    At most two regimes are active at any time. Returns alpha (steps, n_regimes).
    """
    alpha = np.zeros((steps, n_regimes))
    segment = steps / (len(regimes) - 1)
    for t in range(steps):
        s = min(int(t / segment), len(regimes) - 2)
        progress = np.clip((t - s * segment) / segment, 0, 1)
        alpha[t, regimes[s]] = 1 - progress
        alpha[t, regimes[s + 1]] = progress
    return alpha


def transition(system, alpha, batch=500, seed=1):
    """Test data along a weight trajectory alpha (T, K).

    Returns x and y of shape (batch, T, 3, n): `batch` windows per time step.
    """
    rng = np.random.default_rng(seed)
    out = [simulate(system, a, batch, rng) for a in alpha]
    x = np.stack([o[0] for o in out], axis=1).astype(np.float32)
    y = np.stack([o[1] for o in out], axis=1).astype(np.float32)
    return x, y
