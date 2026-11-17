"""Tests for the tracking readout in dyncrl/tracking.py."""

import numpy as np

from dyncrl.tracking import (build_basis, calibrate, project_simplex, select_separated,
                             smooth, track)


def random_means(rng, k=4, n=8):
    return {r: rng.standard_normal(n) for r in range(k)}


def convex_path(rng, means, regimes, length=40):
    """Latents that are exact convex combinations of the regime means."""
    alpha = rng.dirichlet(np.ones(len(regimes)), size=length)
    M = np.stack([means[r] for r in regimes])
    return alpha, alpha @ M


def test_project_simplex_on_random_input():
    rng = np.random.default_rng(0)
    for _ in range(20):
        raw = rng.standard_normal(4)
        w = project_simplex(raw)
        assert w.shape == (5,)
        assert (w >= 0).all()
        assert abs(w.sum() - 1.0) < 1e-12


def test_project_simplex_keeps_feasible_input():
    rng = np.random.default_rng(0)
    raw = rng.dirichlet(np.ones(4)) * 0.8
    w = project_simplex(raw)
    assert np.allclose(w, np.concatenate([[1.0 - raw.sum()], raw]))


def test_track_exact_recovery():
    rng = np.random.default_rng(0)
    means = random_means(rng)
    regimes = [0, 1, 2, 3]
    basis = build_basis(means, regimes)
    alpha, z = convex_path(rng, means, regimes)
    assert np.abs(track(z, basis) - alpha).max() < 1e-8


def test_track_averages_over_batch():
    rng = np.random.default_rng(0)
    means = random_means(rng)
    regimes = [0, 1, 2, 3]
    basis = build_basis(means, regimes)
    alpha, z = convex_path(rng, means, regimes)
    noise = rng.standard_normal(z.shape)
    batch = np.stack([z + noise, z - noise])
    out = track(batch, basis)
    assert out.shape == alpha.shape
    assert np.abs(out - alpha).max() < 1e-8


def test_smooth():
    rng = np.random.default_rng(0)
    alpha = rng.dirichlet(np.ones(3), size=30)
    assert np.allclose(smooth(alpha, 3).sum(axis=1), 1.0)
    const = np.tile([0.2, 0.3, 0.5], (30, 1))
    assert np.allclose(smooth(const, 3), const)
    assert np.array_equal(smooth(alpha, 0), alpha)


def test_calibrate_maps_extremes():
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0.2, 0.6, size=(30, 3))
    alpha[:, 2] = 0.4
    out = calibrate(alpha)
    assert np.allclose(out[:, :2].min(axis=0), 0.0)
    assert np.allclose(out[:, :2].max(axis=0), 1.0)
    assert np.allclose(out[:, 2], 0.4)


def test_select_separated():
    rng = np.random.default_rng(0)
    means = {r: rng.standard_normal(8) for r in range(6)}
    chosen = select_separated(means, 3)
    assert len(chosen) == 3 and len(set(chosen)) == 3
    assert chosen == sorted(chosen)
    pairs = [(a, b) for a in means for b in means if a < b]
    a, b = max(pairs, key=lambda p: np.linalg.norm(means[p[0]] - means[p[1]]))
    assert a in chosen and b in chosen


def test_sigma_min_positive():
    rng = np.random.default_rng(0)
    basis = build_basis(random_means(rng), [0, 1, 2, 3])
    assert basis.sigma_min > 0
