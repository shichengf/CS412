"""Tests for the interpretation readout and the double-well benchmark."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dyncrl.data import doublewell
from dyncrl.interpret import AdditiveDecoder, entropy_penalty, overlap_penalty, regime_shift


def make_decoder(z_dim=4, x_dim=6):
    torch.manual_seed(0)
    decoder = AdditiveDecoder(z_dim, x_dim)
    with torch.no_grad():
        decoder.bias.copy_(torch.randn(x_dim))
        decoder.loc.copy_(torch.randn(z_dim))
        decoder.scale.copy_(torch.rand(z_dim) + 0.5)
    return decoder


def test_additive_decoder_forward_is_sum_of_components():
    decoder = make_decoder()
    for shape in [(5, 4), (5, 3, 4)]:
        z = torch.randn(*shape)
        with torch.no_grad():
            out = decoder(z)
            expected = decoder.components(z).sum(-2) + decoder.bias
        assert out.shape == shape[:-1] + (6,)
        assert torch.allclose(out, expected, atol=1e-6)


def test_influence_shape_and_sign():
    decoder = make_decoder()
    with torch.no_grad():
        for mode in ["probe", "weights"]:
            A = decoder.influence(mode)
            assert A.shape == (6, 4)
            assert (A >= 0).all()


def test_entropy_and_overlap_penalties():
    one_hot = torch.eye(5)[:, :3]
    ones = torch.ones(5, 3)
    assert entropy_penalty(one_hot) < entropy_penalty(ones)
    assert overlap_penalty(one_hot).item() < 1e-6
    same = torch.stack([one_hot[:, 0], one_hot[:, 0], one_hot[:, 1]], dim=1)
    assert overlap_penalty(same).item() > 0


def test_regime_shift():
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.standard_normal((n, 4))
    regime = np.repeat([0, 1], n // 2)
    z[regime == 1, 1] += 2.0
    d = regime_shift(z, regime)
    assert abs(d[1] - 2.0) < 0.15
    assert np.abs(np.delete(d, 1)).max() < 0.15


def test_true_influence():
    A = doublewell.true_influence()
    assert A.shape == (30, 6)
    assert np.array_equal(A.sum(axis=0), [len(doublewell.SUPPORTS[j]) for j in range(6)])
    assert not A[27:].any()


def gauss(q, c, s):
    return np.exp(-(q - c) ** 2 / (2 * s ** 2))


def double_well(q, left, right, depth, barrier, sw, sb, ck):
    mid = (left + right) / 2
    return (-depth * (gauss(q, left, sw) + gauss(q, right, sw)) + barrier * gauss(q, mid, sb)
            + ck * (q - mid) ** 4)


def triple_well(q, left, mid, right, depth, barrier, sw, sb, ck):
    wells = gauss(q, left, sw) + gauss(q, mid, sw) + gauss(q, right, sw)
    barriers = gauss(q, (left + mid) / 2, sb) + gauss(q, (mid + right) / 2, sb)
    return -depth * wells + barrier * barriers + ck * (q - (left + right) / 2) ** 4


def potential(q):
    """Total potential of one state q (6,), with the constants of grad_potential."""
    x, z1, z2, z3, u1, u2 = q
    d1, d2, bh = np.clip(z1, 0.1, 3.0), np.clip(z2, 0.1, 3.0), np.clip(z3, 0.05, 3.0)
    u_x = -d1 * gauss(x, -1, 0.45) - d2 * gauss(x, 1, 0.45) + bh * gauss(x, 0, 0.5) + 0.08 * x ** 4
    return (u_x
            + double_well(z1, 0.5, 1.5, 1.0, 0.4, 0.25, 0.3, 0.05)
            + double_well(z2, 0.5, 1.5, 1.0, 0.4, 0.25, 0.3, 0.05)
            + double_well(z3, 0.2, 1.2, 1.0, 0.4, 0.25, 0.3, 0.05)
            + double_well(u1, -1.0, 1.0, 1.0, 0.5, 0.45, 0.5, 0.08)
            + triple_well(u2, -1.0, 0.0, 1.0, 1.0, 0.5, 0.35, 0.25, 0.06))


def test_grad_potential_matches_finite_difference():
    q = np.array([0.3, 1.1, 0.8, 0.6, 0.4, -0.2])   # no clip is active here
    eps = 1e-5
    numeric = np.array([(potential(q + eps * e) - potential(q - eps * e)) / (2 * eps)
                        for e in np.eye(6)])
    analytic = doublewell.grad_potential(q[None])[0]
    assert np.allclose(analytic, numeric, atol=1e-6)
