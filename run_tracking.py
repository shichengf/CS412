"""Tracking experiment on synthetic transitions.

Trains Stage 1 on pure regimes only, then recovers the mixing weights along a
path that moves through three of them.

    python run_tracking.py
"""

import argparse
import os

import numpy as np
import torch

from dyncrl import TemporalVAE, fit_stage1
from dyncrl import tracking
from dyncrl.data import transitions
from dyncrl.metrics import mcc, weight_correlation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regimes", type=int, default=5, help="number of atomic mechanisms K")
    p.add_argument("--active", type=int, default=3, help="regimes visited by the test path")
    p.add_argument("--per-regime", type=int, default=20000, help="training windows per regime")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps", type=int, default=50, help="length of the test path")
    p.add_argument("--window", type=int, default=5, help="half-width of temporal smoothing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/tracking")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    system = transitions.make_system(n_regimes=args.regimes, seed=args.seed)
    x, y, c = transitions.pure_regimes(system, args.per_regime, seed=args.seed)

    # Stage 1 on pure regimes
    vae = TemporalVAE(x_dim=system["n"], z_dim=system["n"], n_regimes=args.regimes, embed_dim=2)
    fit_stage1(vae, x, c, epochs=args.epochs, device=args.device)
    z = vae.encode(x)[:, -1]
    score, _ = mcc(y[:, -1], z)

    # Readout: differential basis from the regime means, then least squares
    means = tracking.regime_means(z, c)
    active = tracking.select_separated(means, args.active)
    basis = tracking.build_basis(means, active)

    alpha_true = transitions.linear_path(args.steps, active, args.regimes)
    x_path, _ = transitions.transition(system, alpha_true, seed=args.seed + 1)
    z_path = vae.encode(x_path)[:, :, -1]                 # (batch, steps, n)
    alpha_raw = tracking.track(z_path, basis)
    alpha_hat = tracking.track(z_path, basis, window=args.window)
    alpha_true = alpha_true[:, active]

    print(f"\nlatent recovery (MCC)            {score:.3f}")
    print(f"path through regimes             {active}")
    print(f"smallest singular value of B     {basis.sigma_min:.3f}")
    print(f"weight correlation, raw          {weight_correlation(alpha_raw, alpha_true):.3f}")
    print(f"weight correlation, smoothed     {weight_correlation(alpha_hat, alpha_true):.3f}")

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "weights.npz"), alpha_true=alpha_true, alpha_raw=alpha_raw,
             alpha_hat=alpha_hat, alpha_calibrated=tracking.calibrate(alpha_hat), regimes=active)
    print(f"saved {args.out}/weights.npz")


if __name__ == "__main__":
    main()
