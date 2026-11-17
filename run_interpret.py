"""Interpretation experiment on the double-well benchmark.

Stage 1 learns the latents, Stage 2 fits the sparse additive decoder on the
frozen encoder, and the influence matrix is compared with the known supports.

    python run_interpret.py
"""

import argparse
import os

import numpy as np
import torch

from dyncrl import TemporalVAE, fit_stage1
from dyncrl import interpret
from dyncrl.data import doublewell
from dyncrl.metrics import mcc, support_precision


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chains", type=int, default=32)
    p.add_argument("--sim-steps", type=int, default=150_000, help="Langevin steps per chain")
    p.add_argument("--z-dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=80, help="Stage 1 epochs")
    p.add_argument("--epochs2", type=int, default=30, help="Stage 2 epochs")
    p.add_argument("--lam", type=float, default=50.0, help="weight of the sparsity penalty")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/interpret")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    print("simulating the double-well system ...")
    x, y, c = doublewell.generate(args.chains, args.sim_steps, seed=args.seed)
    print(f"{len(x)} windows, {x.shape[-1]} observed variables, {y.shape[-1]} true factors")

    # Stage 1: latents with a dense decoder
    vae = TemporalVAE(x_dim=x.shape[-1], z_dim=args.z_dim, n_regimes=2)
    fit_stage1(vae, x, c, epochs=args.epochs, device=args.device)

    # Stage 2: sparse additive decoder on the frozen encoder
    decoder = interpret.fit_additive_decoder(vae, x, epochs=args.epochs2, lam=args.lam,
                                             device=args.device)
    A = interpret.influence_matrix(decoder)

    z = vae.encode(x)[:, -1]
    score, match = mcc(y[:, -1], z)
    shift = interpret.regime_shift(z, c)
    top = np.argsort(shift)[::-1][:3]
    supports = {j: set(obs) for j, obs in doublewell.SUPPORTS.items()}
    precision = support_precision(A, supports, top, match)

    print(f"\nlatent recovery (MCC)   {score:.3f}")
    print(f"support precision       {precision:.3f}\n")
    print("latent  regime shift  matched factor  most influenced observations  true support")
    for j in top:
        factor = match.get(int(j))
        name = doublewell.FACTORS[factor] if factor is not None else "-"
        truth = sorted(supports.get(factor, []))
        print(f"z{j:<6d} {shift[j]:<13.2f} {name:<15s} {str(interpret.top_features(A, j).tolist()):<29s} {truth}")

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "influence.npz"), influence=A, truth=doublewell.true_influence(),
             regime_shift=shift, match=np.array(sorted(match.items())))
    print(f"\nsaved {args.out}/influence.npz")


if __name__ == "__main__":
    main()
