"""Readout 2: interpreting the latent that changes with the regime.

The Stage 1 encoder is frozen and the dense decoder is replaced by an additive
one, x = sum_j f_j(z_j) + b. Each f_j is then the main effect of latent j, and
its influence on observation i is measured by probing f_j at two points:

    A[i, j] = | f_j(+1)_i - f_j(-1)_i |

An entropy penalty on every column of A makes each latent act on few observed
variables. The latent with the largest shift between regimes is then named by
the observations it influences most.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from .model import make_loader


class AdditiveDecoder(nn.Module):
    """x = sum_j f_j(z_j) + b with one small network f_j: R -> R^D per latent."""

    def __init__(self, z_dim, x_dim, hidden_per_z=4, slope=0.2):
        super().__init__()
        self.z_dim, self.x_dim = z_dim, x_dim
        self.w1 = nn.Parameter(torch.randn(z_dim, hidden_per_z) * math.sqrt(2.0 / (1 + slope ** 2)))
        self.b1 = nn.Parameter(torch.zeros(z_dim, hidden_per_z))
        self.w2 = nn.Parameter(torch.randn(z_dim, x_dim, hidden_per_z)
                               * math.sqrt(2.0 / (hidden_per_z * z_dim * (1 + slope ** 2))))
        self.bias = nn.Parameter(torch.zeros(x_dim))
        self.act = nn.LeakyReLU(slope)
        # latents are standardised first, so the probe points +1 and -1 are one
        # standard deviation away from the mean of each latent
        self.register_buffer("loc", torch.zeros(z_dim))
        self.register_buffer("scale", torch.ones(z_dim))

    def components(self, z, standardised=False):
        """Contribution of every latent before summing: (..., z_dim, x_dim)."""
        if not standardised:
            z = (z - self.loc) / self.scale
        h = self.act(z.unsqueeze(-1) * self.w1 + self.b1)
        return torch.einsum("...zh,zoh->...zo", h, self.w2)

    def forward(self, z):
        return self.components(z).sum(dim=-2) + self.bias

    def influence(self, mode="probe"):
        """Influence matrix A of shape (x_dim, z_dim).

        "probe" is the functional definition |f_j(+1) - f_j(-1)|. "weights" uses
        the norm of the output weights of f_j onto each observation, which is a
        smoother surrogate for training.
        """
        if mode == "weights":
            return self.w2.norm(dim=-1).T
        ones = torch.ones(1, self.z_dim, device=self.bias.device)
        return (self.components(ones, True) - self.components(-ones, True)).abs().squeeze(0).T


def entropy_penalty(A):
    """Mean entropy of the columns of A after normalising each to sum to one.

    Latents whose total influence is negligible are left out so that the
    penalty cannot be lowered by switching a latent off.
    """
    mass = A.sum(dim=0)
    alive = mass > 0.01 * mass.max()
    p = A[:, alive] / (mass[alive] + 1e-8)
    return -(p * torch.log(p + 1e-8)).sum(dim=0).mean()


def overlap_penalty(A):
    """Mean squared cosine similarity between the columns of A.

    Keeps two latents from claiming the same group of observations.
    """
    cols = A / (A.norm(dim=0, keepdim=True) + 1e-8)
    sim = cols.T @ cols
    n = sim.shape[0]
    return (sim - torch.diag(torch.diag(sim))).pow(2).sum() / (n * (n - 1))


def fit_additive_decoder(vae, x, epochs=30, hidden_per_z=4, lam=50.0, ramp_epochs=10,
                         overlap=1.0, penalty_on="weights", batch_size=256, lr=5e-4,
                         device="cpu", log=print):
    """Stage 2: fit a sparse additive decoder on top of the frozen encoder.

    The encoder and the transition prior do not change, so the KL terms of the
    ELBO are constant and only the reconstruction term and the sparsity penalty
    remain. Returns the trained AdditiveDecoder.
    """
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    decoder = AdditiveDecoder(vae.z_dim, vae.x_dim, hidden_per_z)
    z_all = torch.as_tensor(vae.encode(x)).reshape(-1, vae.z_dim)
    decoder.loc.copy_(z_all.mean(0))
    decoder.scale.copy_(z_all.std(0) + 1e-6)
    decoder.to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    loader = make_loader(x, np.zeros(len(x)), batch_size)

    for epoch in range(epochs):
        weight = lam * min(1.0, epoch / max(ramp_epochs, 1))
        total, n = 0.0, 0
        for xb, _ in loader:
            xb = xb.to(device)
            with torch.no_grad():
                z, _, _ = vae.sample(xb)
            recon = vae.recon_loss(xb, decoder(z))
            A = decoder.influence(penalty_on)
            penalty = entropy_penalty(A) + overlap * overlap_penalty(A)
            loss = recon + weight * penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += recon.item()
            n += 1
        if log is not None:
            with torch.no_grad():
                ent = entropy_penalty(decoder.influence("probe")).item()
            log(f"[stage 2] epoch {epoch + 1:3d}/{epochs}  recon {total / n:.3f}  "
                f"column entropy {ent:.3f}  lambda {weight:.1f}")
    return decoder


def influence_matrix(decoder):
    """The probed influence matrix as a numpy array (x_dim, z_dim)."""
    with torch.no_grad():
        return decoder.influence("probe").cpu().numpy()


def regime_shift(z, regime):
    """Cohen's d of every latent between two regimes. z: (N, n), regime in {0, 1}."""
    regime = np.asarray(regime).reshape(-1)
    a, b = z[regime == 0], z[regime == 1]
    pooled = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2 + 1e-10)
    return np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled


def top_features(A, latent, k=3):
    """Indices of the k observations that the given latent influences most."""
    return np.argsort(A[:, latent])[::-1][:k]
