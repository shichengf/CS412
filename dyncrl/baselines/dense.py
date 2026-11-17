"""Influence matrix read directly from the dense Stage 1 decoder.

A dense decoder has no separate main effect per latent, so its influence is
probed around real data points instead: every sampled latent vector is moved
by one standard deviation along latent j in each direction, and the absolute
change of the decoded observations is averaged over the samples. This is the
same one-standard-deviation probe that the additive decoder uses. The
thresholded variant then keeps only the largest entries of every column.
"""

import numpy as np
import torch


def dense_influence(vae, x, n_samples=2000, seed=0):
    """Influence matrix (x_dim, z_dim) of a trained TemporalVAE on windows x (N, T, x_dim)."""
    z = vae.encode(x).reshape(-1, vae.z_dim)
    s = z.std(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z), min(n_samples, len(z)), replace=False)
    device = next(vae.parameters()).device
    base = torch.as_tensor(z[idx], dtype=torch.float32, device=device)

    vae.eval()
    A = np.zeros((vae.x_dim, vae.z_dim))
    with torch.no_grad():
        for j in range(vae.z_dim):
            step = torch.zeros(vae.z_dim, device=device)
            step[j] = float(s[j])
            diff = vae.decoder(base + step) - vae.decoder(base - step)
            A[:, j] = diff.abs().mean(dim=0).cpu().numpy()
    return A


def threshold_influence(A, quantile=0.8):
    """Set the entries of every column below that column's quantile to zero."""
    cut = np.quantile(A, quantile, axis=0)
    return np.where(A < cut, 0.0, A)
