"""Evaluation metrics shared by both readouts."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def correlation_matrix(z_true, z_hat):
    """|Pearson r| between every true factor (rows) and learned latent (columns)."""
    n_true = z_true.shape[1]
    corr = np.corrcoef(z_true.T, z_hat.T)[:n_true, n_true:]
    return np.nan_to_num(np.abs(corr))


def mcc(z_true, z_hat):
    """Mean correlation coefficient after Hungarian matching.

    Returns the score and the matching {learned latent: true factor}.
    """
    corr = correlation_matrix(z_true, z_hat)
    rows, cols = linear_sum_assignment(-corr)
    return float(corr[rows, cols].mean()), {int(c): int(r) for r, c in zip(rows, cols)}


def weight_correlation(alpha_hat, alpha_true):
    """Mean Pearson correlation between recovered and true mixing trajectories.

    Weights that stay constant along the path carry no signal and are skipped.
    """
    scores = [np.corrcoef(alpha_hat[:, k], alpha_true[:, k])[0, 1]
              for k in range(alpha_true.shape[1]) if alpha_true[:, k].std() > 1e-8]
    return float(np.mean(scores))


def support_precision(A, supports, latents, match, k=3, gate=0.5):
    """Precision of the k most influenced observations against the true support.

    A: influence matrix (x_dim, z_dim). supports: {true factor: set of observed
    indices}. latents: the learned latents to score, normally the three with
    the largest regime shift. match: {learned latent: true factor} from mcc().
    A latent counts only if its top-k entries hold at least `gate` of its
    column mass; a diffuse column scores zero. Returns the mean over `latents`.
    """
    scores = []
    for j in latents:
        top = np.argsort(A[:, j])[::-1][:k]
        concentrated = A[top, j].sum() >= gate * A[:, j].sum()
        truth = supports.get(match.get(int(j)), set())
        scores.append(len(set(top.tolist()) & set(truth)) / k if concentrated else 0.0)
    return float(np.mean(scores))
