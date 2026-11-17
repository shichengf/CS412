"""Plot the influence matrices saved by run_interpret.py.

One heat map per matrix, stacked over the shared axis of the 30 observed
variables: the ground truth, the additive decoder and, if present, the dense
decoder. Rows are the true factors; for a learned matrix each row shows the
latent matched to that factor, divided by its maximum.

    python plot_influence.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dyncrl.data import doublewell

CMAP = LinearSegmentedColormap.from_list("influence", ["#f7f9fc", "#0b3a75"])


def factor_rows(A, match, n_factors):
    """Rows (factors, observations) of A taken from the latent matched to each factor."""
    by_factor = {int(f): int(j) for j, f in match}
    rows = np.full((n_factors, A.shape[0]), np.nan)
    for f, j in by_factor.items():
        rows[f] = A[:, j]
    return rows


def normalise_rows(M):
    """Divide every row by its maximum; empty rows stay empty."""
    peak = np.nan_to_num(M).max(axis=1, keepdims=True)
    return M / np.where(peak > 0, peak, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/interpret/influence.npz")
    p.add_argument("--output", default="results/interpret/influence.pdf")
    args = p.parse_args()

    data = np.load(args.input)
    n_factors = len(doublewell.FACTORS)
    panels = [("ground truth", data["truth"].T),
              ("additive decoder", factor_rows(data["influence"], data["match"], n_factors))]
    if "influence_dense" in data.files:
        panels.append(("dense decoder", factor_rows(data["influence_dense"], data["match"], n_factors)))

    n_obs = panels[0][1].shape[1]
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(7.0, 1.35 * len(panels) + 0.6))
    for ax, (title, M) in zip(axes, panels):
        im = ax.imshow(normalise_rows(M), cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto",
                       interpolation="nearest")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_yticks(range(n_factors))
        ax.set_yticklabels(doublewell.FACTORS, fontsize=8)
        ax.tick_params(length=0)
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[-1].set_xticks(range(n_obs))
    axes[-1].set_xticklabels(range(n_obs), fontsize=7)
    axes[-1].set_xlabel("observed variable", fontsize=9)

    bar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("influence (row max = 1)", fontsize=8)
    bar.ax.tick_params(labelsize=7, length=0)
    bar.outline.set_visible(False)

    fig.savefig(args.output, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
