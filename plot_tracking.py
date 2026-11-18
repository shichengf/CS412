"""Plot the mixing weights saved by run_tracking.py.

One panel per regime on the test path: the true weight, the smoothed and raw
tracking estimates and, if present, the hard-assignment baseline.

    python plot_tracking.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRUE, SMOOTHED, RAW, HARD = "#222222", "#2a78d6", "#9cc3ee", "#9a9994"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/tracking/weights.npz")
    p.add_argument("--output", default="results/tracking/weights.pdf")
    args = p.parse_args()

    data = np.load(args.input)
    regimes = [int(k) for k in data["regimes"]]
    steps = np.arange(len(data["alpha_true"]))
    has_hard = "alpha_hard" in data.files

    fig, axes = plt.subplots(1, len(regimes), sharey=True, figsize=(2.4 * len(regimes), 2.2))
    axes = np.atleast_1d(axes)
    for i, (ax, k) in enumerate(zip(axes, regimes)):
        ax.plot(steps, data["alpha_true"][:, i], color=TRUE, lw=1.6, label="true")
        ax.plot(steps, data["alpha_raw"][:, i], color=RAW, lw=0.8, label="tracking, raw")
        ax.plot(steps, data["alpha_hat"][:, i], color=SMOOTHED, lw=1.6, label="tracking, smoothed")
        if has_hard:
            ax.plot(steps, data["alpha_hard"][:, i], color=HARD, lw=1.2, ls="--",
                    label="hard assignment")
        ax.set_title(f"regime {k}", fontsize=9)
        ax.set_xlabel("time step", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("mixing weight", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
