# Causal Representation Learning in Dynamic Environments

CS 412 (Introduction to Machine Learning), Fall 2026, University of Illinois Chicago.
Final project by Shicheng Fan, Jie Yang and Honghui Bao.

A system observed over time often switches between regimes: a vehicle goes from driving
straight to turning, a molecule from folded to unfolded. This project studies two things
that a learned latent representation can tell us about such a change:

1. Tracking. During a transition the mechanism is a mixture of the mechanisms seen in the
   pure regimes. We recover the mixing weights over time.
2. Interpreting. One latent usually carries most of the difference between regimes. We
   recover which observed variables that latent acts on, so that it can be named.

Both parts share the same first stage and differ only in the readout applied to its
frozen encoder.

The project started with the tracking part, which is what the intermediate report in
`intermediate_report/` covers. The interpretation part was added afterwards to widen the
scope for a team of three.

## Repository layout

```
dyncrl/
  model.py          Stage 1: temporal VAE with a regime-conditioned transition prior
  tracking.py       Readout 1: least-squares mixing weights of a transition
  interpret.py      Readout 2: sparse additive decoder and its influence matrix
  metrics.py        MCC, weight correlation, support precision
  data/
    transitions.py  benchmark for tracking (K mechanisms, one changed edge each)
    doublewell.py   benchmark for interpretation (double-well dynamics, known supports)
  baselines/        comparison methods for the two readouts
run_tracking.py     tracking experiment, end to end
run_interpret.py    interpretation experiment, end to end
report/             final report (LaTeX sources and figures)
proposal/            project proposal (PDF)
intermediate_report/ intermediate project report (PDF)
```

## Method in brief

Stage 1 learns latents `z_t` from short windows of observations. The encoder is a
frame-wise Gaussian, the decoder is a dense network, and the prior on `z_t` given its
past is a per-latent flow conditioned on the regime label. Training maximises the
sequence ELBO.

Tracking needs no further training. Let `mu_k` be the mean latent of regime `k` and
`B = [mu_1 - mu_0, ..., mu_{K-1} - mu_0]`. For a latent `z_t` observed during a transition,

```
alpha(t) = Proj_simplex( pinv(B) (z_t - mu_0) )
```

followed by a moving average over time.

Interpreting freezes the encoder and fits an additive decoder `x = sum_j f_j(z_j) + b`.
The influence of latent `j` on observation `i` is `A[i, j] = |f_j(+1)_i - f_j(-1)_i|`,
and an entropy penalty on the columns of `A` makes every latent act on few observations.
The latent with the largest shift between regimes (Cohen's d) is then described by its
most influenced observations.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9 or later. The repository contains no data files: both experiments simulate
their own data from a seed and run on a CPU. Outputs go to `results/`, which is not tracked.

## Running the experiments

```bash
python run_tracking.py
```

```bash
python run_interpret.py
```

Each script prints its metrics and saves the recovered quantities under `results/`.
Longer training improves the latents; `--help` lists the options.

With the default settings and seed 0, each script takes about ten minutes on a laptop CPU
and gives:

| Experiment | Latent recovery (MCC) | Readout metric |
|---|---|---|
| Tracking, path through three of five regimes | 0.970 | weight correlation 0.967 (0.874 before smoothing) |
| Interpreting, double-well with known supports | 0.943 | support precision 0.889 |

In the interpretation run the three latents with the largest regime shift are matched to
the factors `x`, `z1` and `z2`, the three factors that drive the regime, and their most
influenced observations are `[3, 2, 4]`, `[8, 6, 5]` and `[12, 9, 6]`, against true
supports `[0-4]`, `[5-8]` and `[4, 9-12]`.

## Tracking experiment in more detail

`run_tracking.py` trains Stage 1 on windows from the pure regimes, builds the differential
basis from the regime means, and recovers the mixing weights along a test path through
`--active` of the `--regimes` mechanisms. It saves `results/tracking/weights.npz` with the
arrays `alpha_true`, `alpha_raw`, `alpha_hat` (smoothed), `alpha_calibrated` and `regimes`.

## Interpretation experiment in more detail

`run_interpret.py` simulates the double-well system, trains Stage 1, fits the additive
decoder, and compares its influence matrix with the known supports. It saves
`results/interpret/influence.npz` with the arrays `influence` (observations by latents),
`truth` (observations by true factors), `regime_shift` and `match` (pairs of learned latent
and true factor).

## Report

`report/main.tex` is the final report in the CVPR template, with its figures in
`report/figs/` and the TikZ sources of the two schematics in `report/figsrc/`.

## References

- W. Yao, G. Chen, K. Zhang. Temporally Disentangled Representation Learning. NeurIPS 2022.
  The first stage follows this family of temporal VAEs.
- S. Fan, K. Zhang, L. Cheng. TRACE: Trajectory Recovery for Continuous Mechanism Evolution
  in Causal Representation Learning. arXiv:2601.21135, 2026.
- S. Fan, N. Elhendawy, J. Sun, K. Fang, K. Zhang, Y. Wang, L. Cheng. MOSAIC: Module
  Discovery via Sparse Additive Identifiable Causal Learning for Scientific Time Series.
  NeurIPS 2026, arXiv:2605.05524.
