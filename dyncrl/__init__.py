"""Causal representation learning in dynamic environments.

model.py      Stage 1, shared by both readouts
tracking.py   Readout 1, mixing weights of a mechanism transition
interpret.py  Readout 2, observed variables behind the regime-associated latent
metrics.py    MCC, weight correlation, support precision
data/         the two synthetic benchmarks
"""

from .model import TemporalVAE, fit_stage1

__all__ = ["TemporalVAE", "fit_stage1"]
