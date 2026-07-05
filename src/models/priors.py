from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorConfig:
    # Scales referred to standardized PHE/SC.
    mu_scale: float = 1.5
    tau_scale: float = 1.0
    sigma_scale: float = 1.0
    # Uniform prior for the dramatic-class prevalence pi[v].
    beta_alpha: float = 1.0
    beta_beta: float = 1.0
