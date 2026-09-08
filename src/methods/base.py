"""Common interface every uncertainty-estimation method implements.

Without this, the comparison fragments at the code level: every method would
report its predictive distribution differently and every table would need
hand-written glue. `var_aleatoric` and `var_epistemic` must be reported
separately by every method — a method that cannot separate them (e.g. `map`)
reports `var_epistemic` as zeros explicitly rather than folding everything
into `var_aleatoric`.
"""
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Prediction:
    """Predictive distribution, decomposed per the law of total variance."""
    mean: np.ndarray                       # (n,)
    var_aleatoric: np.ndarray              # (n,)  E_q[sigma^2_theta(x)]
    var_epistemic: np.ndarray              # (n,)  Var_q[mu_theta(x)]
    samples: Optional[np.ndarray] = None   # (T, n) per-run means; None for non-sampling methods (GP, LA-linearised)

    @property
    def var_total(self) -> np.ndarray:
        return self.var_aleatoric + self.var_epistemic

    @property
    def std_total(self) -> np.ndarray:
        return np.sqrt(self.var_total)


class UncertaintyMethod(Protocol):
    name: str          # "gp", "bbb", "mcd", "laplace", "ensemble", "map"
    n_parameters: int  # for the cost comparison

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> "UncertaintyMethod": ...
    def predict(self, X: np.ndarray) -> Prediction: ...
