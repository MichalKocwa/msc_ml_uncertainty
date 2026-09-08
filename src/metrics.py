"""The four reported metrics, plus one diagnostic column.

Every function here takes predictions **already un-standardised into original
y units** (DEC-003). Nothing in this module knows about the standardiser: that
conversion happens once, in the experiment script, so there is no path by
which a metric could be computed on standardised inputs by accident.

**Predictive density (DEC-009).** Every method is scored as a single Gaussian
`N(mean, var_aleatoric + var_epistemic)`, including the three sampling methods
(`mcd`, `bbb`, `ensemble`), whose `T` (or `M`) sampled means are moment-matched
rather than treated as a mixture. `gp` and `laplace` are analytically Gaussian,
so for them this is not an approximation at all. The cost of the choice is
recorded in DEC-009: it is not the log-sum-exp estimator Gal & Ghahramani
(2016) publish for MC dropout, so our `mcd` log-likelihood is not directly
comparable with that paper's tables. It is, however, one convention across the
whole table, which is what makes the six columns comparable with each other —
and that is what E1 is for.

**Interval level (DEC-008).** `Z95` is the exact two-sided 95% Gaussian
quantile, so `PICP@95` is measured against a genuinely nominal-95% interval and
the column header is literally true. The figures draw `+/-2 sd` instead
(DEC-001), which is 2% wider; the two are deliberately different conventions,
not an inconsistency, and both the figure captions and the thesis must say so.
"""
from typing import Dict

import numpy as np
from scipy.stats import norm

# Two-sided 95% Gaussian quantile, 1.959964 — computed rather than written out,
# so the constant cannot drift from the level it claims to implement.
Z95 = float(norm.ppf(0.975))

# Nominal levels the calibration curve is evaluated at (DEC-013). Deciles plus
# the two tail levels the literature reports at: 0.95 is the level PICP@95 uses,
# so the curve passes exactly through that column and the two cannot disagree.
# The grid is arbitrary and labelled as such in DEC-013 — a denser one changes
# the summary scalars in the third decimal, not their ordering.
CALIBRATION_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

METRIC_COLUMNS = ("rmse", "ll", "picp95", "mpiw95", "sigma_fitted", "cal_err", "cal_bias")


def rmse(y: np.ndarray, mean: np.ndarray) -> float:
    """Depends only on the predictive mean — reported so that a method with a
    good band and a bad fit cannot hide behind its uncertainty.
    """
    return float(np.sqrt(np.mean((y - mean) ** 2)))


def log_likelihood(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    """Mean Gaussian log-density of the observations under `N(mean, var)`.

    Literature convention: higher is better, values usually negative. The
    `0.5*log(2*pi)` term is INCLUDED — `backbone.gaussian_nll` drops it because
    a constant does not change an optimiser's argmin, but a reported
    log-likelihood that silently omits it is off by 0.919 and not comparable to
    any published number.
    """
    return float(np.mean(-0.5 * np.log(2.0 * np.pi * var) - (y - mean) ** 2 / (2.0 * var)))


def picp(y: np.ndarray, mean: np.ndarray, var: np.ndarray, z: float = Z95) -> float:
    """Prediction Interval Coverage Probability: the fraction of observations
    falling inside `mean +/- z*sd`. Nominal level 0.95 at `Z95`.
    """
    half_width = z * np.sqrt(var)
    inside = (y >= mean - half_width) & (y <= mean + half_width)
    return float(np.mean(inside))


def mpiw(var: np.ndarray, z: float = Z95) -> float:
    """Mean Prediction Interval Width, `2*z*sd` averaged over the region.

    **Never report this without `picp`.** A width on its own is not a quality:
    it can be small because the method is well calibrated or because it is
    overconfident, and nothing in this number distinguishes the two.
    """
    return float(np.mean(2.0 * z * np.sqrt(var)))


def fitted_sigma(var_aleatoric: np.ndarray) -> float:
    """Diagnostic (DEC-010), not one of the four metrics: the observation-noise
    standard deviation each method arrived at, in original y units,
    `sqrt(mean(var_aleatoric))`.

    Reported because the toy problem's true value is known by construction
    (`data.NOISE_SIGMA` = 0.1), so this column reads against ground truth and
    gives N-001 — MC dropout absorbing its injected noise into the single
    global `log_sigma2` — a basis in `results/e1_toy_metrics.csv` rather than
    in a smoke run. It is a root-mean-square over the region, so it is constant
    across regions for every method whose aleatoric term is a single global
    scalar (all six as configured here); it would vary by region only under a
    heteroscedastic model.
    """
    return float(np.sqrt(np.mean(var_aleatoric)))


def calibration_curve(
    y: np.ndarray, mean: np.ndarray, var: np.ndarray, levels=CALIBRATION_LEVELS,
) -> np.ndarray:
    """Empirical coverage of the central interval at each nominal level.

    `PICP@95` is one point of this curve. The rest of it separates two failures
    that a single level cannot tell apart: an interval that is uniformly too
    narrow sits below the diagonal everywhere, whereas an interval of the right
    width around a displaced mean is flat and low — the difference between a
    method that is overconfident and one that is simply wrong.

    **What this is and is not.** Under DEC-009 every method is scored as a
    single Gaussian, so this curve is a monotone re-expression of the
    distribution of the standardised residuals `(y - mean) / sd`. It is
    therefore a better SUMMARY of a measurement already in hand, not a new
    measurement — worth stating plainly, because presenting it as independent
    evidence would overstate it. Its value is that it reduces calibration to
    one number in the form the literature reports (the regression analogue of
    the ECE daxberger2021 report alongside NLL).
    """
    sd = np.sqrt(var)
    residual = np.abs(y - mean)
    return np.array([
        float(np.mean(residual <= norm.ppf(0.5 + 0.5 * level) * sd)) for level in levels
    ])


def calibration_error(
    y: np.ndarray, mean: np.ndarray, var: np.ndarray, levels=CALIBRATION_LEVELS,
):
    """`(cal_err, cal_bias)` — mean absolute and mean signed deviation of the
    calibration curve from the diagonal.

    `cal_err` is the headline scalar: 0 is perfect, and it is what ranks methods
    by calibration. It is an absolute deviation, so it discards direction, which
    is exactly what `cal_bias` keeps: **positive means the intervals over-cover
    (the method is under-confident), negative means they under-cover (it is
    overconfident)**. Reporting only the absolute value would make an
    overconfident method and an over-cautious one look identical, and for this
    comparison that distinction is most of the point.
    """
    coverage = calibration_curve(y, mean, var, levels)
    deviation = coverage - np.asarray(levels, dtype=float)
    return float(np.mean(np.abs(deviation))), float(np.mean(deviation))


def compute_metrics(
    y: np.ndarray,
    mean: np.ndarray,
    var_aleatoric: np.ndarray,
    var_epistemic: np.ndarray,
    z: float = Z95,
) -> Dict[str, float]:
    """All four metrics plus the diagnostic, for one method on one region.

    Takes the two variance components separately rather than a total, so that
    `sigma_fitted` reads the aleatoric term directly and the total is formed in
    exactly one place.
    """
    var_total = var_aleatoric + var_epistemic
    if np.any(var_total <= 0):
        raise ValueError("non-positive predictive variance; log-likelihood is undefined")
    cal_err, cal_bias = calibration_error(y, mean, var_total)
    return {
        "rmse": rmse(y, mean),
        "ll": log_likelihood(y, mean, var_total),
        "picp95": picp(y, mean, var_total, z=z),
        "mpiw95": mpiw(var_total, z=z),
        "sigma_fitted": fitted_sigma(var_aleatoric),
        "cal_err": cal_err,
        "cal_bias": cal_bias,
    }
