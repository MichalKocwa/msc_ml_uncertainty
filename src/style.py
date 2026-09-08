"""Shared plotting style and the E1 predictive panel.

One panel-drawing function, used by both the per-method figures and the
combined 2x3 grid, so the two cannot drift apart in what they draw or in how
they label it.

**Band multiplier (DEC-001 / DEC-008).** Figures draw `+/-2 sd`;
`metrics.Z95` = 1.96 scores the tables. The 2% difference is deliberate — the
`+/-2 sd` band is the convention DEC-001 fixed and the one Foong et al. (2019)
plot, while `PICP@95` has to be measured against an exactly-95% interval to
deserve its name. A caption that shows a band must therefore say `+/-2 sd`,
never `95%`.

**Nested bands (DEC-001).** Inner band is 2 sd of the ALEATORIC term, outer is
2 sd of the TOTAL. This is the decomposition Figure 2.3 of the thesis already
promises the reader, and it makes MAP's identically-zero epistemic term visible
as a collapsed outer band rather than as an unexplained absence.

Note the departure from Foong et al. (2019), whose 1D figures plot the
predictive over `f_theta(x)` WITHOUT output noise: their band is comparable to
the gap between our two bands, not to either one of them. A side-by-side visual
comparison with that paper's figures is therefore not like-for-like, and the
thesis has to say so.
"""
from typing import Dict, Sequence

import matplotlib
import numpy as np

from src.experiment_io import METHOD_ORDER  # noqa: F401  (re-exported; see below)

matplotlib.use("Agg")  # figures are written to files, never shown; keeps the script headless
import matplotlib.pyplot as plt  # noqa: E402

# Standard deviations drawn on every band. See the module docstring for why this
# is not `metrics.Z95`.
FIGURE_SD_MULTIPLIER = 2.0

# `METHOD_ORDER` now lives in `src/experiment_io.py` and is re-exported here:
# E2 produces no figures and must not import matplotlib just to order a table,
# while everything that already read the order from this module keeps working.

METHOD_LABELS: Dict[str, str] = {
    "map": "MAP (deterministic baseline)",
    "mcd": "MC dropout",
    "ensemble": "Deep ensemble",
    "bbb": "Bayes by Backprop",
    "laplace": "Laplace (linearised)",
    "gp": "Gaussian process",
}

# One colour per method, held constant across every figure in the chapter so a
# reader can carry a colour between panels. Chosen for distinguishability in
# greyscale print as well as on screen.
METHOD_COLOURS: Dict[str, str] = {
    "map": "#7f7f7f",
    "mcd": "#d62728",
    "ensemble": "#ff7f0e",
    "bbb": "#9467bd",
    "laplace": "#2ca02c",
    "gp": "#1f77b4",
}

_TRAIN_POINT_COLOUR = "#333333"
_TRUTH_COLOUR = "#000000"
_EXTRAPOLATION_SHADE = "#bbbbbb"


def apply_style() -> None:
    """Matplotlib settings shared by every E1 figure."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
    })


def plot_predictive_panel(
    ax,
    method: str,
    x_test: np.ndarray,
    mean: np.ndarray,
    var_aleatoric: np.ndarray,
    var_epistemic: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_true: np.ndarray,
    train_range,
    show_legend: bool = False,
) -> None:
    """Draw one method's predictive distribution on `ax`.

    All arrays are in ORIGINAL y units (DEC-003) and come from a single
    `predict()` call per method per seed (DEC-007) — the same array the metrics
    are sliced from, which is what keeps the band in this figure and the
    `mpiw95` in the table describing the same numbers.
    """
    x = np.asarray(x_test).ravel()
    sd_total = np.sqrt(var_aleatoric + var_epistemic)
    sd_aleatoric = np.sqrt(var_aleatoric)
    k = FIGURE_SD_MULTIPLIER
    colour = METHOD_COLOURS[method]

    lo, hi = train_range
    ax.axvspan(x.min(), lo, color=_EXTRAPOLATION_SHADE, alpha=0.35, lw=0, zorder=0)
    ax.axvspan(hi, x.max(), color=_EXTRAPOLATION_SHADE, alpha=0.35, lw=0,
               zorder=0, label="extrapolation")

    ax.fill_between(x, mean - k * sd_total, mean + k * sd_total,
                    color=colour, alpha=0.22, lw=0, zorder=1,
                    label=rf"$\pm{k:g}\sigma$ total")
    ax.fill_between(x, mean - k * sd_aleatoric, mean + k * sd_aleatoric,
                    color=colour, alpha=0.42, lw=0, zorder=2,
                    label=rf"$\pm{k:g}\sigma$ aleatoric")

    ax.plot(x, y_true, color=_TRUTH_COLOUR, ls="--", lw=1.1, zorder=3, label=r"$\sin(x)$")
    ax.plot(x, mean, color=colour, lw=1.5, zorder=4, label="predictive mean")
    ax.scatter(np.asarray(x_train).ravel(), y_train, s=4, color=_TRAIN_POINT_COLOUR,
               alpha=0.45, lw=0, zorder=5, label="training data")

    ax.set_title(METHOD_LABELS[method])
    if show_legend:
        ax.legend(loc="upper left", ncol=2)


def shared_limits(x_test: np.ndarray, bands: Sequence[np.ndarray], pad: float = 0.05):
    """X and Y limits covering every panel, so the six are visually comparable.

    `bands` is every array that must stay inside the frame — each method's
    `mean +/- k*sd_total`, plus the training targets. NOTHING IS CLIPPED: a
    method whose extrapolation band is enormous squashes the others rather than
    running off the top, because a clipped band reads as a narrow one and would
    invert the very comparison this figure exists to make. If the result is
    unreadable, the fix is a decision about the y-range, taken and recorded,
    not a silent cap here.
    """
    x = np.asarray(x_test).ravel()
    finite = np.concatenate([np.asarray(b).ravel() for b in bands])
    finite = finite[np.isfinite(finite)]
    y_lo, y_hi = float(finite.min()), float(finite.max())
    span = y_hi - y_lo
    return (float(x.min()), float(x.max())), (y_lo - pad * span, y_hi + pad * span)
