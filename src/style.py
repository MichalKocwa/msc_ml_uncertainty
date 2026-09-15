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
from pathlib import Path
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
    samples: np.ndarray = None,
    n_sample_lines: int = 0,
) -> None:
    """Draw one method's predictive distribution on `ax`.

    All arrays are in ORIGINAL y units (DEC-003) and come from a single
    `predict()` call per method per seed (DEC-007) — the same array the metrics
    are sliced from, which is what keeps the band in this figure and the
    `mpiw95` in the table describing the same numbers.

    `samples` (`(T, n)`, original y units) with `n_sample_lines > 0` draws up
    to that many individual per-run means as thin lines — the ensemble
    members, or a subset of the MC dropout / BBB passes. The thesis's Figure
    3.10 caption promises exactly this for the ensemble ("thin lines show the
    predictive means of the individual members"), and it is the one place
    where the disagreement the epistemic term is computed from is visible
    directly rather than through the band. The first `n_sample_lines` rows
    are drawn, not a random subset, so the figure is a deterministic function
    of the prediction. Methods without samples (`gp`, `laplace`, `map`) draw
    nothing extra.
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

    if samples is not None and n_sample_lines > 0:
        drawn = np.asarray(samples)[:n_sample_lines]
        for i, line in enumerate(drawn):
            ax.plot(x, line, color=colour, lw=0.6, alpha=0.7, zorder=3,
                    label="individual runs ({})".format(len(drawn)) if i == 0 else None)

    ax.plot(x, y_true, color=_TRUTH_COLOUR, ls="--", lw=1.1, zorder=3, label=r"$\sin(x)$")
    ax.plot(x, mean, color=colour, lw=1.5, zorder=4, label="predictive mean")
    ax.scatter(np.asarray(x_train).ravel(), y_train, s=4, color=_TRAIN_POINT_COLOUR,
               alpha=0.45, lw=0, zorder=5, label="training data")

    ax.set_title(METHOD_LABELS[method])
    if show_legend:
        ax.legend(loc="upper left", ncol=2)


def make_method_figures(
    panels: Dict[str, dict],
    x_test: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_true: np.ndarray,
    train_range,
    out_dir,
    prefix: str,
    suptitle: str = "",
    layout: str = "both",
    n_sample_lines: int = 0,
    ylim=None,
    methods: Sequence[str] = None,
) -> list:
    """Per-method figures and/or one combined grid, from prediction arrays.

    `panels[method]` holds `mean`, `var_aleatoric`, `var_epistemic` and
    optionally `samples`, all in original y units. `layout` is `"separate"`
    (one PNG per method), `"all"` (one 2x3 grid) or `"both"`. Shared axis
    limits across every panel, computed from every band with nothing clipped
    (`shared_limits`), unless `ylim` is given explicitly — an explicit limit is
    a recorded decision about the y-range, not a silent cap.

    Written as library code rather than inside an experiment script so that a
    figure can be redrawn from SAVED predictions (a different layout, with or
    without individual-run lines) without re-running any fit. Returns the
    paths written.
    """
    apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x_test).ravel()
    names = [n for n in (methods or METHOD_ORDER) if n in panels]

    bands = [np.asarray(y_train), np.asarray(y_true)]
    for n in names:
        p = panels[n]
        sd = np.sqrt(p["var_aleatoric"] + p["var_epistemic"])
        bands += [p["mean"] - FIGURE_SD_MULTIPLIER * sd, p["mean"] + FIGURE_SD_MULTIPLIER * sd]
    xlim, auto_ylim = shared_limits(x, bands)
    ylim = tuple(ylim) if ylim is not None else auto_ylim

    def draw(ax, name, show_legend):
        p = panels[name]
        plot_predictive_panel(
            ax, name, x, p["mean"], p["var_aleatoric"], p["var_epistemic"],
            x_train, y_train, y_true, train_range, show_legend=show_legend,
            samples=p.get("samples"), n_sample_lines=n_sample_lines,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    written = []
    if layout in ("separate", "both"):
        for name in names:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            draw(ax, name, show_legend=True)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            path = out_dir / "{}_{}.png".format(prefix, name)
            fig.savefig(path)
            plt.close(fig)
            written.append(path)

    if layout in ("all", "both"):
        n_cols = 3 if len(names) > 2 else len(names)
        n_rows = int(np.ceil(len(names) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)
        flat = axes.ravel()
        for ax, name in zip(flat, names):
            draw(ax, name, show_legend=(name == names[0]))
        for ax in flat[len(names):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("$x$")
        for ax in axes[:, 0]:
            ax.set_ylabel("$y$")
        if suptitle:
            fig.suptitle(suptitle)
        path = out_dir / "{}_all.png".format(prefix)
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


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
