"""Figure 2.2 — epistemic uncertainty illustrated as function spread.

Illustration for Section "Epistemic uncertainty" of the thesis (label
`fig:epistemic-uncertainty`). A Gaussian process is conditioned on noisy
observations of a sine restricted to the training range [0, 6] and several
functions are drawn from its posterior over the wider range [-2, 10], together
with the posterior mean and its +/-2 sd band. Inside the training range the
draws agree with one another; outside it they fan out,
because many functions consistent with the observations remain — this spread
is the epistemic uncertainty of thesis eq. `predictive`.

A GP is used because its posterior is exact (thesis Sec. 3.1), so the drawn
curves are genuine posterior samples rather than an approximation of them.
The observation noise is small and constant, so the only thing that changes
along `x` is the presence or absence of training data. This is a THEORY
illustration, not an experiment: nothing here feeds a results table.

Run from the repository root:

    .venv/Scripts/python.exe figures_for_theory_generator/fig2_2_epistemic_function_spread.py

Output: `figures_for_theory_generator/output/fig2_2_epistemic_function_spread.png`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.seeding import set_seed  # noqa: E402
from src.style import FIGURE_SD_MULTIPLIER, apply_style, plt  # noqa: E402

# ----------------------------------------------------------------------------
# Data-generating process. All values below are arbitrary choices made for
# visual clarity of the illustration (assistant's choice, no experiment or
# literature behind them); nothing in the thesis depends on their exact value.
# ----------------------------------------------------------------------------
TRAIN_MIN, TRAIN_MAX = 0.0, 6.0     # where observations exist
PLOT_MIN, PLOT_MAX = -2.0, 10.0     # where the posterior is evaluated
AMPLITUDE = 1.0
NOISE_SD = 0.05                     # small, constant: keeps the aleatoric term visually negligible
N_TRAIN = 40
N_GRID = 600
N_SAMPLES = 20                      # posterior function draws shown


def true_mean(x: np.ndarray) -> np.ndarray:
    return AMPLITUDE * np.sin(x)


def make_data(rng: np.random.Generator, n: int):
    x = rng.uniform(TRAIN_MIN, TRAIN_MAX, size=n)
    y = true_mean(x) + rng.normal(0.0, NOISE_SD, size=n)
    return x, y


# ----------------------------------------------------------------------------
# GP posterior (thesis Sec. 3.1). Observation noise enters through `alpha`
# (a fixed diagonal on the training kernel) rather than a `WhiteKernel`, so
# that the predictive covariance below is that of the latent function f and
# the drawn curves are noise-free function samples, not noisy observations.
# ----------------------------------------------------------------------------
def fit_gp(x: np.ndarray, y: np.ndarray, seed: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=NOISE_SD**2, normalize_y=False, n_restarts_optimizer=5, random_state=seed,
    )
    gp.fit(x[:, None], y)
    return gp


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def draw(x_train, y_train, x_grid, mean, sd, samples, out_path: Path) -> None:
    apply_style()
    k = FIGURE_SD_MULTIPLIER
    colour = "#1f77b4"

    fig, ax = plt.subplots(figsize=(8.0, 3.6))

    # Shade where no training data exist, so the eye can pair "fan-out" with
    # "no observations" without any annotation.
    for lo, hi in ((PLOT_MIN, TRAIN_MIN), (TRAIN_MAX, PLOT_MAX)):
        ax.axvspan(lo, hi, color="#bbbbbb", alpha=0.18, linewidth=0, zorder=0)

    # Posterior sd of the latent function only (no observation noise, see
    # `fit_gp`): the band is the epistemic term alone, the counterpart of the
    # purely aleatoric band in Figure 2.1.
    ax.fill_between(
        x_grid, mean - k * sd, mean + k * sd, color=colour, alpha=0.18, linewidth=0, zorder=1,
        label=f"posterior band (mean ± {k:g} posterior std)",
    )
    for i, s in enumerate(samples):
        ax.plot(
            x_grid, s, color=colour, alpha=0.35, linewidth=0.8, zorder=2,
            label="posterior function samples" if i == 0 else None,
        )
    ax.plot(
        x_grid, true_mean(x_grid), color="#000000", linestyle="--", linewidth=1.0, zorder=3,
        label="true noise-free function",
    )
    ax.plot(x_grid, mean, color=colour, linewidth=1.8, zorder=4, label="posterior mean")
    ax.scatter(
        x_train, y_train, s=14, color="#333333", alpha=0.85, linewidths=0, zorder=5,
        label="training points",
    )

    ax.set_xlim(PLOT_MIN, PLOT_MAX)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper left", ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "output" / "fig2_2_epistemic_function_spread.png",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x_train, y_train = make_data(rng, N_TRAIN)

    gp = fit_gp(x_train, y_train, args.seed)
    print(f"fitted kernel: {gp.kernel_}")

    x_grid = np.linspace(PLOT_MIN, PLOT_MAX, N_GRID)
    mean, sd = gp.predict(x_grid[:, None], return_std=True)
    samples = gp.sample_y(x_grid[:, None], n_samples=args.n_samples, random_state=args.seed).T

    # Sanity check printed to the console: posterior sd inside vs outside the data.
    inside = (x_grid >= TRAIN_MIN) & (x_grid <= TRAIN_MAX)
    print(f"mean posterior sd inside [{TRAIN_MIN:g}, {TRAIN_MAX:g}]: {sd[inside].mean():.3f}")
    print(f"mean posterior sd outside:                       {sd[~inside].mean():.3f}")

    draw(x_train, y_train, x_grid, mean, sd, samples, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
