"""Figure 2.1 — heteroscedastic aleatoric uncertainty on synthetic data.

Illustration for Section "Aleatoric uncertainty" of the thesis (label
`fig:heteroscedastic-aleatoric`). A two-head MLP predicts the mean
`mu_theta(x)` and the log-variance `log sigma^2_theta(x)` and is trained by
minimising the Gaussian negative log-likelihood (thesis eq. `gaussian-nll`):

    L(theta) = 1/N sum_i [ (y_i - mu(x_i))^2 / (2 sigma^2(x_i)) + 1/2 log sigma^2(x_i) ]

The data are a sine whose observation noise is piecewise constant in `x`:
small on `[0, 5)`, smaller still on `[5, 7)`, large on `[7, 10]`. The training
points cover the whole plotted range densely and uniformly, so the only thing
that changes along `x` is the noise level — the widening of the
`+/-2 sigma_theta(x)` band can therefore be read as noise inherent to the
data (visible directly as the scatter of the points), not as a lack of
observations. This is a THEORY illustration, not an
experiment: nothing here feeds a results table.

Run from the repository root:

    .venv/Scripts/python.exe figures_for_theory_generator/fig2_1_heteroscedastic_aleatoric.py

Output: `figures_for_theory_generator/output/fig2_1_heteroscedastic_aleatoric.png`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.seeding import set_seed  # noqa: E402
from src.style import FIGURE_SD_MULTIPLIER, apply_style, plt  # noqa: E402

# ----------------------------------------------------------------------------
# Data-generating process. All values below are arbitrary choices made for
# visual clarity of the illustration (assistant's choice, no experiment or
# literature behind them); nothing in the thesis depends on their exact value.
# ----------------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 10.0
AMPLITUDE = 1.0
# Noise regimes: (upper edge of the segment, noise sd on that segment).
# Abrupt changes at x = 5 and x = 7 so the band is seen to follow the local
# scatter of the points rather than a smooth trend.
NOISE_SEGMENTS = [(5.0, 0.15), (7.0, 0.04), (X_MAX, 0.50)]
N_TRAIN = 500                       # dense enough that no gap in coverage is visible
N_GRID = 1000


def true_mean(x: np.ndarray) -> np.ndarray:
    return AMPLITUDE * np.sin(x)


def true_sigma(x: np.ndarray) -> np.ndarray:
    """Piecewise-constant noise standard deviation (see NOISE_SEGMENTS)."""
    x = np.asarray(x, dtype=float)
    sigma = np.empty_like(x)
    lower = X_MIN
    for upper, sd in NOISE_SEGMENTS:
        sigma[(x >= lower) & (x <= upper)] = sd
        lower = upper
    return sigma


def make_data(rng: np.random.Generator, n: int):
    x = rng.uniform(X_MIN, X_MAX, size=n)
    y = true_mean(x) + rng.normal(0.0, 1.0, size=n) * true_sigma(x)
    return x, y


# ----------------------------------------------------------------------------
# Two-head network + Gaussian NLL (thesis eq. gaussian-nll)
# ----------------------------------------------------------------------------
class TwoHeadMLP(nn.Module):
    """Shared trunk, two linear heads: predictive mean and log-variance.

    Predicting the log-variance (rather than the variance) keeps
    `sigma^2 = exp(.) > 0` without a constraint, the standard parametrisation
    of Kendall & Gal (2017), Sec. 3.
    """

    def __init__(self, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers, d_in = [], 1
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.Tanh()]
            d_in = hidden
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.mean_head(h).squeeze(-1), self.logvar_head(h).squeeze(-1)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (0.5 * (y - mean) ** 2 * torch.exp(-logvar) + 0.5 * logvar).mean()


def train(
    x: np.ndarray, y: np.ndarray, epochs: int, lr: float = 1e-2, weight_decay: float = 1e-3, verbose: bool = True,
) -> TwoHeadMLP:
    # Standardise inputs for the tanh trunk; y is left in original units so the
    # predicted sigma is directly comparable with the true noise sd.
    x_t = torch.as_tensor((x - x.mean()) / x.std())[:, None]
    y_t = torch.as_tensor(y)
    model = TwoHeadMLP()
    # Two hidden layers so the log-variance head can follow the abrupt noise
    # changes at x = 5 and x = 7; a light weight decay and a short schedule
    # (1500 epochs) keep the band edge from tracking local fluctuations of the
    # sample scatter on the noisy segment. Values chosen by eye from eight
    # trial settings — arbitrary, nothing in the thesis depends on them.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        opt.zero_grad()
        mean, logvar = model(x_t)
        loss = gaussian_nll(mean, logvar, y_t)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:5d}  NLL = {loss.item():.4f}")
    model.x_mean, model.x_std = float(x.mean()), float(x.std())
    return model


@torch.no_grad()
def predict(model: TwoHeadMLP, x: np.ndarray):
    x_t = torch.as_tensor((x - model.x_mean) / model.x_std)[:, None]
    mean, logvar = model(x_t)
    return mean.numpy(), np.exp(0.5 * logvar.numpy())


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def draw(x_train, y_train, x_grid, mu_hat, sigma_hat, out_path: Path) -> None:
    apply_style()
    k = FIGURE_SD_MULTIPLIER
    band_colour = "#1f77b4"

    fig, ax = plt.subplots(figsize=(8.0, 3.6))

    ax.fill_between(
        x_grid, mu_hat - k * sigma_hat, mu_hat + k * sigma_hat,
        color=band_colour, alpha=0.22, linewidth=0,
        label=f"predicted aleatoric band (mean ± {k:g} predicted std)",
    )
    ax.plot(
        x_grid, true_mean(x_grid), color="#000000", linestyle="--", linewidth=1.0,
        label="true noise-free function",
    )
    ax.plot(
        x_grid, mu_hat, color=band_colour, linewidth=1.8,
        label="predictive mean",
    )
    ax.scatter(
        x_train, y_train, s=9, color="#333333", alpha=0.55, linewidths=0, zorder=3,
        label="training points",
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper left", ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--quick", action="store_true", help="few epochs; only checks the script runs")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "output" / "fig2_1_heteroscedastic_aleatoric.png",
    )
    args = parser.parse_args()
    epochs = 100 if args.quick else args.epochs

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x_train, y_train = make_data(rng, N_TRAIN)

    model = train(x_train, y_train, epochs=epochs)

    x_grid = np.linspace(X_MIN, X_MAX, N_GRID)
    mu_hat, sigma_hat = predict(model, x_grid)

    # Sanity check printed to the console: predicted vs true noise sd at both ends.
    lower = X_MIN
    for upper, sd in NOISE_SEGMENTS:
        in_seg = (x_grid >= lower) & (x_grid <= upper)
        print(f"x in [{lower:g}, {upper:g}]   true sigma = {sd:.3f}   mean predicted sigma = {sigma_hat[in_seg].mean():.3f}")
        lower = upper

    draw(x_train, y_train, x_grid, mu_hat, sigma_hat, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
