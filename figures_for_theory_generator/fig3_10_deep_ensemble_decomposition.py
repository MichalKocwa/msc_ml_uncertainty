"""Illustration for Section 3.5: predictive distribution of a deep ensemble on a
1D toy problem, with the predictive variance split into its aleatoric and
epistemic terms.

Toy problem (constructed for this figure, deliberately NOT the sine of E1):
    f(x)     = 0.3 * (x - 1) * (x - 3) * (x - 5)
    sigma(x) = 0.10 + 0.45 * (x / 6)^2            (heteroscedastic noise)
120 training inputs drawn uniformly from [0, 6], evaluation on [-2, 10], so the
figure shows extrapolation on both sides.

Method, following Section 3.5 (lakshminarayanan2017):
  - M = 5 members, each a 1-50-2 tanh MLP with a mean head and a variance
    head; the variance goes through softplus plus a small floor (1e-6);
  - each member is trained on the Gaussian negative log-likelihood (not MSE),
    on the WHOLE training set (no bagging);
  - diversity comes only from the initialisation seed and the minibatch order,
    both derived from one per-member seed;
  - Adam, lr 1e-2, batch_size 128, EPOCHS epochs, no weight decay and no
    adversarial training;
  - x and y standardised on training statistics; predictions are mapped back
    before plotting, including sigma_original = sigma_standardised * y_std.

Prediction (mixture moments, Section 3.5.2):
    mean      = (1/M) sum_m mu_m(x)
    aleatoric = (1/M) sum_m sigma_m^2(x)
    epistemic = spread of mu_m(x) around mean, divisor M (EPISTEMIC_DDOF = 0,
                the form written in the thesis) or M - 1 (EPISTEMIC_DDOF = 1,
                the estimator used in src/methods/ensemble.py).

Self-contained: torch + numpy + matplotlib only, nothing imported from src/.
The y-axis is framed on the ensemble's +-2 sigma band over FRAME_X_RANGE, so the
divergence of the members outside [0, 6] is visible; the true cubic leaves the
frame, since it grows far faster than a tanh network extrapolates.

Outputs (written next to this file; an existing name gets a numeric suffix
rather than being overwritten):
    fig3_10_deep_ensemble_decomposition.png   the figure (300 dpi, one panel)
    fig3_10_deep_ensemble_decomposition.csv   x, mean, std_aleatoric, std_epistemic

Usage:
    python fig3_10_deep_ensemble_decomposition.py
    python fig3_10_deep_ensemble_decomposition.py --seed 1 --epochs 3000
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Problem definition — swap these to change the target function / ranges.
# ----------------------------------------------------------------------------
X_TRAIN_RANGE = (0.0, 6.0)
X_EVAL_RANGE = (-2.0, 10.0)
N_TRAIN = 120
N_EVAL = 600


def f_true(x):
    """Cubic with roots at 1, 3, 5 — grows fast outside the training range."""
    return 0.3 * (x - 1.0) * (x - 3.0) * (x - 5.0)


def sigma_true(x):
    """Heteroscedastic noise level: small near 0, largest at the right end."""
    return 0.10 + 0.45 * (x / 6.0) ** 2


# ----------------------------------------------------------------------------
# Method hyperparameters (Section 3.5; values fixed by the figure spec).
# ----------------------------------------------------------------------------
M_MEMBERS = 5
HIDDEN = 50
LR = 1e-2
BATCH_SIZE = 128  # > N_TRAIN, so every "minibatch" is in fact the full set — see note in train_member
EPOCHS = 2000
VAR_FLOOR = 1e-6  # small floor added to softplus(variance head) for numerical stability

# 0 -> divisor M   (the (1/M) sum (sigma_m^2 + mu_m^2) - mu_*^2 form of the thesis / lakshminarayanan2017)
# 1 -> divisor M-1 (the ddof=1 estimator used in src/methods/ensemble.py)
EPISTEMIC_DDOF = 0

BASE_SEED = 0
DTYPE = torch.float64

# Points at which the epistemic std is compared against its in-range median (stdout report).
REPORT_POINTS = (8.0, -2.0)

# The y-axis is framed on the ensemble's +-2 sigma total band over this x-interval
# (plus a margin), so the members' divergence outside [0, 6] stays visible. The true
# cubic still leaves the frame, since it grows far faster than a tanh network can
# extrapolate. Set to X_TRAIN_RANGE to frame on the training targets only.
FRAME_X_RANGE = X_EVAL_RANGE

OUT_STEM = "fig3_10_deep_ensemble_decomposition"


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed NumPy and torch, and pin everything else that affects the result.

    The thread count matters for reproducibility: torch's intra-op parallelism
    changes the order of float64 reductions, and thousands of Adam steps amplify
    those rounding differences into measurably different networks. One thread
    keeps the run bit-reproducible from the seed alone.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(DTYPE)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def member_seed(base_seed: int, m: int) -> int:
    """Per-member seed derived deterministically from the single base seed."""
    return int(np.random.SeedSequence([base_seed, m]).generate_state(1)[0])


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class HeteroscedasticMLP(nn.Module):
    """1 -> HIDDEN (tanh) -> 2: mean head mu(x) and variance head sigma^2(x).

    The variance is softplus(raw) + VAR_FLOOR, as in lakshminarayanan2017.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = nn.Linear(1, hidden, dtype=DTYPE)
        self.out = nn.Linear(hidden, 2, dtype=DTYPE)

    def forward(self, x: torch.Tensor):
        h = torch.tanh(self.hidden(x))
        o = self.out(h)
        mu = o[:, :1]
        var = F.softplus(o[:, 1:]) + VAR_FLOOR
        return mu, var


def gaussian_nll(mu, var, y):
    """Mean Gaussian NLL, without the constant 0.5*log(2*pi)."""
    return torch.mean(0.5 * torch.log(var) + 0.5 * (y - mu) ** 2 / var)


def train_member(x_std, y_std, seed: int, epochs: int, log_every: int = 0):
    """Train one member from `seed` (initialisation AND batch order).

    Note on diversity: with N_TRAIN = 120 < BATCH_SIZE = 128, `bs` equals N and
    each epoch is a single step on the full set. The per-epoch permutation then
    only changes the summation order inside the mean (differences at the
    rounding floor), so in this configuration member diversity comes
    essentially from the initialisation alone. Batch order becomes an active
    source of diversity only when N_TRAIN > BATCH_SIZE.
    """
    torch.manual_seed(seed)  # weight initialisation draws from this stream
    model = HeteroscedasticMLP(HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)  # minibatch order

    n = x_std.shape[0]
    bs = min(BATCH_SIZE, n)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            opt.zero_grad()
            mu, var = model(x_std[idx])
            loss = gaussian_nll(mu, var, y_std[idx])
            loss.backward()
            opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs):
            with torch.no_grad():
                mu, var = model(x_std)
                history.append((epoch, float(gaussian_nll(mu, var, y_std))))
    model.eval()
    return model, history


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def unique_path(path: Path) -> Path:
    """Return `path` if free, otherwise the first `stem_<k>.suffix` that is."""
    if not path.exists():
        return path
    k = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{k}{path.suffix}")
        if not candidate.exists():
            return candidate
        k += 1


def crossings(x, a, b):
    """x-coordinates where curve `a` crosses curve `b` (linear interpolation)."""
    d = a - b
    out = []
    for i in range(len(d) - 1):
        if d[i] == 0.0:
            out.append(float(x[i]))
        elif d[i] * d[i + 1] < 0.0:
            t = d[i] / (d[i] - d[i + 1])
            out.append(float(x[i] + t * (x[i + 1] - x[i])))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    set_seed(args.seed)

    # --- data (drawn AFTER seeding, from the base seed) ---------------------
    rng = np.random.default_rng(args.seed)
    x_train = rng.uniform(*X_TRAIN_RANGE, size=N_TRAIN)
    y_train = f_true(x_train) + sigma_true(x_train) * rng.standard_normal(N_TRAIN)
    x_eval = np.linspace(*X_EVAL_RANGE, N_EVAL)

    # --- standardisation on training statistics ------------------------------
    x_mean, x_sd = x_train.mean(), x_train.std()
    y_mean, y_sd = y_train.mean(), y_train.std()
    xs_train = torch.as_tensor((x_train - x_mean) / x_sd, dtype=DTYPE).reshape(-1, 1)
    ys_train = torch.as_tensor((y_train - y_mean) / y_sd, dtype=DTYPE).reshape(-1, 1)
    xs_eval = torch.as_tensor((x_eval - x_mean) / x_sd, dtype=DTYPE).reshape(-1, 1)

    # --- train M members -----------------------------------------------------
    member_mu = np.empty((M_MEMBERS, N_EVAL))
    member_var = np.empty((M_MEMBERS, N_EVAL))
    print(f"seed={args.seed}  epochs={args.epochs}  M={M_MEMBERS}  N_train={N_TRAIN}  threads={torch.get_num_threads()}")
    for m in range(M_MEMBERS):
        seed_m = member_seed(args.seed, m)
        model, history = train_member(xs_train, ys_train, seed_m, args.epochs, log_every=max(1, args.epochs // 4))
        trace = "  ".join(f"ep{e}: {nll:.4f}" for e, nll in history)
        print(f"member {m}  seed={seed_m}  train NLL (standardised)  {trace}")
        with torch.no_grad():
            mu, var = model(xs_eval)
        # back to original units: mu -> mu*y_sd + y_mean, sigma -> sigma*y_sd, so var -> var*y_sd^2
        member_mu[m] = mu.numpy().ravel() * y_sd + y_mean
        member_var[m] = var.numpy().ravel() * y_sd ** 2

    # --- mixture moments -----------------------------------------------------
    mean = member_mu.mean(axis=0)
    var_aleatoric = member_var.mean(axis=0)
    var_epistemic = member_mu.var(axis=0, ddof=EPISTEMIC_DDOF)
    std_aleatoric = np.sqrt(var_aleatoric)
    std_epistemic = np.sqrt(var_epistemic)
    std_total = np.sqrt(var_aleatoric + var_epistemic)

    # --- stdout report -------------------------------------------------------
    in_range = (x_eval >= X_TRAIN_RANGE[0]) & (x_eval <= X_TRAIN_RANGE[1])
    epi_median_in = float(np.median(std_epistemic[in_range]))
    print(f"\nEPISTEMIC_DDOF = {EPISTEMIC_DDOF}  (divisor {'M' if EPISTEMIC_DDOF == 0 else 'M-1'})")
    print(f"median std_epistemic on [{X_TRAIN_RANGE[0]:g}, {X_TRAIN_RANGE[1]:g}] = {epi_median_in:.6f}")
    for x0 in REPORT_POINTS:
        epi_x0 = float(np.interp(x0, x_eval, std_epistemic))
        print(f"std_epistemic(x={x0:g}) = {epi_x0:.6f}   ratio to in-range median = {epi_x0 / epi_median_in:.3f}")
    xc = crossings(x_eval, std_aleatoric, std_epistemic)
    if xc:
        print("std_aleatoric and std_epistemic cross at x = " + ", ".join(f"{v:.4f}" for v in xc))
    else:
        print("std_aleatoric and std_epistemic do not cross on the evaluation grid")

    # --- CSV -------------------------------------------------------------------
    csv_path = unique_path(args.out_dir / f"{OUT_STEM}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "mean", "std_aleatoric", "std_epistemic"])
        for row in zip(x_eval, mean, std_aleatoric, std_epistemic):
            writer.writerow([f"{v:.10g}" for v in row])

    # --- figure (one panel) --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    # extrapolation regions greyed out
    ax.axvspan(X_EVAL_RANGE[0], X_TRAIN_RANGE[0], color="0.85", alpha=0.6, lw=0, zorder=0)
    ax.axvspan(X_TRAIN_RANGE[1], X_EVAL_RANGE[1], color="0.85", alpha=0.6, lw=0, zorder=0)

    # nested bands: wider = total, narrower = aleatoric only; the gap is the epistemic term
    ax.fill_between(x_eval, mean - 2 * std_total, mean + 2 * std_total,
                    color="tab:blue", alpha=0.18, lw=0, zorder=1,
                    label=r"$\pm 2\sigma$, total (aleatoric + epistemic)")
    ax.fill_between(x_eval, mean - 2 * std_aleatoric, mean + 2 * std_aleatoric,
                    color="tab:blue", alpha=0.40, lw=0, zorder=2,
                    label=r"$\pm 2\sigma$, aleatoric only")

    for m in range(M_MEMBERS):
        ax.plot(x_eval, member_mu[m], color="tab:blue", lw=0.7, alpha=0.7, zorder=3,
                label="member means" if m == 0 else None)
    ax.plot(x_eval, mean, color="tab:blue", lw=2.2, zorder=4, label="ensemble mean")
    ax.plot(x_eval, f_true(x_eval), color="black", ls="--", lw=1.2, zorder=5, label="true function")
    ax.scatter(x_train, y_train, s=12, color="black", alpha=0.7, zorder=6, label="training data")

    # frame on the ensemble's total band over FRAME_X_RANGE (and the training targets)
    in_frame = (x_eval >= FRAME_X_RANGE[0]) & (x_eval <= FRAME_X_RANGE[1])
    y_lo = min(y_train.min(), (mean - 2 * std_total)[in_frame].min())
    y_hi = max(y_train.max(), (mean + 2 * std_total)[in_frame].max())
    pad = 0.05 * (y_hi - y_lo)
    ax.set_xlim(*X_EVAL_RANGE)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    png_path = unique_path(args.out_dir / f"{OUT_STEM}.png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"\nwrote {png_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
