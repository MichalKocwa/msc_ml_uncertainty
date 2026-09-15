"""Illustration for Section 3.3: predictive distribution of MC dropout on a 1D toy problem.

Toy problem (constructed for this figure): y = x^2/4 + eps, eps ~ N(0, 0.3^2),
20 training inputs drawn uniformly from [-4, 4], evaluation on [-6, 6].

Model: ReLU network 1-H-H-1, dropout on both hidden layers (no input dropout,
since with d = 1 it would zero the only feature), trained with the dropout
objective (squared error + weight decay). Prediction: T stochastic passes, each
with ONE mask shared by all inputs, i.e. one weight sample = one function.
The aleatoric term tau^{-1} is set to the true noise variance, so the bands show
what MC dropout adds: the epistemic term.

Several dropout probabilities give several panels on shared axes; the same data,
initialization seed and training budget are used in every panel.

Pure NumPy (manual backprop + Adam), deterministic for a fixed seed.

Usage:
  python fig_mc_dropout_toy.py --p-drop 0.3 --out img3_6.png           # figure in 3.3
  python fig_mc_dropout_toy.py --p-drop 0.1,0.3,0.5 --out panels.png   # rate comparison
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_TRAIN, X_TRAIN_RANGE, X_EVAL_RANGE = 20, (-4.0, 4.0), (-6.0, 6.0)
NOISE_STD = 0.3
Y_LIM = (-2.0, 11.0)


def f_true(x):
    return 0.25 * x ** 2


class MCDropoutMLP:
    """1-H-H-1 ReLU MLP with inverted dropout on both hidden layers."""

    def __init__(self, rng, hidden, p_drop):
        self.rng, self.hidden, self.p_drop = rng, hidden, p_drop
        he = lambda fan_in, shape: rng.normal(0.0, np.sqrt(2.0 / fan_in), shape)
        self.params = {
            "W1": he(1, (1, hidden)), "b1": np.zeros(hidden),
            "W2": he(hidden, (hidden, hidden)), "b2": np.zeros(hidden),
            "W3": he(hidden, (hidden, 1)), "b3": np.zeros(1),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def _mask(self, shape):
        keep = 1.0 - self.p_drop
        return (self.rng.random(shape) < keep) / keep

    def forward(self, x, shared_mask=False, cache=False):
        """shared_mask=False: new mask per input (training, as in the dropout objective).
        shared_mask=True: one mask for all inputs = one weight sample (prediction)."""
        P, H = self.params, self.hidden
        z1 = x @ P["W1"] + P["b1"]; h1 = np.maximum(z1, 0.0)
        m1 = self._mask((1, H) if shared_mask else h1.shape); d1 = h1 * m1
        z2 = d1 @ P["W2"] + P["b2"]; h2 = np.maximum(z2, 0.0)
        m2 = self._mask((1, H) if shared_mask else h2.shape); d2 = h2 * m2
        out = d2 @ P["W3"] + P["b3"]
        if cache:
            self.cache = (x, z1, m1, d1, z2, m2, d2)
        return out

    def step(self, x, y, lr, weight_decay):
        P, n = self.params, x.shape[0]
        out = self.forward(x, cache=True)
        x, z1, m1, d1, z2, m2, d2 = self.cache
        g_out = 2.0 * (out - y) / n
        grads = {"W3": d2.T @ g_out, "b3": g_out.sum(0)}
        g_h2 = (g_out @ P["W3"].T) * m2 * (z2 > 0)
        grads["W2"] = d1.T @ g_h2; grads["b2"] = g_h2.sum(0)
        g_h1 = (g_h2 @ P["W2"].T) * m1 * (z1 > 0)
        grads["W1"] = x.T @ g_h1; grads["b1"] = g_h1.sum(0)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in P:
            g = grads[k] + 2.0 * weight_decay * P[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            P[k] -= lr * mh / (np.sqrt(vh) + eps)


def make_data(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(*X_TRAIN_RANGE, size=(N_TRAIN, 1))
    y = f_true(x) + rng.normal(0.0, NOISE_STD, size=(N_TRAIN, 1))
    return x, y


def fit_predict(x_tr, y_tr, x_grid, p_drop, args):
    rng = np.random.default_rng(args.seed + 1)          # same init stream in every panel
    x_scale = X_TRAIN_RANGE[1]
    y_mean, y_std = y_tr.mean(), y_tr.std()
    net = MCDropoutMLP(rng, args.hidden, p_drop)
    for _ in range(args.steps):
        net.step(x_tr / x_scale, (y_tr - y_mean) / y_std, args.lr, args.weight_decay)
    passes = np.stack([net.forward(x_grid / x_scale, shared_mask=True)[:, 0]
                       for _ in range(args.T)]) * y_std + y_mean
    mean = passes.mean(0)
    var_epi = passes.var(0, ddof=1)                     # T-1 divisor, as in the text
    return passes, mean, np.sqrt(var_epi), np.sqrt(NOISE_STD ** 2 + var_epi)


def draw_panel(ax, xg, x_tr, y_tr, passes, mean, std_epi, std_tot, n_shown):
    for lo, hi in [(X_EVAL_RANGE[0], X_TRAIN_RANGE[0]), (X_TRAIN_RANGE[1], X_EVAL_RANGE[1])]:
        ax.axvspan(lo, hi, color="0.93", zorder=0)
    ax.fill_between(xg, mean - 2 * std_tot, mean + 2 * std_tot, color="#9ecae1", alpha=0.6,
                    linewidth=0, label=r"$\pm 2$ total std", zorder=1)
    ax.fill_between(xg, mean - 2 * std_epi, mean + 2 * std_epi, color="#3182bd", alpha=0.55,
                    linewidth=0, label=r"$\pm 2$ epistemic std", zorder=2)
    for t in range(n_shown):
        ax.plot(xg, passes[t], color="0.25", linewidth=0.6, alpha=0.7, zorder=3,
                label="single stochastic passes" if t == 0 else None)
    ax.plot(xg, f_true(xg), color="k", linestyle="--", linewidth=1.0,
            label=r"true function $x^2/4$", zorder=4)
    ax.plot(xg, mean, color="#08306b", linewidth=1.6, label="predictive mean", zorder=5)
    ax.scatter(x_tr[:, 0], y_tr[:, 0], s=12, color="#d7301f", zorder=6, label="training data")
    ax.set_xlim(*X_EVAL_RANGE); ax.set_ylim(*Y_LIM)
    ax.set_xlabel("$x$")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="img3_6.png")
    parser.add_argument("--p-drop", default="0.1,0.3,0.5",
                        help="comma-separated probabilities of DROPPING a unit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--n-shown", type=int, default=12)
    args = parser.parse_args()

    p_values = [float(v) for v in args.p_drop.split(",")]
    x_tr, y_tr = make_data(args.seed)
    x_grid = np.linspace(*X_EVAL_RANGE, 400)[:, None]
    xg = x_grid[:, 0]
    inside = (xg >= X_TRAIN_RANGE[0]) & (xg <= X_TRAIN_RANGE[1])

    n = len(p_values)
    fig, axes = plt.subplots(1, n, figsize=(6.0 if n == 1 else 3.2 * n, 3.8 if n == 1 else 3.3),
                             sharey=True, squeeze=False)
    for ax, p in zip(axes[0], p_values):
        passes, mean, std_epi, std_tot = fit_predict(x_tr, y_tr, x_grid, p, args)
        print(f"p_drop={p}: mean epistemic std inside {std_epi[inside].mean():.2f}, "
              f"outside {std_epi[~inside].mean():.2f} "
              f"(ratio {std_epi[~inside].mean() / std_epi[inside].mean():.1f})")
        draw_panel(ax, xg, x_tr, y_tr, passes, mean, std_epi, std_tot, args.n_shown)
        if n > 1:
            ax.set_title(f"dropout probability {p:g}", fontsize=10)
    axes[0][0].set_ylabel("$y$")
    if n == 1:
        axes[0][0].legend(frameon=False, fontsize=8, loc="upper center")
    else:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=6)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    if n == 1:
        fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
