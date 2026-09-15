"""Rysunek 3.8: rozkład predykcyjny aproksymacji Laplace'a dopasowanej post hoc.

Kroki (zgodnie z Podrozdziałem 3.4):
  1. Deterministyczny trening MLP 1-50-1 (tanh lub ReLU) na [0, 6], estymator MAP.
  2. Krzywizna: GGN dla gaussowskiej wiarygodności ze stałą wariancją,
     Lambda = J^T J / sigma^2 + I / sigma_p^2 (pełna macierz, P = 151 parametrów).
  3. sigma^2 i sigma_p^2 dobrane post hoc przez maksymalizację Laplace'owskiej
     aproksymacji evidence (eq:laplace-evidence).
  4. Predykcja zlinearyzowana (eq:laplace-predictive):
     mean = f_MAP(x*), var = J(x*)^T Sigma J(x*) + sigma^2.

Wymaga tylko numpy, scipy i matplotlib. Uruchomienie:
  python fig_laplace_posterior.py --func logistic --activation tanh --out img3_8.png

Uwaga o aktywacji: z ReLU pełna (all-weights) linearyzacja daje wąskie piki pasma
w miejscach, gdzie kilka neuronów ma załamanie między dwoma punktami treningowymi.
To własność metody (Jakobian jest nieciągły), nie błąd. Dla ilustracji domyślnie tanh.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

FUNCTIONS = {
    "bumps": lambda x: 0.3 * x - 1.2 * np.exp(-(x - 1.5) ** 2)
                       + 1.5 * np.exp(-(x - 4.5) ** 2 / 0.8),
    "logistic": lambda x: 2.0 / (1.0 + np.exp(-3.0 * (x - 3.0))) - 1.0 + 0.1 * x,
    "cubic": lambda x: 0.05 * x ** 3 - 0.4 * x ** 2 + 0.6 * x,
}


# ---------------------------------------------------------------- MLP 1-H-1
def init_params(rng, hidden):
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0), size=hidden),        # wejście 1D
        "b1": np.zeros(hidden),
        "w2": rng.normal(0.0, np.sqrt(2.0 / hidden), size=hidden),
        "b2": np.zeros(1),
    }


def flatten(p):
    return np.concatenate([p["W1"], p["b1"], p["w2"], p["b2"]])


ACT = "relu"


def act(h):
    return np.maximum(h, 0.0) if ACT == "relu" else np.tanh(h)


def dact(h):
    return (h > 0).astype(float) if ACT == "relu" else 1.0 - np.tanh(h) ** 2


def forward(p, x):
    h = np.outer(x, p["W1"]) + p["b1"]          # (n, H)
    a = act(h)
    return a @ p["w2"] + p["b2"][0], h, a


def jacobian(p, x):
    """J[n, :] = d f(x_n) / d theta w kolejności [W1, b1, w2, b2]."""
    out, h, a = forward(p, x)
    g = dact(h) * p["w2"]                       # df/db1
    return np.hstack([g * x[:, None], g, a, np.ones((len(x), 1))]), out


def train_map(p, x, y, weight_decay, steps, lr):
    """Pełnowsadowy Adam na MSE + weight decay (deterministyczny)."""
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(x)
    for t in range(1, steps + 1):
        out, h, a = forward(p, x)
        r = 2.0 * (out - y) / n                 # dMSE/dout
        g_w2 = a.T @ r
        g_b2 = np.array([r.sum()])
        d_h = np.outer(r, p["w2"]) * dact(h)
        grads = {"W1": d_h.T @ x, "b1": d_h.sum(0), "w2": g_w2, "b2": g_b2}
        for k in p:
            grads[k] = grads[k] + 2.0 * weight_decay * p[k]
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            p[k] -= lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
    return p


# ------------------------------------------------------------ Laplace + marglik
def log_evidence(log_s2, log_sp2, resid, theta, eig):
    """Laplace'owska aproksymacja log p(D), wyrazy 2*pi się skracają."""
    s2, sp2 = np.exp(log_s2), np.exp(log_sp2)
    n, P = len(resid), len(theta)
    log_lik = -0.5 * n * np.log(2 * np.pi * s2) - 0.5 * resid @ resid / s2
    log_prior = -0.5 * P * np.log(sp2) - 0.5 * theta @ theta / sp2
    log_det_prec = np.sum(np.log(eig / s2 + 1.0 / sp2))
    return log_lik + log_prior - 0.5 * log_det_prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="logistic", choices=FUNCTIONS)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=50)
    ap.add_argument("--activation", default="tanh", choices=["relu", "tanh"])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="img3_8.png")
    args = ap.parse_args()

    global ACT
    ACT = args.activation
    rng = np.random.default_rng(args.seed)
    f = FUNCTIONS[args.func]

    # dane: trening [0, 6], ewaluacja [-2, 10]
    x_tr = np.sort(rng.uniform(0.0, 6.0, args.n_train))
    y_tr = f(x_tr) + args.noise * rng.normal(size=args.n_train)
    x_te = np.linspace(-2.0, 10.0, 300)

    # standaryzacja (predykcja wraca do skali oryginalnej)
    xm, xs = x_tr.mean(), x_tr.std()
    ym, ys = y_tr.mean(), y_tr.std()
    xt, yt = (x_tr - xm) / xs, (y_tr - ym) / ys
    xe = (x_te - xm) / xs

    # 1. MAP
    p = train_map(init_params(rng, args.hidden), xt, yt,
                  args.weight_decay, args.steps, args.lr)
    theta = flatten(p)
    J_tr, out_tr = jacobian(p, xt)
    resid = yt - out_tr

    # 2-3. GGN i marglik po (sigma^2, sigma_p^2)
    JtJ = J_tr.T @ J_tr
    eig = np.clip(np.linalg.eigvalsh(JtJ), 0.0, None)
    x0 = np.array([np.log(max(resid.var(), 1e-4)), 0.0])
    res = minimize(lambda z: -log_evidence(z[0], z[1], resid, theta, eig),
                   x0, method="L-BFGS-B", bounds=[(-12, 4), (-8, 8)])
    s2, sp2 = np.exp(res.x)
    Sigma = np.linalg.inv(JtJ / s2 + np.eye(len(theta)) / sp2)

    # 4. predykcja zlinearyzowana
    J_te, mean_std = jacobian(p, xe)
    mean_net, _, _ = forward(p, xe)
    assert np.array_equal(mean_std, mean_net), "średnia musi być wyjściem sieci MAP"
    var_epi = np.einsum("ij,jk,ik->i", J_te, Sigma, J_te)
    std = np.sqrt(var_epi + s2) * ys
    mean = mean_std * ys + ym

    print(f"funkcja={args.func}  aktywacja={ACT}  P={len(theta)}  "
          f"sigma (marglik, skala y)={np.sqrt(s2) * ys:.3f}  "
          f"(prawdziwa {args.noise})  prior precision={1 / sp2:.3g}")

    # rysunek
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(0, 6, color="0.93", zorder=0, label="training interval")
    ax.fill_between(x_te, mean - 2 * std, mean + 2 * std,
                    color="C0", alpha=0.25, lw=0, label=r"$\pm 2$ predictive std")
    ax.plot(x_te, f(x_te), "k--", lw=1.2, label="true function")
    ax.plot(x_te, mean, color="C0", lw=1.8, label="predictive mean")
    ax.scatter(x_tr, y_tr, s=12, color="k", zorder=3, label="training data")
    ax.set_xlim(-2, 10)
    lo = min(f(x_te).min(), (mean - 2 * std).min())
    hi = max(f(x_te).max(), (mean + 2 * std).max())
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False,
              handlelength=1.5, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print("zapisano", args.out)


if __name__ == "__main__":
    main()
