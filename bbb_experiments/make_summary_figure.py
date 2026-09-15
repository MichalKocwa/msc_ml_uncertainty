"""Two-panel summary of B-002: (left) seed-0 predictive bands of the tanh and
ReLU headline configurations on one axis; (right) band ratio per seed for the
four configurations, with the E1 GP / Laplace / bbb ratios from N-014 as
dotted reference lines (orientation only — different implementation).

    .venv/Scripts/python.exe bbb_experiments/make_summary_figure.py

Reads only `bbb_experiments/results/`; writes only `bbb_experiments/figures/`.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import NOISE_SIGMA, TRAIN_RANGE  # noqa: E402
from src.style import apply_style  # noqa: E402

CONFIGS = ["repo_like", "tanh_lrt_fixed", "relu_fullbatch_K4", "relu_lrt_fixed"]
COLOURS = {"repo_like": "#9467bd", "tanh_lrt_fixed": "#c5b0d5", "relu_fullbatch_K4": "#d62728", "relu_lrt_fixed": "#ff9896"}
# N-014 (20 seeds, repo implementation) — reference lines, not sandbox numbers
E1_REFERENCE = {"gp 2.90x": 2.90, "laplace 2.30x": 2.30, "repo bbb 1.01x": 1.01}


def main():
    apply_style()
    df = pd.read_csv(HERE / "results" / "seeds5_metrics.csv")
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))

    for name in CONFIGS:
        d = np.load(HERE / "results" / "predictions" / f"seeds5_{name}_seed0.npz")
        x, mean = d["x_test"], d["mean"]
        sd = np.sqrt(d["var_aleatoric"] + d["var_epistemic"])
        ax_l.fill_between(x, mean - 2 * sd, mean + 2 * sd, color=COLOURS[name], alpha=0.25, lw=0)
        ax_l.plot(x, mean, color=COLOURS[name], lw=1.3, label=name)
    ax_l.plot(x, np.sin(x), "k--", lw=1, label="sin(x)")
    ax_l.scatter(d["x_train"], d["y_train"], s=4, color="#333333", alpha=0.45, lw=0, label="training data")
    for edge in TRAIN_RANGE:
        ax_l.axvline(edge, color="#999999", lw=0.8)
    ax_l.set_ylim(-2.6, 2.6)
    ax_l.set_xlabel("x")
    ax_l.set_ylabel("y")
    ax_l.set_title(r"seed 0 — predictive mean $\pm 2\sigma$ (total)")
    ax_l.legend(loc="upper left", ncol=2)

    for i, name in enumerate(CONFIGS):
        g = df[df.config == name]
        inr = g[g.region == "in_range"].set_index("seed")["mpiw95"]
        ext = g[g.region == "extrapolation"].set_index("seed")["mpiw95"]
        ratio = (ext / inr).sort_index()
        ax_r.scatter(np.full(len(ratio), i), ratio.values, color=COLOURS[name], s=28, zorder=3)
        ax_r.hlines(ratio.mean(), i - 0.25, i + 0.25, color=COLOURS[name], lw=2, zorder=4)
    for label, value in E1_REFERENCE.items():
        ax_r.axhline(value, color="#555555", ls=":", lw=0.9)
        ax_r.text(3.45, value, label, fontsize=7, va="bottom", ha="right", color="#555555")
    ax_r.set_xticks(range(len(CONFIGS)))
    ax_r.set_xticklabels(CONFIGS, rotation=20, ha="right")
    ax_r.set_ylabel("MPIW@95 extrapolation / in-range")
    ax_r.set_title("band ratio, 5 seeds (dots) and mean (bar)")
    ax_r.set_ylim(0.9, 3.1)

    fig.suptitle(f"BBB sandbox — tanh vs ReLU on the E1 sine (N=250, sigma={NOISE_SIGMA}, 4000 epochs)")
    fig.savefig(HERE / "figures" / "seeds5_tanh_vs_relu.png")
    print("wrote figures/seeds5_tanh_vs_relu.png")


if __name__ == "__main__":
    main()
