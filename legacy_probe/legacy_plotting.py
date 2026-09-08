"""The pre-rewrite `plot_uncertainty_for_1d_data`, recovered verbatim with
`git show 28b7615^:src/plotting.py`.

The only addition is the optional `out_path` argument, so the probe can also
drop the PNGs on disk. Everything that determines what the figure LOOKS like —
figure size, colours, the +-2 sigma band drawn from the TOTAL standard deviation,
the dashed line being the CLEAN sin(x) — is unchanged from the version that
produced the old notebook outputs.
"""
from pathlib import Path

import matplotlib.pyplot as plt


def plot_uncertainty_for_1d_data(X_train, y_train, X_eval, y_eval, y_pred, y_std,
                                 title="", out_path=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(X_train, y_train, s=15, alpha=0.6, label="train", color="C0")
    if y_eval is not None:
        ax.plot(X_eval, y_eval, "k--", alpha=0.4, label="true function")
    ax.plot(X_eval, y_pred, "r-", linewidth=2, label="prediction")
    ax.fill_between(X_eval.ravel(),
                    y_pred - 2 * y_std,
                    y_pred + 2 * y_std,
                    alpha=0.25, color="red", label="±2σ")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    plt.show()
