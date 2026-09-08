"""One MC dropout configuration on the E1 toy problem, against the GP reference.

    .venv/Scripts/python.exe mc_dropout_experiments/run_single.py

Defaults to `activation="relu"`, `dropout_p=0.005` — the configuration asked
for, and the one gal2016 actually uses on the smaller UCI sets (N-015), against
this project's `mcd` default of tanh / 0.1.

**This is a variant run, not a result of E1.** The E1 table compares posterior
approximations across one shared backbone (`src/methods/backbone.py`); changing
the activation changes the network, so a row from here can never be placed
beside the other five. It is its own study: how close MC dropout gets to the
exact GP when the network is allowed to move.

Nothing outside this directory is written. `src/` is imported read-only, and
the model cache is redirected below so the shared `cache/` is not added to.
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.methods import cache  # noqa: E402

# Redirect the model cache into this directory BEFORE any method fits. Both
# `cache.cache_path` and `cache.save` read the module attribute at call time,
# so rebinding it here is enough — no file under the shared `cache/` is read
# or written by this script.
cache.CACHE_DIR = HERE / "cache"

import matplotlib.pyplot as plt  # noqa: E402

from src.data import (  # noqa: E402
    N_TEST, N_TRAIN, NOISE_SIGMA, REGIONS, TRAIN_RANGE, make_toy_data, true_function,
)
from src.methods.backbone import DEFAULT_EPOCHS  # noqa: E402
from src.methods.gp import GPMethod  # noqa: E402
from src.methods.mcd import MCDropoutMethod  # noqa: E402
from src.metrics import METRIC_COLUMNS, Z95, compute_metrics  # noqa: E402
from src.seeding import DEFAULT_TORCH_THREADS  # noqa: E402
from src.style import apply_style, plot_predictive_panel, shared_limits  # noqa: E402

RESULTS_DIR = HERE / "results"
FIGURE_DIR = HERE / "figures"
FIGURE_SEED = 0  # matches E1's figure seed (DEC-006), so the panels are comparable


def build_variants(activation, dropout_p, hidden, depth, epochs, T):
    """`(label, panel_key, title, factory)` for each column of the figure.

    `panel_key` is the name `src/style.py` keys its colour and label tables by.
    The title is overridden after drawing, since two of the three columns are
    the same method under different settings and would otherwise carry the same
    label.
    """
    return [
        (
            "mcd_default",
            "mcd",
            "MC dropout — repo default (tanh, $p$=0.1)",
            lambda: MCDropoutMethod(epochs=epochs, T=T),
        ),
        (
            "mcd_variant",
            "mcd",
            "MC dropout — {}, $p$={:g}".format(activation, dropout_p),
            lambda: MCDropoutMethod(activation=activation, dropout_p=dropout_p,
                                    hidden=hidden, depth=depth, epochs=epochs, T=T),
        ),
        ("gp", "gp", "Gaussian process (reference)", GPMethod),
    ]


def run_seed(seed, variants, use_cache):
    """Fit and predict every variant on one seed's data.

    One `fit` and one `predict` per variant, on the full test grid — the figure
    and the metrics are sliced from the same `Prediction` (DEC-007). MC
    dropout's masks depend on the size of the evaluation batch (N-006), so a
    second `predict()` call would put a different band in the figure from the
    one the table reports.
    """
    data = make_toy_data(seed)
    rows, panels = [], {}

    for label, _, _, factory in variants:
        method = factory()
        t0 = time.perf_counter()
        method.fit(data.x_train_std, data.y_train_std, seed=seed, use_cache=use_cache)
        fit_seconds = time.perf_counter() - t0
        pred = method.predict(data.x_test_std)

        s = data.standardiser
        mean = s.unstandardise_mean(pred.mean)
        var_aleatoric = s.unstandardise_var(pred.var_aleatoric)
        var_epistemic = s.unstandardise_var(pred.var_epistemic)
        panels[label] = dict(mean=mean, var_aleatoric=var_aleatoric, var_epistemic=var_epistemic)

        for region in REGIONS:
            mask = data.regions[region]
            rows.append(dict(
                variant=label, seed=seed, region=region, n_points=int(mask.sum()),
                **compute_metrics(
                    y=data.y_test[mask], mean=mean[mask],
                    var_aleatoric=var_aleatoric[mask], var_epistemic=var_epistemic[mask],
                ),
                n_parameters=int(method.n_parameters), fit_seconds=round(fit_seconds, 3),
            ))
        print("  seed {}  {:<12} fit {:6.1f}s  sigma_fitted {:.4f}  overall LL {:+.3f}".format(
            seed, label, fit_seconds, rows[-1]["sigma_fitted"], rows[-1]["ll"]))

    return rows, panels, data


def make_figure(panels, data, variants, suffix):
    """Three panels on shared limits, in `src/style.py`'s E1 conventions.

    Shared, unclipped limits (`style.shared_limits`): the default
    configuration's band is the widest thing on the figure, and clipping it
    would read as a narrow band and invert the comparison.
    """
    apply_style()
    y_true = true_function(data.x_test_flat)

    bands = [data.y_train, y_true]
    for p in panels.values():
        sd = np.sqrt(p["var_aleatoric"] + p["var_epistemic"])
        bands += [p["mean"] - 2.0 * sd, p["mean"] + 2.0 * sd]
    xlim, ylim = shared_limits(data.x_test_flat, bands)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9), sharex=True, sharey=True)
    for ax, (label, panel_key, title, _) in zip(axes, variants):
        p = panels[label]
        plot_predictive_panel(
            ax, panel_key, data.x_test_flat, p["mean"], p["var_aleatoric"], p["var_epistemic"],
            data.x_train, data.y_train, y_true, TRAIN_RANGE, show_legend=(ax is axes[0]),
        )
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    fig.suptitle(
        "MC dropout configuration variant, toy problem seed {} "
        r"(inner band $\pm2\sigma$ aleatoric, outer $\pm2\sigma$ total)".format(data.seed)
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "mcd_variant_vs_gp{}.png".format(suffix)
    fig.savefig(path)
    plt.close(fig)
    return path


def print_summary(df, variants, n_seeds):
    order = [v[0] for v in variants]
    agg = df.groupby(["variant", "region"])[list(METRIC_COLUMNS)].agg(["mean", "std"])
    agg.columns = ["{}_{}".format(m, s) for m, s in agg.columns]
    agg = agg.reset_index()

    for region in REGIONS:
        block = agg[agg.region == region].set_index("variant").reindex(order)
        print("\n### {}  (mean +/- std over {} seeds, original y units)\n".format(region, n_seeds))
        print("| config | RMSE | LL | PICP@95 | MPIW@95 | sigma_fitted | cal_err | cal_bias |")
        print("|---|---|---|---|---|---|---|---|")
        for name, r in block.iterrows():
            def cell(metric, digits=3):
                sd = r["{}_std".format(metric)]
                sd_txt = "n/a" if pd.isna(sd) else "{:.{d}f}".format(sd, d=digits)
                return "{:.{d}f} +/- {}".format(r["{}_mean".format(metric)], sd_txt, d=digits)
            print("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name, cell("rmse"), cell("ll"), cell("picp95"), cell("mpiw95"),
                cell("sigma_fitted", 4), cell("cal_err", 4), cell("cal_bias", 4)))

    wide = agg.pivot(index="variant", columns="region", values="mpiw95_mean").reindex(order)
    print("\n### Band expansion outside the training range\n")
    print("| config | MPIW in-range | MPIW extrapolation | ratio |")
    print("|---|---|---|---|")
    for name, r in wide.iterrows():
        print("| {} | {:.3f} | {:.3f} | {:.2f}x |".format(
            name, r["in_range"], r["extrapolation"], r["extrapolation"] / r["in_range"]))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activation", default="relu", choices=("relu", "tanh"))
    parser.add_argument("--dropout-p", type=float, default=0.005)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    variants = build_variants(args.activation, args.dropout_p, args.hidden,
                              args.depth, args.epochs, args.T)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "{}_p{:g}_h{}_d{}".format(args.activation, args.dropout_p, args.hidden, args.depth)

    print("MC dropout variant | run {} | seeds {} | {}".format(run_id, args.seeds, tag))
    print("N_train {}, grid {} on [-2, 8], sigma {}, epochs {}, T {}, z {:.6f}, torch threads {}".format(
        N_TRAIN, N_TEST, NOISE_SIGMA, args.epochs, args.T, Z95, DEFAULT_TORCH_THREADS))
    print("*** VARIANT RUN: not a result of E1 — the backbone differs from the shared one. ***")

    all_rows, figure_panels, figure_data = [], None, None
    t0 = time.perf_counter()
    for seed in args.seeds:
        rows, panels, data = run_seed(seed, variants, use_cache=not args.no_cache)
        all_rows += rows
        if seed == FIGURE_SEED or figure_panels is None:
            figure_panels, figure_data = panels, data
    wall = time.perf_counter() - t0

    df = pd.DataFrame(all_rows)
    df.insert(0, "run_id", run_id)
    for col, val in dict(
        activation=args.activation, dropout_p=args.dropout_p, hidden=args.hidden,
        depth=args.depth, T=args.T, epochs=args.epochs, n_train=N_TRAIN,
        grid_points=N_TEST, noise_sigma=NOISE_SIGMA, z=Z95,
        torch_threads=DEFAULT_TORCH_THREADS, cache=not args.no_cache,
    ).items():
        df[col] = val

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "mcd_variant_{}.csv".format(tag)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    print("\nappended {} rows to {}".format(len(df), out.relative_to(REPO_ROOT)))

    print_summary(df, variants, len(args.seeds))
    path = make_figure(figure_panels, figure_data, variants, "_" + tag)
    print("\nwrote figure {}".format(path.relative_to(REPO_ROOT)))
    print("total wall clock {:.1f}s".format(wall))


if __name__ == "__main__":
    main()
