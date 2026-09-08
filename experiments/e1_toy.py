"""E1 - 1D synthetic sine problem: six methods, metrics and figures.

    .venv/Scripts/python.exe experiments/e1_toy.py            # the real run
    .venv/Scripts/python.exe experiments/e1_toy.py --quick    # plumbing check only

Configuration is fixed by DEC-001..DEC-010 in `docs/experiment_notes.md` and is
not re-opened here. The CLI flags vary what a diagnostic run does, never what
the reported run does: `--quick` cuts epochs and seeds so hard that its output
is NOT A RESULT (CLAUDE.md), and it says so on every line it prints.

**The one structural rule this script exists to enforce (N-006 / DEC-007):**
each method is fitted once per seed and predicted **once**, on a single array -
the full test grid. The figure data and the metric data are then sliced out of
that one `Prediction`. MC dropout's predictions depend on the size of the
evaluation batch (`F.dropout` draws `n * hidden` variates in one go, so a given
row's mask depends on how many other rows were in the call), which means two
separate `predict()` calls would put a different band in the figure from the
`mpiw95` in the table, with nothing to flag it. Predicting once removes that by
construction rather than by hoping the two agree.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# `python experiments/e1_toy.py` puts `experiments/` on sys.path, not the
# repository root, so the `src.*` imports below would fail on the documented
# invocation (README). Prepending the root here keeps that command working
# without requiring `-m experiments.e1_toy` or an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import (  # noqa: E402
    EXTRAPOLATION, IN_RANGE, N_TEST, N_TRAIN, NOISE_SIGMA, REGIONS, TRAIN_RANGE,
    make_toy_data, true_function,
)
from src.experiment_io import append_rows, parse_overrides, variant_label  # noqa: E402
from src.methods import METHODS  # noqa: E402
from src.methods.backbone import DEFAULT_EPOCHS  # noqa: E402
from src.metrics import CALIBRATION_LEVELS, METRIC_COLUMNS, Z95, compute_metrics  # noqa: E402
from src.seeding import DEFAULT_TORCH_THREADS, TORCH_THREADS_ENV  # noqa: E402
from src.style import METHOD_ORDER, apply_style, plot_predictive_panel, shared_limits  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "e1_toy_metrics.csv"
FIGURE_DIR = REPO_ROOT / "figures"
FIGURE_PREFIX = "e1_toy"

DEFAULT_SEEDS = tuple(range(20))  # DEC-006 (revised: 20, matching E2's 20 splits)
FIGURE_SEED = 0                   # DEC-006
QUICK_EPOCHS = 30

# `--quick` overrides. Deliberately crude: the point is to execute every code
# path in seconds, not to approximate the real run. Anything produced under
# these is unreportable, by CLAUDE.md's rule on --quick.
QUICK_OVERRIDES = {
    "map": dict(epochs=QUICK_EPOCHS),
    "mcd": dict(epochs=QUICK_EPOCHS, T=10),
    "ensemble": dict(epochs=QUICK_EPOCHS, M=2),
    "bbb": dict(epochs=QUICK_EPOCHS, T=10),
    "laplace": dict(epochs=QUICK_EPOCHS),
    "gp": dict(n_restarts_optimizer=0),
}


def build_method(name, quick, overrides=None):
    """Construct a method at REPO DEFAULTS, unless `--set` overrides something.

    With no `--set`, no hyperparameter is chosen here - every default lives with
    its own method in `src/methods/`, where its basis (or its explicit lack of
    one, e.g. `T = 100`) is documented next to it. That is what makes the
    baseline run's configuration readable from the methods alone.

    `--set` exists so a sensitivity sweep does not require editing source, and
    every override is recorded in the `variant` column rather than left implicit
    in shell history. An override never silently becomes a default.
    """
    kwargs = dict(QUICK_OVERRIDES[name]) if quick else {}
    kwargs.update((overrides or {}).get(name, {}))
    return METHODS[name](**kwargs)


def run_seed(seed, method_names, quick, use_cache, overrides=None):
    """Fit and predict every method on one seed's data.

    Returns `(rows, panels, data)` where `rows` are the metric records and
    `panels` maps a method name to its un-standardised predictive arrays over
    the whole grid - sliced from the SAME `Prediction` the metrics came from.
    """
    data = make_toy_data(seed)
    rows, panels = [], {}

    for name in method_names:
        method = build_method(name, quick, overrides)

        t0 = time.perf_counter()
        method.fit(data.x_train_std, data.y_train_std, seed=seed, use_cache=use_cache)
        fit_seconds = time.perf_counter() - t0

        # SINGLE predict call on the full grid - see the module docstring.
        pred = method.predict(data.x_test_std)

        s = data.standardiser
        mean = s.unstandardise_mean(pred.mean)
        var_aleatoric = s.unstandardise_var(pred.var_aleatoric)
        var_epistemic = s.unstandardise_var(pred.var_epistemic)

        panels[name] = dict(mean=mean, var_aleatoric=var_aleatoric, var_epistemic=var_epistemic)

        for region in REGIONS:
            mask = data.regions[region]
            values = compute_metrics(
                y=data.y_test[mask],
                mean=mean[mask],
                var_aleatoric=var_aleatoric[mask],
                var_epistemic=var_epistemic[mask],
            )
            rows.append(dict(
                method=name, seed=seed, region=region, n_points=int(mask.sum()),
                **values,
                n_parameters=int(method.n_parameters), fit_seconds=round(fit_seconds, 3),
            ))

        last = rows[-1]
        print("  seed {}  {:<9} fit {:6.1f}s  sigma_fitted {:.4f}  overall LL {:+.3f}".format(
            seed, name, fit_seconds, last["sigma_fitted"], last["ll"]))

    return rows, panels, data


def make_figures(panels, data, quick, variant=""):
    """One figure per method plus a combined 2x3 panel, shared X and Y limits.

    Shared limits across all panels, computed from every method's band with
    nothing clipped (`style.shared_limits`) - a clipped band would read as a
    narrow one and invert the comparison the figure exists to make.
    """
    apply_style()
    names = [n for n in METHOD_ORDER if n in panels]
    y_true = true_function(data.x_test_flat)

    bands = [data.y_train, y_true]
    for p in panels.values():
        sd = np.sqrt(p["var_aleatoric"] + p["var_epistemic"])
        bands += [p["mean"] - 2.0 * sd, p["mean"] + 2.0 * sd]
    xlim, ylim = shared_limits(data.x_test_flat, bands)

    suffix = "_QUICK_NOT_A_RESULT" if quick else ""
    if variant:
        # a variant must never overwrite the baseline figures the thesis uses
        suffix += "_VARIANT"
    written = []

    def draw(ax, name, show_legend):
        p = panels[name]
        plot_predictive_panel(
            ax, name, data.x_test_flat, p["mean"], p["var_aleatoric"], p["var_epistemic"],
            data.x_train, data.y_train, y_true, TRAIN_RANGE, show_legend=show_legend,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    for name in names:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        draw(ax, name, show_legend=True)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        path = FIGURE_DIR / "{}_{}{}.png".format(FIGURE_PREFIX, name, suffix)
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        draw(ax, name, show_legend=(name == names[0]))
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("$x$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$y$")
    fig.suptitle(
        "E1 toy problem, seed {}: predictive distributions "
        r"(inner band $\pm2\sigma$ aleatoric, outer $\pm2\sigma$ total)".format(data.seed)
        + ("  [QUICK RUN - NOT A RESULT]" if quick else "")
    )
    path = FIGURE_DIR / "{}_all{}.png".format(FIGURE_PREFIX, suffix)
    fig.savefig(path)
    plt.close(fig)
    written.append(path)
    return written


def summarise(df):
    """Mean +/- std over seeds, per method and region (DEC-006)."""
    agg = df.groupby(["method", "region"])[list(METRIC_COLUMNS)].agg(["mean", "std"])
    agg.columns = ["{}_{}".format(m, s) for m, s in agg.columns]
    return agg.reset_index()


def print_summary(summary, n_seeds):
    for region in REGIONS:
        block = summary[summary.region == region]
        present = [m for m in METHOD_ORDER if m in set(block.method)]
        block = block.set_index("method").reindex(present)
        print("\n### {}  (mean +/- std over {} seeds, original y units)\n".format(region, n_seeds))
        print("| method | RMSE | LL | PICP@95 | MPIW@95 | sigma_fitted | cal_err | cal_bias |")
        print("|---|---|---|---|---|---|---|---|")
        for name, r in block.iterrows():
            def cell(metric, digits=3):
                sd = r["{}_std".format(metric)]
                sd_txt = "n/a" if pd.isna(sd) else "{:.{d}f}".format(sd, d=digits)
                return "{:.{d}f} +/- {}".format(r["{}_mean".format(metric)], sd_txt, d=digits)
            print("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name, cell("rmse"), cell("ll"), cell("picp95"), cell("mpiw95"),
                cell("sigma_fitted", 4), cell("cal_err", 4), cell("cal_bias", 4)))


def print_sanity_check(summary):
    """The band-expansion ordering that must hold before any number is believed.

    Expected (N-007, and the literature it agrees with): `gp` and `laplace`
    expand strongly outside the training range, `ensemble` a little, `bbb`
    barely at all, `map` not at all. If this ordering is absent, suspect the
    pipeline before the methods.
    """
    wide = summary.pivot(index="method", columns="region", values="mpiw95_mean")
    wide["ratio"] = wide[EXTRAPOLATION] / wide[IN_RANGE]
    wide = wide.sort_values("ratio", ascending=False)
    print("\n### Sanity check - band expansion outside the training range\n")
    print("| method | MPIW in-range | MPIW extrapolation | ratio |")
    print("|---|---|---|---|")
    for name, r in wide.iterrows():
        print("| {} | {:.3f} | {:.3f} | {:.2f}x |".format(
            name, r[IN_RANGE], r[EXTRAPOLATION], r["ratio"]))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--methods", nargs="+", default=list(METHOD_ORDER), choices=list(METHODS))
    parser.add_argument("--quick", action="store_true",
                        help="plumbing check only - cuts epochs and seeds; the output is NOT a result")
    parser.add_argument("--no-cache", action="store_true",
                        help="force a real retrain (src/methods/cache.py)")
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="METHOD:PARAM=VALUE",
        help="override one method hyperparameter, repeatable (e.g. --set mcd:dropout_p=0.2). "
             "Marks the run as a VARIANT (see the `variant` column); a variant run is a sensitivity check, not a result of E1.")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    variant = variant_label(overrides)
    seeds = [args.seeds[0]] if args.quick else args.seeds
    threads = int(os.environ.get(TORCH_THREADS_ENV, DEFAULT_TORCH_THREADS))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("E1 toy problem | run {} | seeds {} | methods {}".format(run_id, seeds, args.methods))
    print("N_train {}, grid {} on [-2, 8], sigma {}, epochs {}, z {:.6f}, torch threads {}, cache {}".format(
        N_TRAIN, N_TEST, NOISE_SIGMA, QUICK_EPOCHS if args.quick else DEFAULT_EPOCHS,
        Z95, threads, "off" if args.no_cache else "on"))
    if args.quick:
        print("*** --quick: reduced epochs/seeds. THIS OUTPUT IS NOT A RESULT. ***")
    if variant:
        print("*** VARIANT RUN: {} ***".format(variant))
        print("    A variant run is a sensitivity check (DEC-014), not a result of E1.")

    figure_seed = seeds[0] if args.quick else FIGURE_SEED
    all_rows, figure_panels, figure_data = [], None, None
    t0 = time.perf_counter()
    for seed in seeds:
        rows, panels, data = run_seed(seed, args.methods, args.quick,
                                      use_cache=not args.no_cache, overrides=overrides)
        all_rows += rows
        if seed == figure_seed:
            figure_panels, figure_data = panels, data
    wall = time.perf_counter() - t0

    df = pd.DataFrame(all_rows)
    df.insert(0, "run_id", run_id)
    metadata = dict(variant=variant, n_train=N_TRAIN, grid_points=N_TEST, noise_sigma=NOISE_SIGMA,
                    epochs=QUICK_EPOCHS if args.quick else DEFAULT_EPOCHS,
                    z=Z95, cal_levels="|".join(str(v) for v in CALIBRATION_LEVELS),
                    torch_threads=threads, quick=args.quick, cache=not args.no_cache)
    for col, val in metadata.items():
        df[col] = val

    args.out.parent.mkdir(parents=True, exist_ok=True)
    append_rows(df, args.out)
    print("\nappended {} rows to {}".format(len(df), args.out))

    summary = summarise(df)
    print_summary(summary, len(seeds))
    print_sanity_check(summary)

    if not args.no_figures and figure_panels is not None:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        written = make_figures(figure_panels, figure_data, args.quick, variant)
        print("\nwrote {} figures to {}".format(len(written), FIGURE_DIR))

    print("\ntotal wall clock {:.1f}s".format(wall))


if __name__ == "__main__":
    main()
