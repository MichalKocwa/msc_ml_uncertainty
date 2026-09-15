r"""Architecture sensitivity sweep on the E1 sine problem.

    ..\.venv\Scripts\python.exe run_sweep.py --quick        # plumbing check
    ..\.venv\Scripts\python.exe run_sweep.py                # stage 1, seed 0
    ..\.venv\Scripts\python.exe run_sweep.py --seeds 0 1 2 --configs d1_h50_tanh

**What this is.** A sensitivity sweep over the shared backbone's architecture
knobs (`depth`, `activation`; width is fixed, see `sweep_config.HIDDEN`),
applied identically to every network-based method. It is NOT tuning and its
output is not a result of E1. The distinction is DEC-014's: varying a setting
for all methods at once and reporting the whole sweep is a measurement; varying
it and reporting only the configuration whose picture looked best is selection
on the evaluation set, which E1's design cannot support because its test grid
IS its evaluation set.

**Why architecture is worth sweeping here.** `DEFAULT_DEPTH = 1` is not an
arbitrary default: it is the setting Foong et al. (2019) prove their mean-field
result for, which is what makes their prediction testable on this problem. So
depth is load-bearing for the baseline and must not be quietly replaced. It is
also the knob that decided whether a deep ensemble's band widens outside the
training range at all, so its effect on the other methods is an open question.

**Isolation.** Everything this script writes goes under `arch_sweep/`. It
imports from `src/` read-only, constructs methods through the same `METHODS`
registry as `experiments/e1_toy.py`, and passes the architecture as explicit
keyword arguments -- so no value in `sweep_config.py` can become a repo default.
It never touches the repository's own `results/` or `figures/`.

**Protocol, inherited from `experiments/e1_toy.py` unchanged.** `set_seed`
before data and models; fit on the standardised arrays; a single `predict` per
method per seed; un-standardise before scoring; metrics against the noisy
`y_test` in original y units, split by the project's own region masks.
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from src.data import REGIONS, TRAIN_RANGE, make_toy_data, true_function  # noqa: E402
from src.experiment_io import append_rows  # noqa: E402
from src.methods import METHODS  # noqa: E402
from src.metrics import METRIC_COLUMNS, compute_metrics  # noqa: E402
from src.seeding import set_seed  # noqa: E402
from src.style import apply_style, plot_predictive_panel, shared_limits  # noqa: E402

import sweep_config  # noqa: E402

RESULTS_CSV = HERE / "results" / "arch_sweep_metrics.csv"
FIGURE_DIR = HERE / "figures"

# Plumbing-check overrides, same spirit as e1_toy.py's QUICK_OVERRIDES: execute
# every code path in seconds. Anything produced under --quick is unreportable.
QUICK_EPOCHS = 20
QUICK_OVERRIDES = {
    "map": dict(epochs=QUICK_EPOCHS),
    "mcd": dict(epochs=QUICK_EPOCHS, T=5),
    "ensemble": dict(epochs=QUICK_EPOCHS, M=2),
    "bbb": dict(epochs=QUICK_EPOCHS, T=5),
    "laplace": dict(epochs=QUICK_EPOCHS),
    "gp": dict(n_restarts_optimizer=0),
}


def build(name, architecture, quick):
    """Construct one method at repo defaults plus this configuration's architecture.

    `architecture` is empty for the reference methods, which take no such
    arguments -- that is the one asymmetry in the sweep and the reason `gp`'s
    rows carry blank architecture columns rather than invented ones.
    """
    kwargs = dict(QUICK_OVERRIDES[name]) if quick else {}
    kwargs.update(architecture)
    return METHODS[name](**kwargs)


def run_one(name, architecture, data, seed, quick, use_cache):
    """Fit, predict, un-standardise. Returns predictions in original y units."""
    method = build(name, architecture, quick)

    t0 = time.perf_counter()
    method.fit(data.x_train_std, data.y_train_std, seed=seed, use_cache=use_cache)
    fit_seconds = time.perf_counter() - t0

    pred = method.predict(data.x_test_std)   # ONE predict call, as in e1_toy.py

    s = data.standardiser
    return {
        "mean": s.unstandardise_mean(pred.mean),
        "var_aleatoric": s.unstandardise_var(pred.var_aleatoric),
        "var_epistemic": s.unstandardise_var(pred.var_epistemic),
        "fit_seconds": fit_seconds,
        "n_parameters": int(method.n_parameters),
    }


def metric_rows(name, config, panel, data, seed):
    rows = []
    for region in REGIONS:
        mask = data.regions[region]
        values = compute_metrics(
            y=data.y_test[mask],
            mean=panel["mean"][mask],
            var_aleatoric=panel["var_aleatoric"][mask],
            var_epistemic=panel["var_epistemic"][mask],
        )
        rows.append(dict(
            method=name, seed=seed, region=region, n_points=int(mask.sum()),
            config_id=config["config_id"], depth=config["depth"],
            hidden=config["hidden"], activation=config["activation"],
            **values,
            n_parameters=panel["n_parameters"],
            fit_seconds=round(panel["fit_seconds"], 3),
        ))
    return rows


def _bands(panels, data, y_true):
    out = [data.y_train, y_true]
    for p in panels:
        sd = np.sqrt(p["var_aleatoric"] + p["var_epistemic"])
        out += [p["mean"] - 2.0 * sd, p["mean"] + 2.0 * sd]
    return out


def figure_master(panels, data, seed, quick, row_limits, written):
    """ONE image: rows are configurations, columns are methods.

    Reading across a row compares the methods at a fixed architecture; reading
    down a column compares one method across architectures. Both comparisons the
    sweep exists to make are therefore in a single file, which is the point --
    flipping between separate PNGs is how a difference gets missed.

    `gp` is repeated in every row. It has no architecture, so the panel is
    identical each time; it is drawn anyway because a row is far easier to read
    against the one exact posterior in the comparison than against nothing.

    Y limits are shared across ALL panels by default, so a band in one row is
    directly comparable to a band in another. `row_limits=True` shares them
    within each row instead -- worth using if one configuration's band is so
    wide that it flattens every other panel. Nothing is ever clipped either way
    (`src/style.shared_limits`): a squashed panel is honest, a cropped one lies.
    """
    y_true = true_function(data.x_test_flat)
    configs = [c for c in sweep_config.grid() if c["config_id"] in panels]
    methods = [m for m in (*sweep_config.NN_METHODS, *sweep_config.REFERENCE_METHODS)
               if any(m in panels[c["config_id"]] for c in configs)]

    if not row_limits:
        every = [panels[c["config_id"]][m] for c in configs for m in methods
                 if m in panels[c["config_id"]]]
        global_xlim, global_ylim = shared_limits(data.x_test_flat, _bands(every, data, y_true))

    n_rows, n_cols = len(configs), len(methods)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.2 * n_rows),
                             sharex=True, sharey=not row_limits, squeeze=False)

    for i, config in enumerate(configs):
        cid = config["config_id"]
        row_panels = [panels[cid][m] for m in methods if m in panels[cid]]
        if row_limits:
            xlim, ylim = shared_limits(data.x_test_flat, _bands(row_panels, data, y_true))
        else:
            xlim, ylim = global_xlim, global_ylim

        for j, name in enumerate(methods):
            ax = axes[i][j]
            if name not in panels[cid]:
                ax.text(0.5, 0.5, "not run\n(intractable)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="#999999")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            plot_predictive_panel(
                ax, name, data.x_test_flat, panels[cid][name]["mean"],
                panels[cid][name]["var_aleatoric"], panels[cid][name]["var_epistemic"],
                data.x_train, data.y_train, y_true, TRAIN_RANGE,
                show_legend=(i == 0 and j == 0),
            )
            ax.set_title("")           # the helper writes the method label; use headers instead
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if i == 0:
                header = name + (" (reference)" if name in sweep_config.REFERENCE_METHODS else "")
                ax.set_title(header, fontsize=13, pad=10)

        axes[i][0].set_ylabel("{}\n\n$y$".format(cid), fontsize=11)

    for ax in axes[-1]:
        ax.set_xlabel("$x$")

    fig.suptitle(
        "Architecture sweep on the E1 sine, seed {} - rows: configuration, columns: method"
        "   (inner band $\\pm2\\sigma$ aleatoric, outer total)".format(seed)
        + ("   [QUICK - NOT A RESULT]" if quick else ""),
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = FIGURE_DIR / "arch_sweep_seed{}{}{}.png".format(
        seed, "_rowlimits" if row_limits else "", "_QUICK" if quick else "")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)
    return path


def summarise(df):
    """MPIW expansion ratio per method and configuration -- the sweep's headline.

    `mpiw95` outside the training range divided by `mpiw95` inside it. A method
    whose ratio sits at 1.0 reports the same interval two units past the data as
    it does among the data, which is the failure this sweep is looking for.
    """
    wide = df[df.region.isin(("in_range", "extrapolation"))].pivot_table(
        index=["method", "config_id"], columns="region", values="mpiw95", aggfunc="mean")
    wide["ratio"] = wide["extrapolation"] / wide["in_range"]
    return wide.reset_index()


def print_summary(df):
    """ASCII only: this prints to a Windows console whose code page mangles
    anything else, and a mangled table is worse than a plain one."""
    summary = summarise(df)
    present = set(summary.config_id)
    configs = [c["config_id"] for c in sweep_config.grid() if c["config_id"] in present]

    print("\n### MPIW expansion ratio (extrapolation / in-range), higher is better\n")
    print("| method | " + " | ".join(configs) + " |")
    print("|---" * (len(configs) + 1) + "|")
    for name in sweep_config.NN_METHODS:
        block = summary[summary.method == name].set_index("config_id")
        if block.empty:
            continue
        cells = ["{:.2f}".format(block.loc[c, "ratio"]) if c in block.index else "n/a"
                 for c in configs]
        print("| {} | ".format(name) + " | ".join(cells) + " |")

    # Reference methods have no architecture: one number, not a row of repeats
    # that would invite reading a trend into a constant.
    for name in sweep_config.REFERENCE_METHODS:
        block = summary[summary.method == name]
        if not block.empty:
            print("\nreference: {} ratio {:.2f} (no architecture, one value)".format(
                name, float(block["ratio"].iloc[0])))

    baseline = sweep_config.baseline_config_id()
    if baseline in present:
        print("\nbaseline configuration {} reproduces E1 - check it against "
              "results/e1_toy_metrics.csv before believing the rest.".format(baseline))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[sweep_config.FIGURE_SEED])
    parser.add_argument("--configs", nargs="+", default=None,
                        help="config_id values to run; default is the whole grid")
    parser.add_argument("--methods", nargs="+", default=list(sweep_config.NN_METHODS))
    parser.add_argument("--no-reference", action="store_true", help="skip the gp reference")
    parser.add_argument("--quick", action="store_true",
                        help="plumbing check only - cuts epochs; the output is NOT a result")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--row-limits", action="store_true",
                        help="share y limits within each row instead of across the whole figure")
    parser.add_argument("--use-cache", action="store_true",
                        help="reuse cached fits; makes fit_seconds meaningless")
    parser.add_argument("--out", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    grid = sweep_config.grid()
    if args.configs:
        known = {c["config_id"] for c in grid}
        unknown = set(args.configs) - known
        if unknown:
            raise SystemExit("ERROR: unknown config_id {}".format(sorted(unknown)))
        grid = [c for c in grid if c["config_id"] in set(args.configs)]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("architecture sweep | run {} | seeds {} | {} configs | methods {}".format(
        run_id, args.seeds, len(grid), args.methods), flush=True)
    print("baseline configuration (reproduces E1): {}".format(
        sweep_config.baseline_config_id()), flush=True)
    if args.quick:
        print("*** --quick: {} epochs. THIS OUTPUT IS NOT A RESULT. ***".format(QUICK_EPOCHS))

    all_rows, written = [], []
    t_start = time.perf_counter()

    for seed in args.seeds:
        set_seed(seed)
        data = make_toy_data(seed)
        panels = {}

        reference = {}
        if not args.no_reference:
            for name in sweep_config.REFERENCE_METHODS:
                panel = run_one(name, {}, data, seed, args.quick, args.use_cache)
                reference[name] = panel
                all_rows += metric_rows(
                    name, {"config_id": "reference", "depth": "", "hidden": "", "activation": ""},
                    panel, data, seed)
                print("  seed {}  {:<9} reference        fit {:7.1f}s".format(
                    seed, name, panel["fit_seconds"]), flush=True)

        for config in grid:
            architecture = {k: config[k] for k in ("depth", "hidden", "activation")}
            panels[config["config_id"]] = dict(reference)
            for name in args.methods:
                panel = run_one(name, architecture, data, seed, args.quick, args.use_cache)
                panels[config["config_id"]][name] = panel
                all_rows += metric_rows(name, config, panel, data, seed)
                print("  seed {}  {:<9} {:<16} fit {:7.1f}s  {} params".format(
                    seed, name, config["config_id"], panel["fit_seconds"],
                    panel["n_parameters"]), flush=True)

        if not args.no_figures:
            path = figure_master(panels, data, seed, args.quick, args.row_limits, written)
            print("  figure -> {}".format(path.name), flush=True)

    wall = time.perf_counter() - t_start

    df = pd.DataFrame(all_rows)
    df.insert(0, "run_id", run_id)
    for col, val in dict(quick=args.quick, cache=args.use_cache,
                         n_train=len(data.x_train), grid_points=len(data.x_test)).items():
        df[col] = val

    args.out.parent.mkdir(parents=True, exist_ok=True)
    append_rows(df, args.out)
    print("\nappended {} rows to {}".format(len(df), args.out))
    print("wrote {} figure(s) to {}".format(len(written), FIGURE_DIR))
    print("wall clock {:.1f} min".format(wall / 60.0))

    print_summary(df)


if __name__ == "__main__":
    main()
