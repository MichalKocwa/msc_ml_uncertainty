"""E1b - the E1 sine problem with a TWO-hidden-layer backbone: metrics, saved predictions, figures.

    .venv/Scripts/python.exe experiments/e1b_toy_depth2.py            # the real run (~65 min cold)
    .venv/Scripts/python.exe experiments/e1b_toy_depth2.py --quick    # plumbing check only

A separate experiment from E1, in a separate script with separate outputs, so
the two architectures can never be mixed in one table or one figure by
accident:

    results/e1b_toy_depth2_metrics.csv              one row per (method, seed, region)
    results/e1b_toy_depth2_predictions/seed<N>.npz  every method's full predictive
                                                    arrays on the test grid, per seed
    figures/e1b_toy_depth2_<method>.png, _all.png   drawn from seed 0's saved predictions

**What changes against E1, and nothing else:** `depth=2` for the five
network-based methods (`map`, `mcd`, `ensemble`, `bbb`, `laplace`), i.e. two
hidden layers of 50 tanh units — 2702 parameters instead of 152, so
N_train/P = 0.09 instead of 1.64. `gp` has no architecture and is fitted
exactly as in E1 (same cache key, same numbers): it is the same reference in
both tables. Data (DEC-002/004/005/006), prior (gamma = 1), epochs (4000,
DEC-012), batch size, `dropout_p = 0.1` (DEC-018), `T = 100`, `M = 5`, the
metrics and the region split are all E1's, unchanged.

**Basis for the configuration (DEC-019):** author's choice after seeing the
whole depth x activation sweep in `arch_sweep/` at seed 0 — i.e. selection on
the evaluation grid, named as such. The literature reason the depth axis is
worth reporting at all is Foong et al. (2019): their mean-field limitation is
proved for one hidden layer and their universality result for two or more,
so depth 1 vs 2 is precisely the axis on which that theory distinguishes
cases. It is not a reason to prefer depth 2, and this run does not replace
E1.

**Predictions are saved, figures are derived.** Every seed's full predictive
arrays (mean, aleatoric and epistemic variance, and the per-run means for the
sampling methods) go to an `.npz`; the figures are then drawn from seed 0's
file by `src.style.make_method_figures`, and can be redrawn in any layout by
`experiments/e1b_toy_depth2_figures.py` without refitting anything. The
metrics and the figures therefore come from one array per method per seed
(DEC-007), and the figure is reproducible from the saved file alone.

The one structural rule inherited from E1 (N-006 / DEC-007): each method is
fitted once per seed and predicted ONCE, on the full test grid.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.data import (  # noqa: E402
    EXTRAPOLATION, IN_RANGE, N_TEST, N_TRAIN, NOISE_SIGMA, REGIONS, TRAIN_RANGE,
    make_toy_data, true_function,
)
from src.experiment_io import (  # noqa: E402
    METHOD_ORDER, append_rows, load_predictions, parse_overrides, save_predictions, variant_label,
)
from src.methods import METHODS  # noqa: E402
from src.methods.backbone import DEFAULT_EPOCHS  # noqa: E402
from src.metrics import CALIBRATION_LEVELS, METRIC_COLUMNS, Z95, compute_metrics  # noqa: E402
from src.seeding import DEFAULT_TORCH_THREADS, TORCH_THREADS_ENV  # noqa: E402
from src.style import make_method_figures  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "e1b_toy_depth2"
RESULTS_CSV = REPO_ROOT / "results" / "{}_metrics.csv".format(EXPERIMENT)
PREDICTIONS_DIR = REPO_ROOT / "results" / "{}_predictions".format(EXPERIMENT)
FIGURE_DIR = REPO_ROOT / "figures"

# The architecture this experiment exists for. `hidden` and `activation` are
# the repo defaults and are passed explicitly anyway, so the configuration is
# readable from this one dict and recorded in every row.
ARCH = dict(depth=2, hidden=50, activation="tanh")
ARCH_ID = "d{depth}_h{hidden}_{activation}".format(**ARCH)
NN_METHODS = ("map", "mcd", "ensemble", "bbb", "laplace")   # take ARCH
REFERENCE_METHODS = ("gp",)                                  # no architecture

DEFAULT_SEEDS = tuple(range(20))   # DEC-006 (revised): 20, as E1 and E2
FIGURE_SEED = 0                    # DEC-006, as E1
QUICK_EPOCHS = 30
QUICK_OVERRIDES = {
    "map": dict(epochs=QUICK_EPOCHS),
    "mcd": dict(epochs=QUICK_EPOCHS, T=10),
    "ensemble": dict(epochs=QUICK_EPOCHS, M=2),
    "bbb": dict(epochs=QUICK_EPOCHS, T=10),
    "laplace": dict(epochs=QUICK_EPOCHS),
    "gp": dict(n_restarts_optimizer=0),
}


def build_method(name, quick, overrides=None):
    """Repo defaults + this experiment's architecture (+ `--quick` / `--set`).

    `ARCH` is applied to the network methods only, as explicit keyword
    arguments — the same route `--set` takes — so nothing here can become a
    repo default. `--set` still marks the run a VARIANT (DEC-014).
    """
    kwargs = dict(QUICK_OVERRIDES[name]) if quick else {}
    if name in NN_METHODS:
        kwargs.update(ARCH)
    kwargs.update((overrides or {}).get(name, {}))
    return METHODS[name](**kwargs)


def predictions_path(seed, quick=False, variant=""):
    tag = ("_QUICK_NOT_A_RESULT" if quick else "") + ("_VARIANT" if variant else "")
    return PREDICTIONS_DIR / "seed{}{}.npz".format(seed, tag)


def run_seed(seed, method_names, quick, use_cache, overrides=None):
    """Fit and predict every method on one seed's data.

    Returns `(rows, panels, data)`; `panels[method]` holds the un-standardised
    arrays the metric rows were sliced from, plus `samples` where the method
    has them.
    """
    data = make_toy_data(seed)
    rows, panels = [], {}

    for name in method_names:
        method = build_method(name, quick, overrides)

        t0 = time.perf_counter()
        method.fit(data.x_train_std, data.y_train_std, seed=seed, use_cache=use_cache)
        fit_seconds = time.perf_counter() - t0

        pred = method.predict(data.x_test_std)   # ONE predict call on the full grid

        s = data.standardiser
        mean = s.unstandardise_mean(pred.mean)
        var_aleatoric = s.unstandardise_var(pred.var_aleatoric)
        var_epistemic = s.unstandardise_var(pred.var_epistemic)
        samples = None if pred.samples is None else s.unstandardise_mean(pred.samples)
        panels[name] = dict(mean=mean, var_aleatoric=var_aleatoric,
                            var_epistemic=var_epistemic, samples=samples)

        for region in REGIONS:
            mask = data.regions[region]
            values = compute_metrics(
                y=data.y_test[mask], mean=mean[mask],
                var_aleatoric=var_aleatoric[mask], var_epistemic=var_epistemic[mask],
            )
            rows.append(dict(
                method=name, seed=seed, region=region, n_points=int(mask.sum()),
                arch=ARCH_ID if name in NN_METHODS else "",
                depth=ARCH["depth"] if name in NN_METHODS else "",
                **values,
                n_parameters=int(method.n_parameters), fit_seconds=round(fit_seconds, 3),
            ))

        last = rows[-1]
        print("  seed {}  {:<9} fit {:6.1f}s  P {:>5}  sigma_fitted {:.4f}  overall LL {:+.3f}".format(
            seed, name, fit_seconds, method.n_parameters, last["sigma_fitted"], last["ll"]))

    return rows, panels, data


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
        print("\n### {}  ({}; mean +/- std over {} seeds, original y units)\n".format(
            region, ARCH_ID, n_seeds))
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
    wide = summary.pivot(index="method", columns="region", values="mpiw95_mean")
    wide["ratio"] = wide[EXTRAPOLATION] / wide[IN_RANGE]
    wide = wide.sort_values("ratio", ascending=False)
    print("\n### Band expansion outside the training range ({})\n".format(ARCH_ID))
    print("| method | MPIW in-range | MPIW extrapolation | ratio |")
    print("|---|---|---|---|")
    for name, r in wide.iterrows():
        print("| {} | {:.3f} | {:.3f} | {:.2f}x |".format(name, r[IN_RANGE], r[EXTRAPOLATION], r["ratio"]))


def figures_from_saved(seed, quick, variant, layout="both", n_sample_lines=0):
    """Draw the figures from the SAVED predictions of `seed`, never from memory,
    so the figure written here and any later redraw come from one file."""
    arrays, panels, meta = load_predictions(predictions_path(seed, quick, variant))
    suffix = ("_QUICK_NOT_A_RESULT" if quick else "") + ("_VARIANT" if variant else "")
    title = "E1b, {} , seed {}: predictive distributions (inner band $\\pm2\\sigma$ aleatoric, outer $\\pm2\\sigma$ total)".format(
        ARCH_ID, seed) + ("  [QUICK RUN - NOT A RESULT]" if quick else "")
    return make_method_figures(
        panels, arrays["x_test"], arrays["x_train"], arrays["y_train"], arrays["y_true"],
        TRAIN_RANGE, FIGURE_DIR, EXPERIMENT + suffix, suptitle=title,
        layout=layout, n_sample_lines=n_sample_lines,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--methods", nargs="+", default=list(METHOD_ORDER), choices=list(METHODS))
    parser.add_argument("--quick", action="store_true", help="plumbing check only; the output is NOT a result")
    parser.add_argument("--no-cache", action="store_true", help="force a real retrain (src/methods/cache.py)")
    parser.add_argument("--set", dest="overrides", action="append", metavar="METHOD:PARAM=VALUE",
                        help="override one hyperparameter; marks the run a VARIANT (DEC-014)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--members", type=int, default=5,
                        help="individual-run lines drawn per sampling method in the figures (0 = none)")
    parser.add_argument("--out", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    variant = variant_label(overrides)
    seeds = [args.seeds[0]] if args.quick else args.seeds
    threads = int(os.environ.get(TORCH_THREADS_ENV, DEFAULT_TORCH_THREADS))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    epochs = QUICK_EPOCHS if args.quick else DEFAULT_EPOCHS

    print("E1b sine, {} | run {} | seeds {} | methods {}".format(ARCH_ID, run_id, seeds, args.methods))
    print("N_train {}, grid {} on [-2, 8], sigma {}, epochs {}, z {:.6f}, torch threads {}, cache {}".format(
        N_TRAIN, N_TEST, NOISE_SIGMA, epochs, Z95, threads, "off" if args.no_cache else "on"))
    if args.quick:
        print("*** --quick: reduced epochs/seeds. THIS OUTPUT IS NOT A RESULT. ***")
    if variant:
        print("*** VARIANT RUN: {} *** (a sensitivity check, DEC-014, not a result of E1b)".format(variant))

    figure_seed = seeds[0] if args.quick else FIGURE_SEED
    all_rows = []
    t0 = time.perf_counter()
    for seed in seeds:
        rows, panels, data = run_seed(seed, args.methods, args.quick, use_cache=not args.no_cache, overrides=overrides)
        all_rows += rows
        save_predictions(
            predictions_path(seed, args.quick, variant),
            arrays=dict(x_test=data.x_test_flat, y_test=data.y_test, y_true=true_function(data.x_test_flat),
                        x_train=data.x_train.ravel(), y_train=data.y_train),
            panels=panels,
            meta=dict(experiment=EXPERIMENT, arch=ARCH_ID, run_id=run_id, seed=seed, epochs=epochs,
                      variant=variant, quick=args.quick, noise_sigma=NOISE_SIGMA, torch_threads=threads),
        )
    wall = time.perf_counter() - t0

    df = pd.DataFrame(all_rows)
    df.insert(0, "run_id", run_id)
    metadata = dict(variant=variant, n_train=N_TRAIN, grid_points=N_TEST, noise_sigma=NOISE_SIGMA,
                    epochs=epochs, z=Z95, cal_levels="|".join(str(v) for v in CALIBRATION_LEVELS),
                    torch_threads=threads, quick=args.quick, cache=not args.no_cache)
    for col, val in metadata.items():
        df[col] = val
    args.out.parent.mkdir(parents=True, exist_ok=True)
    append_rows(df, args.out)
    print("\nappended {} rows to {}".format(len(df), args.out))
    print("saved predictions to {}".format(PREDICTIONS_DIR))

    summary = summarise(df)
    print_summary(summary, len(seeds))
    print_sanity_check(summary)

    if not args.no_figures and figure_seed in seeds:
        written = figures_from_saved(figure_seed, args.quick, variant, layout="both", n_sample_lines=args.members)
        print("\nwrote {} figures to {}".format(len(written), FIGURE_DIR))

    print("\ntotal wall clock {:.1f}s".format(wall))


if __name__ == "__main__":
    main()
