"""E2 - UCI benchmark on the Hernandez-Lobato & Adams splits: six methods, metrics only.

    .venv/Scripts/python.exe experiments/e2_uci.py                    # the real run (~3.6 h)
    .venv/Scripts/python.exe experiments/e2_uci.py --quick            # plumbing check only
    .venv/Scripts/python.exe experiments/e2_uci.py --datasets yacht   # one dataset end to end

**No figures.** E2 reports numbers only (`CLAUDE.md`'s stage table); the
predictive-band figures belong to E1, where the problem is 1D and a band can be
drawn.

**No region column.** E1's three regions exist because the toy problem has a
defined training domain and everything outside it is extrapolation. A real UCI
test split is drawn from the same distribution as its training split, so there
is no extrapolation region to separate: one row per (method, dataset, split).

Configuration is inherited from E1 and not re-opened here: DEC-003
(standardise on training statistics, un-standardise before scoring), DEC-006
(mean +/- std over 20 repetitions), DEC-008 (z = 1.959964), DEC-009 (one
moment-matched Gaussian per method), DEC-010 as amended (`sigma_fitted` is a
*relative* diagnostic here - UCI has no known noise level to read it against),
DEC-012 (4000 epochs), DEC-013 (the calibration columns) and DEC-014 (`--set`
marks a run as a VARIANT). The splits themselves are the literature's, read
from fixed index files by `src/uci.py`.

**Seeding differs from E1, and it matters for reading the spread (DEC-017).**
Here the split is fixed by the index files, so `seed = split` controls only
model initialisation and batch order - not the data. E2's `+/- std` therefore
mixes split-to-split data variation with initialisation variation, locked
together. That is the reference protocol's own arrangement (one training run
per split), but it must be stated rather than assumed.

**One fit, one predict, per (dataset, split, method)** - DEC-007 / N-006. MC
dropout's predictions depend on the size of the evaluation batch, so predicting
the test split in pieces would give numbers that depend on how it was cut up.
"""
import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# `python experiments/e2_uci.py` puts `experiments/` on sys.path, not the
# repository root, so the `src.*` imports below would fail on the documented
# invocation (README). Prepending the root here keeps that command working
# without requiring `-m experiments.e2_uci` or an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from src.experiment_io import (  # noqa: E402
    METHOD_ORDER, append_rows, parse_overrides, variant_label,
)
from src.methods import METHODS  # noqa: E402
from src.methods.backbone import DEFAULT_EPOCHS  # noqa: E402
from src.metrics import CALIBRATION_LEVELS, METRIC_COLUMNS, Z95, compute_metrics  # noqa: E402
from src.seeding import DEFAULT_TORCH_THREADS, TORCH_THREADS_ENV  # noqa: E402
from src.uci import UCI_DATASETS, load_dataset, load_reference, load_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "e2_uci_metrics.csv"

# GP kernel hyperparameters go to their own file, not into the metrics CSV.
# N-012 requires checking that no kernel hyperparameter has been fixed by its
# bound rather than by the data before any `gp` number is quoted; that check
# needs one row per hyperparameter, which does not fit the one-row-per-(method,
# dataset, split) shape of the results file. Keeping it separate also keeps the
# reported schema stable.
GP_KERNEL_CSV = REPO_ROOT / "results" / "e2_uci_gp_kernels.csv"

QUICK_EPOCHS = 30

# `--quick` overrides, matching E1's. Deliberately crude: the point is to
# execute every code path in a couple of minutes, not to approximate the real
# run. Anything produced under these is unreportable, by CLAUDE.md's rule.
QUICK_OVERRIDES = {
    "map": dict(epochs=QUICK_EPOCHS),
    "mcd": dict(epochs=QUICK_EPOCHS, T=10),
    "ensemble": dict(epochs=QUICK_EPOCHS, M=2),
    "bbb": dict(epochs=QUICK_EPOCHS, T=10),
    "laplace": dict(epochs=QUICK_EPOCHS),
    "gp": dict(n_restarts_optimizer=0),
}

# A log-scale hyperparameter is treated as "on its bound" when it sits within
# this distance of it in log space, i.e. within ~0.5% of the bound's value.
# ARBITRARY choice (assistant): nothing measured or cited fixes it. It only
# decides when the diagnostic file's `on_bound` flag fires - the fitted value,
# its bounds and the margin are all written out, so a reader can apply any
# threshold. Fitted values sitting at a bound are the condition N-012 warns
# about: the value is then set by the bound, not by the data.
BOUND_TOLERANCE = 5e-3


def build_method(name, quick, overrides=None):
    """Construct a method at REPO DEFAULTS, unless `--set` overrides something.

    With no `--set`, no hyperparameter is chosen here — every default lives with
    its own method in `src/methods/`, where its basis (or its explicit lack of
    one, e.g. `T = 100`) is documented next to it. That is what makes the
    baseline run's configuration readable from the methods alone.

    Kept per script rather than shared with E1: `QUICK_OVERRIDES` is a property
    of what a given experiment's plumbing check has to exercise, not library
    behaviour.
    """
    kwargs = dict(QUICK_OVERRIDES[name]) if quick else {}
    kwargs.update((overrides or {}).get(name, {}))
    return METHODS[name](**kwargs)


def gp_kernel_rows(method, dataset_name, split, fitted_here, warning_count):
    """One row per fitted kernel hyperparameter, for the N-012 bound check.

    `kernel_.theta` and `kernel_.bounds` are in log space, which is where
    sklearn's optimiser actually works and therefore where "on the bound" is
    the meaningful comparison; the natural-scale value is written alongside so
    the file is readable without exponentiating by hand.

    `fitted_here` records whether this call actually ran L-BFGS: on a cache hit
    the fitted kernel is real but no `ConvergenceWarning` could have been
    observed, so a zero warning count would otherwise read as evidence of
    convergence when it is only evidence of a cache hit.
    """
    kernel = method.gp.kernel_
    theta, bounds = kernel.theta, kernel.bounds
    names = [h.name for h in kernel.hyperparameters for _ in range(h.n_elements)]
    rows = []
    for i, (name, value) in enumerate(zip(names, theta)):
        lower, upper = float(bounds[i][0]), float(bounds[i][1])
        margin = float(min(value - lower, upper - value))
        rows.append(dict(
            dataset=dataset_name, split=split, hyperparameter=name,
            value=float(np.exp(value)),
            lower_bound=float(np.exp(lower)), upper_bound=float(np.exp(upper)),
            log_value=float(value), log_lower=lower, log_upper=upper,
            log_margin_to_nearest_bound=margin,
            on_bound=bool(margin <= BOUND_TOLERANCE),
            log_marginal_likelihood=float(method.gp.log_marginal_likelihood_value_),
            fitted_here=fitted_here, convergence_warnings=int(warning_count),
        ))
    return rows


def run_split(dataset, split, method_names, quick, use_cache, overrides=None):
    """Fit and predict every method on one split.

    Returns `(rows, gp_rows, predictions)`. `predictions` maps a method name to
    its un-standardised predictive arrays — the same arrays the metrics were
    computed from, returned so a test can re-derive the rows from them
    (`tests/test_e2_pipeline.py`).

    `seed = split` (DEC-017): one training run per split, matching the reference
    protocol, and the row is reproducible from its own `seed` column.
    """
    data = load_split(dataset, split)
    seed = split
    rows, gp_rows, predictions = [], [], {}

    for name in method_names:
        method = build_method(name, quick, overrides)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            t0 = time.perf_counter()
            method.fit(data.x_train_std, data.y_train_std, seed=seed, use_cache=use_cache)
            fit_seconds = time.perf_counter() - t0
        convergence_warnings = sum(
            issubclass(w.category, ConvergenceWarning) for w in caught)
        for w in caught:
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

        # SINGLE predict call on the whole test split — see the module docstring.
        pred = method.predict(data.x_test_std)

        s = data.standardiser
        mean = s.unstandardise_mean(pred.mean)
        var_aleatoric = s.unstandardise_var(pred.var_aleatoric)
        var_epistemic = s.unstandardise_var(pred.var_epistemic)
        predictions[name] = dict(
            mean=mean, var_aleatoric=var_aleatoric, var_epistemic=var_epistemic)

        values = compute_metrics(
            y=data.y_test, mean=mean,
            var_aleatoric=var_aleatoric, var_epistemic=var_epistemic,
        )
        rows.append(dict(
            method=name, dataset=dataset.name, split=split, seed=seed,
            n_train=data.n_train, n_test=data.n_test, d=data.d,
            **values,
            n_parameters=int(method.n_parameters), fit_seconds=round(fit_seconds, 3),
        ))

        if name == "gp":
            # `fit_seconds` under a millisecond means the fitted object came
            # back from the cache rather than from L-BFGS.
            gp_rows += gp_kernel_rows(
                method, dataset.name, split,
                fitted_here=fit_seconds > 1e-3, warning_count=convergence_warnings)

        print("  {:<9} split {:<2} {:<9} fit {:6.1f}s  RMSE {:8.4f}  LL {:+8.3f}  "
              "PICP {:.3f}  sigma_fitted {:.4f}".format(
                  dataset.name, split, name, fit_seconds, values["rmse"], values["ll"],
                  values["picp95"], values["sigma_fitted"]))

    return rows, gp_rows, predictions


def summarise(df):
    """Mean +/- std over splits, per method and dataset (DEC-006).

    No cross-dataset aggregate (DEC-016): the three targets are in different
    units — concrete in MPa, energy a heating load, yacht a residuary
    resistance — so a mean of RMSE across them is not a quantity.
    """
    agg = df.groupby(["method", "dataset"])[list(METRIC_COLUMNS)].agg(["mean", "std"])
    agg.columns = ["{}_{}".format(m, s) for m, s in agg.columns]
    return agg.reset_index()


def print_summary(summary, df, datasets):
    n_splits = df.groupby("dataset")["split"].nunique().to_dict()
    for name in datasets:
        block = summary[summary.dataset == name]
        if block.empty:
            continue
        present = [m for m in METHOD_ORDER if m in set(block.method)]
        block = block.set_index("method").reindex(present)
        n = n_splits.get(name, 0)
        rows = df[df.dataset == name]
        print("\n### {}  (N_train {}, N_test {}, d {}; mean +/- std over {} splits, "
              "original target units)\n".format(
                  name, int(rows.n_train.iloc[0]), int(rows.n_test.iloc[0]),
                  int(rows.d.iloc[0]), n))
        print("| method | RMSE | LL | PICP@95 | MPIW@95 | sigma_fitted | cal_err | cal_bias |")
        print("|---|---|---|---|---|---|---|---|")
        for method, r in block.iterrows():
            def cell(metric, digits=3):
                sd = r["{}_std".format(metric)]
                sd_txt = "n/a" if pd.isna(sd) else "{:.{d}f}".format(sd, d=digits)
                return "{:.{d}f} +/- {}".format(r["{}_mean".format(metric)], sd_txt, d=digits)
            print("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                method, cell("rmse"), cell("ll"), cell("picp95"), cell("mpiw95"),
                cell("sigma_fitted", 4), cell("cal_err", 4), cell("cal_bias", 4)))
        # DEC-006/DEC-015: the spread is a standard DEVIATION. The published
        # tables this chapter sits next to report standard ERRORS, and the two
        # differ by exactly sqrt(n) — stated here so the table carries its own
        # conversion rather than relying on the reader to know the convention.
        print("\nSpread is the standard deviation over n = {} splits; "
              "SE = std / sqrt({}) = std / {:.3f}.".format(n, n, np.sqrt(n) if n else float("nan")))
        print("MPIW@95 is reported only alongside PICP@95 (CLAUDE.md): a width "
              "alone does not say whether the interval is calibrated.")


def print_reference_check(df, datasets):
    """Compare our `mcd` against Gal & Ghahramani's published per-split numbers.

    **A sanity check, not a result.** Four differences remain between the two
    columns and none of them is being closed here: `dropout_p` fixed at 0.1
    against their tuned `(dropout_rate, tau)` grid (N-015 records that they use
    0.05 and 0.005 on these datasets at this network size); a `gamma`-derived
    prior penalty against their `tau`-derived L2; DEC-009's moment-matched
    Gaussian against their log-sum-exp mixture log-likelihood; and their MC
    dropout RMSE averages the stochastic passes. **Same order of magnitude is
    the pass criterion; equality is not.**

    Their tables report mean +/- standard ERROR, ours mean +/- std (DEC-006), so
    the reference column below is CONVERTED to a standard deviation
    (`SE * sqrt(n)`) — done explicitly, and stated, rather than letting two
    conventions sit side by side unlabelled.
    """
    if "mcd" not in set(df.method):
        return
    print("\n### Sanity check - our `mcd` against the published per-split results\n")
    print("Reference: `data/uci_splits/<name>/results/test_MC_rmse_*` and `test_ll_*` "
          "(yaringal/DropoutUncertaintyExps, 1 hidden layer, 40x100 epochs, 20 splits).")
    print("Both columns are mean +/- STD here; the published tables state standard "
          "errors, so their spread has been converted (std = SE * sqrt(n)). Not a "
          "result of this repository, and not directly comparable - see the four "
          "differences in DEC-012, DEC-014 and N-015.\n")
    print("| dataset | our RMSE | reference RMSE | our LL | reference LL |")
    print("|---|---|---|---|---|")
    for name in datasets:
        ours = df[(df.dataset == name) & (df.method == "mcd")]
        if ours.empty:
            continue
        try:
            ref = load_reference(name)
        except FileNotFoundError:
            continue
        print("| {} | {:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} | {:+.3f} +/- {:.3f} | "
              "{:+.3f} +/- {:.3f} |".format(
                  name, ours.rmse.mean(), ours.rmse.std(),
                  ref["rmse"].mean(), ref["rmse"].std(ddof=1),
                  ours.ll.mean(), ours.ll.std(),
                  ref["ll"].mean(), ref["ll"].std(ddof=1)))


# Methods whose `n_parameters` is one plain backbone, so the N-013 ratio means
# what it says. `ensemble` reports M networks and `bbb` reports two variational
# parameters per weight, so neither is a parameter count of the shared
# architecture and neither belongs in that table.
PLAIN_BACKBONE_METHODS = ("map", "mcd", "laplace")


def print_capacity_note(df):
    """N_train / P per dataset - the N-013 regime marker, from measured rows.

    `laplace`'s GGN has rank at most `N_train`, so wherever this ratio is below
    1 some parameter directions are determined by the prior alone rather than by
    the data. Printed from the run's own `n_parameters` column so it cannot
    drift from what was actually fitted.

    Note the counting convention: `count_parameters` includes the single global
    `log_sigma2`, so P here is `50*d + 102`, one more than the `50*d + 101` in
    N-013, which counted `.mlp` alone. The ratio is unaffected at two decimals.
    """
    block_all = df[df.method.isin(PLAIN_BACKBONE_METHODS)]
    if block_all.empty:
        return
    print("\n### Capacity regime (N-013)\n")
    print("| dataset | N_train | d | P (one backbone, incl. log_sigma2) | N_train / P |")
    print("|---|---|---|---|---|")
    for name, block in block_all.groupby("dataset", sort=False):
        counts = set(block.n_parameters.astype(int))
        if len(counts) != 1:
            # map, mcd and laplace share one architecture; if they ever report
            # different counts the table would be quietly averaging two things.
            print("| {} | (inconsistent parameter counts across methods: {}) |".format(
                name, sorted(counts)))
            continue
        p = counts.pop()
        n_train = int(block.n_train.iloc[0])
        print("| {} | {} | {} | {} | {:.2f} |".format(
            name, n_train, int(block.d.iloc[0]), p, n_train / p))


def load_reported_rows(path):
    """The baseline rows of the most recent run per dataset, from an existing CSV.

    Exists so the Markdown table can be regenerated from
    `results/e2_uci_metrics.csv` without re-running ~3.6 h of fits, and so that
    a table covering all three datasets can be printed after they were run in
    separate invocations.

    Filters out anything that is not a reported result: `--quick` rows (not a
    result, by CLAUDE.md) and `variant` rows (a `--set` sensitivity run,
    DEC-014). Where a dataset has been run more than once, the latest `run_id`
    wins — earlier rows stay in the file, since results are appended and never
    overwritten, but a superseded run must not be averaged into the current one.
    """
    df = pd.read_csv(path)
    df = df[~df.quick.astype(bool)]
    df = df[df.variant.isna() | (df.variant.astype(str).str.strip() == "")]
    if df.empty:
        raise SystemExit("ERROR: {} holds no baseline (non-quick, non-variant) rows".format(path))
    latest = df.groupby("dataset")["run_id"].max()
    keep = df.apply(lambda r: r["run_id"] == latest[r["dataset"]], axis=1)
    dropped = len(df) - int(keep.sum())
    if dropped:
        print("note: ignored {} row(s) from superseded runs; using the latest run_id "
              "per dataset ({}).".format(
                  dropped, ", ".join("{}={}".format(k, v) for k, v in latest.items())))
    return df[keep]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(UCI_DATASETS),
                        choices=list(UCI_DATASETS))
    parser.add_argument("--splits", type=int, nargs="+", default=None,
                        help="split indices to run; default is every split in n_splits.txt (20)")
    parser.add_argument("--methods", nargs="+", default=list(METHOD_ORDER), choices=list(METHODS))
    parser.add_argument("--quick", action="store_true",
                        help="plumbing check only - one split, cut epochs; the output is NOT a result")
    parser.add_argument("--no-cache", action="store_true",
                        help="force a real retrain (src/methods/cache.py); required after "
                             "editing anything in src/methods/, which the cache key does not cover")
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="METHOD:PARAM=VALUE",
        help="override one method hyperparameter, repeatable (e.g. --set mcd:dropout_p=0.05). "
             "Marks the run as a VARIANT (see the `variant` column); a variant run is a "
             "sensitivity check (DEC-014), not a result of E2.")
    parser.add_argument("--out", type=Path, default=RESULTS_CSV)
    parser.add_argument("--gp-out", type=Path, default=GP_KERNEL_CSV)
    parser.add_argument("--report-only", action="store_true",
                        help="print the tables from an existing --out CSV without fitting anything")
    args = parser.parse_args()

    if args.report_only:
        df = load_reported_rows(args.out)
        datasets = [d for d in UCI_DATASETS if d in set(df.dataset)]
        print("E2 UCI benchmark | report from {} | datasets {}".format(args.out, datasets))
        print("epochs {}, z {:.6f}, torch threads {}".format(
            sorted(df.epochs.unique()), Z95, sorted(df.torch_threads.unique())))
        print_capacity_note(df)
        print_summary(summarise(df), df, datasets)
        print_reference_check(df, datasets)
        return

    overrides = parse_overrides(args.overrides)
    variant = variant_label(overrides)
    threads = int(os.environ.get(TORCH_THREADS_ENV, DEFAULT_TORCH_THREADS))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    epochs = QUICK_EPOCHS if args.quick else DEFAULT_EPOCHS

    print("E2 UCI benchmark | run {} | datasets {} | methods {}".format(
        run_id, args.datasets, args.methods))
    print("epochs {}, z {:.6f}, torch threads {}, cache {}, seed = split index (DEC-017)".format(
        epochs, Z95, threads, "off" if args.no_cache else "on"))
    if args.quick:
        print("*** --quick: one split, reduced epochs. THIS OUTPUT IS NOT A RESULT. ***")
    if variant:
        print("*** VARIANT RUN: {} ***".format(variant))
        print("    A variant run is a sensitivity check (DEC-014), not a result of E2.")

    metadata = dict(variant=variant, epochs=epochs, z=Z95,
                    cal_levels="|".join(str(v) for v in CALIBRATION_LEVELS),
                    torch_threads=threads, quick=args.quick, cache=not args.no_cache)

    def stamp(rows):
        frame = pd.DataFrame(rows)
        frame.insert(0, "run_id", run_id)
        for col, val in metadata.items():
            frame[col] = val
        return frame

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows, all_gp_rows = [], []
    t0 = time.perf_counter()
    for name in args.datasets:
        dataset = load_dataset(name)
        splits = args.splits if args.splits is not None else list(range(dataset.n_splits))
        if args.quick:
            splits = splits[:1]
        print("\n{}: N {}, d {}, {} splits -> running {}".format(
            name, dataset.n, dataset.d, dataset.n_splits, splits))
        dataset_rows, dataset_gp_rows = [], []
        for split in splits:
            rows, gp_rows, _ = run_split(
                dataset, split, args.methods, args.quick,
                use_cache=not args.no_cache, overrides=overrides)
            dataset_rows += rows
            dataset_gp_rows += gp_rows

        # Written per dataset rather than once at the end: the full run is
        # ~3.6 h, and a failure in the last dataset must not discard the hours
        # already computed. `append_rows` still guards the schema on every call.
        append_rows(stamp(dataset_rows), args.out)
        print("  -> appended {} rows for {} to {}".format(
            len(dataset_rows), name, args.out))
        all_rows += dataset_rows
        all_gp_rows += dataset_gp_rows
    wall = time.perf_counter() - t0

    df = pd.DataFrame(all_rows)
    df.insert(0, "run_id", run_id)
    for col, val in metadata.items():
        df[col] = val
    print("\nwrote {} rows in total to {}".format(len(df), args.out))

    if all_gp_rows:
        gp_df = pd.DataFrame(all_gp_rows)
        gp_df.insert(0, "run_id", run_id)
        gp_df["variant"] = variant
        gp_df["quick"] = args.quick
        append_rows(gp_df, args.gp_out)
        on_bound = int(gp_df.on_bound.sum())
        print("appended {} GP kernel rows to {} ({} on a bound)".format(
            len(gp_df), args.gp_out, on_bound))
        if on_bound:
            print("*** WARNING: a GP kernel hyperparameter is at its bound - the fitted "
                  "value is set by the bound, not by the data (N-012). Do not quote the "
                  "affected `gp` rows without addressing this. ***")
            for _, r in gp_df[gp_df.on_bound].iterrows():
                print("    {} split {}: {} = {:.4g} (bounds {:.4g}..{:.4g})".format(
                    r.dataset, r.split, r.hyperparameter, r.value,
                    r.lower_bound, r.upper_bound))
        stale = gp_df[(~gp_df.fitted_here) & (gp_df.convergence_warnings == 0)]
        if len(stale):
            print("note: {} of these rows came from the fit cache, so no convergence "
                  "warning could have been observed for them; re-run with --no-cache to "
                  "check convergence itself.".format(len(stale)))

    print_capacity_note(df)
    print_summary(summarise(df), df, args.datasets)
    print_reference_check(df, args.datasets)
    print("\ntotal wall clock {:.1f}s".format(wall))


if __name__ == "__main__":
    main()
