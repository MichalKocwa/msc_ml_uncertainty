"""Presentation tables built from the results CSVs, for the notebook and the thesis.

**This module formats; it never computes a result.** Every number it returns is
read from `results/e1_toy_metrics.csv` or `results/e2_uci_metrics.csv` and
aggregated (mean, standard deviation) exactly as the experiment scripts already
report it. Nothing here fits a model, and nothing here can invent a value.

It exists so that `notebooks/wyniki.ipynb` contains no logic of its own:
`CLAUDE.md` puts code in `src/` and `experiments/`, so the notebook loads and
displays, and the code that decides what a table says lives here, under test.

**Reporting conventions carried over, not re-decided.**

- Spread is the **standard deviation** over repetitions (DEC-006 for E1's 20
  seeds, DEC-015 for E2's 20 splits). `SE = std / sqrt(n)`, and `n` is returned
  alongside every table so a caption can state it.
- Metrics are in **original target units** (DEC-003), already un-standardised by
  the experiment scripts.
- Rows that are not results are excluded by `reported_rows`: `--quick` runs
  (CLAUDE.md) and `--set` variant runs (DEC-014), plus any superseded earlier
  run of the same group.
"""
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from src.experiment_io import METHOD_ORDER
from src.metrics import METRIC_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent
E1_CSV = REPO_ROOT / "results" / "e1_toy_metrics.csv"
E2_CSV = REPO_ROOT / "results" / "e2_uci_metrics.csv"
E2_GP_KERNELS_CSV = REPO_ROOT / "results" / "e2_uci_gp_kernels.csv"
FIGURE_DIR = REPO_ROOT / "figures"

# Column headers as they should appear in a table. `sigma_fitted` keeps its
# code name deliberately: it is a diagnostic, not one of the four reported
# metrics, and E2 reads it only relatively (DEC-010's amendment).
METRIC_HEADERS: Dict[str, str] = {
    "rmse": "RMSE",
    "ll": "LL",
    "picp95": "PICP@95",
    "mpiw95": "MPIW@95",
    "sigma_fitted": "sigma_fitted",
    "cal_err": "cal_err",
    "cal_bias": "cal_bias",
}

# Decimal places per metric, matching what the experiment scripts print, so the
# notebook's tables and the scripts' stdout cannot disagree in the last digit.
METRIC_DIGITS: Dict[str, int] = {
    "rmse": 3, "ll": 3, "picp95": 3, "mpiw95": 3,
    "sigma_fitted": 4, "cal_err": 4, "cal_bias": 4,
}

E1_REGION_ORDER = ("in_range", "extrapolation", "overall")
E1_REGION_TITLES = {
    "in_range": "w zakresie treningowym [0, 6]",
    "extrapolation": "ekstrapolacja [-2, 0] ∪ [6, 8]",
    "overall": "cała siatka testowa [-2, 8]",
}
E2_DATASET_ORDER = ("yacht", "energy", "concrete")


def reported_rows(path: Path, group_cols: Sequence[str]) -> pd.DataFrame:
    """Baseline rows only, keeping the most recent run per group.

    Drops `--quick` rows and `--set` variant rows, then, because results are
    appended and never overwritten, keeps only the latest `run_id` within each
    group (a region for E1, a dataset for E2) so a superseded run cannot be
    averaged into a current one.
    """
    df = pd.read_csv(path)
    df = df[~df["quick"].astype(bool)]
    variant = df["variant"]
    df = df[variant.isna() | (variant.astype(str).str.strip() == "")]
    if df.empty:
        raise ValueError("{} holds no baseline (non-quick, non-variant) rows".format(path))
    group = list(group_cols) if group_cols else None
    if group:
        latest = df.groupby(group)["run_id"].transform("max")
        df = df[df["run_id"] == latest]
    else:
        df = df[df["run_id"] == df["run_id"].max()]
    return df.reset_index(drop=True)


def summarise(df: pd.DataFrame, by: Sequence[str],
              metrics: Sequence[str] = METRIC_COLUMNS) -> pd.DataFrame:
    """Mean and standard deviation of each metric, grouped by `by`."""
    agg = df.groupby(list(by))[list(metrics)].agg(["mean", "std"])
    agg.columns = ["{}_{}".format(metric, stat) for metric, stat in agg.columns]
    return agg.reset_index()


def format_summary(summary: pd.DataFrame, metrics: Sequence[str] = METRIC_COLUMNS,
                   methods: Sequence[str] = METHOD_ORDER) -> pd.DataFrame:
    """`mean +/- std` strings, one row per method in the display order.

    Methods absent from the summary are dropped rather than shown empty, so a
    partial run produces a shorter table instead of a table with holes.
    """
    present = [m for m in methods if m in set(summary["method"])]
    block = summary.set_index("method").reindex(present)
    out = {}
    for metric in metrics:
        digits = METRIC_DIGITS[metric]
        mean = block["{}_mean".format(metric)]
        std = block["{}_std".format(metric)]
        out[METRIC_HEADERS[metric]] = [
            "{:.{d}f} ± {}".format(m, "n/a" if pd.isna(s) else "{:.{d}f}".format(s, d=digits), d=digits)
            for m, s in zip(mean, std)
        ]
    table = pd.DataFrame(out, index=block.index)
    table.index.name = "metoda"
    return table


def stage_tables(df: pd.DataFrame, group_col: str, order: Sequence[str],
                 metrics: Sequence[str] = METRIC_COLUMNS) -> Dict[str, pd.DataFrame]:
    """One formatted table per group (E1 region, or E2 dataset), in `order`."""
    summary = summarise(df, ["method", group_col], metrics)
    tables = {}
    for key in order:
        block = summary[summary[group_col] == key]
        if not block.empty:
            tables[key] = format_summary(block, metrics)
    return tables


def repetitions(df: pd.DataFrame, group_col: str, key: str) -> int:
    """How many seeds (E1) or splits (E2) stand behind one group's table."""
    block = df[df[group_col] == key]
    column = "seed" if "seed" in block.columns else "split"
    return int(block[column].nunique())


def style(table: pd.DataFrame, caption: str = ""):
    """A pandas Styler for on-screen display: monospaced numbers, banded rows.

    Purely cosmetic. `to_latex` below, not this, is what produces something the
    thesis can use — a screenshot of a styled table is not a table.
    """
    styler = table.style.set_table_styles([
        {"selector": "th", "props": [("background-color", "#f2f2f2"),
                                     ("text-align", "left"),
                                     ("padding", "4px 10px"),
                                     ("border-bottom", "1px solid #999")]},
        {"selector": "td", "props": [("font-family", "DejaVu Sans Mono, monospace"),
                                     ("text-align", "right"),
                                     ("padding", "3px 10px")]},
        {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fafafa")]},
        {"selector": "caption", "props": [("caption-side", "top"),
                                          ("text-align", "left"),
                                          ("font-weight", "600"),
                                          ("padding-bottom", "6px")]},
    ])
    if caption:
        styler = styler.set_caption(caption)
    return styler


def to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    """A booktabs table, ready to paste into the thesis.

    Printed for the author to copy — `CLAUDE.md` forbids editing `.tex` files,
    and this does not: it produces text, it does not write anywhere.

    `±` becomes `$\\pm$` and `_` in a header becomes `\\_`, so the output
    compiles without further editing.
    """
    body = table.copy()
    body = body.map(lambda v: str(v).replace("±", "$\\pm$"))
    body.columns = [str(c).replace("_", r"\_") for c in body.columns]
    body.index = [str(i).replace("_", r"\_") for i in body.index]
    body.index.name = table.index.name

    column_format = "l" + "r" * body.shape[1]
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{%s}" % caption,
        r"  \label{%s}" % label,
        r"  \begin{tabular}{%s}" % column_format,
        r"    \toprule",
        "    " + " & ".join([body.index.name or ""] + list(body.columns)) + r" \\",
        r"    \midrule",
    ]
    for name, row in body.iterrows():
        lines.append("    " + " & ".join([name] + [str(v) for v in row.values]) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def e1_configuration(df: pd.DataFrame) -> Dict[str, object]:
    """The configuration E1's rows were produced under, read from the rows."""
    return dict(
        n_train=int(df["n_train"].iloc[0]),
        grid_points=int(df["grid_points"].iloc[0]),
        noise_sigma=float(df["noise_sigma"].iloc[0]),
        epochs=int(df["epochs"].iloc[0]),
        seeds=int(df["seed"].nunique()),
        torch_threads=int(df["torch_threads"].iloc[0]),
        run_id=str(df["run_id"].iloc[0]),
    )


def e2_configuration(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset shape and capacity, from the rows themselves (N-013, N-017)."""
    plain = df[df["method"].isin(["map", "mcd", "laplace"])]
    rows = []
    for name in E2_DATASET_ORDER:
        block = plain[plain["dataset"] == name]
        if block.empty:
            continue
        p = int(block["n_parameters"].max())
        n_train = int(block["n_train"].iloc[0])
        rows.append(dict(
            **{"zbiór": name}, N_train=n_train, N_test=int(block["n_test"].iloc[0]),
            d=int(block["d"].iloc[0]), P=p,
            **{"N_train/P": round(n_train / p, 2)},
            splity=int(block["split"].nunique()),
        ))
    return pd.DataFrame(rows).set_index("zbiór")


def e1_band_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """MPIW@95 outside the training range divided by MPIW@95 inside it.

    E1's structural check (N-007): `gp` and `laplace` should widen strongly
    outside the training data, `ensemble` a little, `bbb` barely, `map` not at
    all. Recomputed here from the CSV rather than quoted from the notes.
    """
    summary = summarise(df, ["method", "region"], ["mpiw95"])
    wide = summary.pivot(index="method", columns="region", values="mpiw95_mean")
    wide = wide.reindex([m for m in METHOD_ORDER if m in wide.index])
    out = pd.DataFrame({
        "MPIW@95 w zakresie": wide["in_range"].round(3),
        "MPIW@95 ekstrapolacja": wide["extrapolation"].round(3),
        "iloraz": (wide["extrapolation"] / wide["in_range"]).round(2),
    })
    out.index.name = "metoda"
    return out.sort_values("iloraz", ascending=False)


def e1_noise_check(df: pd.DataFrame, region: str = "in_range") -> pd.DataFrame:
    """Each method's fitted noise against E1's KNOWN true noise level.

    E1 is the only stage where this reads against ground truth: the toy
    problem's sigma is 0.1 by construction (DEC-005). DEC-010's amendment
    records that this basis does **not** transfer to E2, where no true noise
    level exists — hence the separate `mcd_noise_inflation` below, which is
    relative only.
    """
    block = df[df["region"] == region]
    true_sigma = float(block["noise_sigma"].iloc[0])
    summary = summarise(block, ["method"], ["sigma_fitted"])
    summary = summary.set_index("method").reindex(
        [m for m in METHOD_ORDER if m in set(summary["method"])])
    out = pd.DataFrame({
        "sigma_fitted": summary["sigma_fitted_mean"].round(4),
        "std": summary["sigma_fitted_std"].round(4),
        "sigma prawdziwa": true_sigma,
        "iloraz (sd)": (summary["sigma_fitted_mean"] / true_sigma).round(2),
        "iloraz (wariancja)": ((summary["sigma_fitted_mean"] / true_sigma) ** 2).round(2),
    })
    out.index.name = "metoda"
    return out


def mcd_noise_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """`mcd`'s fitted noise against the mean of the other five, per dataset.

    The E2 form of N-001. UCI has no known noise level (DEC-010's amendment), so
    the only honest reading is relative: does MC dropout inflate its aleatoric
    term compared with the other five methods **on the same data**?
    """
    others = [m for m in METHOD_ORDER if m != "mcd"]
    rows = []
    for name in E2_DATASET_ORDER:
        block = df[df["dataset"] == name]
        if block.empty:
            continue
        per_method = block.groupby("method")["sigma_fitted"].mean()
        if "mcd" not in per_method.index:
            continue
        rest = per_method.reindex([m for m in others if m in per_method.index]).mean()
        ratio = per_method["mcd"] / rest
        rows.append({
            "zbiór": name,
            "mcd sigma_fitted": round(float(per_method["mcd"]), 4),
            "pozostałe pięć (średnia)": round(float(rest), 4),
            "iloraz (sd)": round(float(ratio), 2),
            "iloraz (wariancja)": round(float(ratio ** 2), 1),
        })
    return pd.DataFrame(rows).set_index("zbiór")


def e2_reference_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Our `mcd` beside Gal & Ghahramani's published per-split numbers.

    **A sanity check, never a result of this repository** (N-022). Their tables
    report standard errors; this converts them to standard deviations
    (`std = SE * sqrt(n)`) so both columns are on DEC-015's convention, and the
    caller must say so wherever the table is shown.
    """
    from src.uci import load_reference

    rows = []
    for name in E2_DATASET_ORDER:
        ours = df[(df["dataset"] == name) & (df["method"] == "mcd")]
        if ours.empty:
            continue
        try:
            reference = load_reference(name)
        except FileNotFoundError:
            continue
        rows.append({
            "zbiór": name,
            "RMSE (nasze)": "{:.3f} ± {:.3f}".format(ours["rmse"].mean(), ours["rmse"].std()),
            "RMSE (literatura)": "{:.3f} ± {:.3f}".format(
                reference["rmse"].mean(), reference["rmse"].std(ddof=1)),
            "LL (nasze)": "{:+.3f} ± {:.3f}".format(ours["ll"].mean(), ours["ll"].std()),
            "LL (literatura)": "{:+.3f} ± {:.3f}".format(
                reference["ll"].mean(), reference["ll"].std(ddof=1)),
        })
    return pd.DataFrame(rows).set_index("zbiór")


def gp_kernel_summary(path: Path = E2_GP_KERNELS_CSV) -> pd.DataFrame:
    """Fitted GP kernel hyperparameters and how close they sit to their bounds.

    The evidence behind N-021: a value on its bound is set by the bound, not by
    the data, so the affected `gp` row is not quotable without that entry.
    """
    gp = pd.read_csv(path)
    summary = gp.groupby(["dataset", "hyperparameter"]).agg(
        mediana=("value", "median"),
        **{"na granicy": ("on_bound", "sum")},
        **{"min. margines (log)": ("log_margin_to_nearest_bound", "min")},
        **{"ostrzeżenia": ("convergence_warnings", "sum")},
        splity=("value", "size"),
    )
    order = [d for d in E2_DATASET_ORDER if d in summary.index.get_level_values(0)]
    return summary.reindex(order, level=0).round(4)
