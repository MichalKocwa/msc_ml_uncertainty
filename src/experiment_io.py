"""CLI and results-file helpers shared by every experiment script.

These three functions were written for `experiments/e1_toy.py` and moved here
when `experiments/e2_uci.py` needed them (2026-09-05). They are deliberately
library code rather than something E2 imports from E1: `CLAUDE.md` fixes the
layout as "CLI scripts in `experiments/`, one per experiment; library code in
`src/`", an experiment-to-experiment import would execute E1's module-level
code (its `sys.path` insert and its matplotlib import) inside a figure-free
experiment, and it would let an edit to E1's script break E2 silently.

Nothing here touches a model, a metric or a cache key, so the move changes no
number in either stage's results.
"""
import ast
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.methods import METHODS

# Display order, shared by E1's combined panel and by any table in either stage
# that groups methods: deterministic baseline first, then the sampling
# approximations, then the analytic ones. Not an ordering by quality.
#
# It lives here rather than in `src/style.py` (where it was defined until
# 2026-09-05) so that E2, which produces no figures, can order its tables
# without importing matplotlib. `style.py` re-exports it, so nothing that read
# it from there had to change.
METHOD_ORDER: Sequence[str] = ("map", "mcd", "ensemble", "bbb", "laplace", "gp")


def parse_overrides(assignments):
    """`["mcd:dropout_p=0.2", "ensemble:M=10"]` -> `{"mcd": {"dropout_p": 0.2}, ...}`.

    Values are parsed with `ast.literal_eval`, so `0.2`, `10`, `True`, `None` and
    `"relu"` all arrive as the right Python type; anything that is not a literal
    is kept as a string, which is what makes `--set mcd:activation=relu` work
    without quoting.

    **This is for sensitivity analysis, not for tuning** — see DEC-014. A run
    that sets anything here is a variant run: it is stamped into the `variant`
    column so its rows can never be mistaken for the shared-defaults baseline,
    and it is not a result of E1 or E2.
    """
    overrides = {}
    for item in assignments or []:
        if ":" not in item or "=" not in item.split(":", 1)[1]:
            raise SystemExit(
                "ERROR: --set expects method:param=value, got {!r}".format(item))
        method, rest = item.split(":", 1)
        param, raw = rest.split("=", 1)
        if method not in METHODS:
            raise SystemExit("ERROR: --set names unknown method {!r}".format(method))
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        overrides.setdefault(method, {})[param] = value
    return overrides


def variant_label(overrides):
    """Stable, sorted description of the overrides, empty for a baseline run.

    Goes into the CSV's `variant` column. Sorted so that the same set of
    overrides always produces the same label regardless of the order they were
    given on the command line — otherwise two identical variants would look like
    two different ones when grouping the results.
    """
    return ";".join(
        "{}:{}={!r}".format(m, p, v)
        for m in sorted(overrides)
        for p, v in sorted(overrides[m].items())
    )


def append_rows(df: pd.DataFrame, path: Path) -> None:
    """Append to a results CSV, refusing to do so if the columns differ.

    `results/*.csv` is appended, never overwritten (CLAUDE.md). `to_csv(mode="a",
    header=False)` writes positionally, so appending a frame whose columns have
    changed — a metric added, a configuration field recorded that was not
    recorded before — would silently misalign every value in every new row
    against the old header. Nothing would raise; the file would simply be wrong,
    and a metric would be read out of a configuration column.

    Refusing is the right response rather than reconciling automatically: a
    changed column set means the configuration changed, so the old rows are from
    a different experiment and the two should not share a file in the first
    place. Archive the old file under a name that says what it was, and start a
    new one.
    """
    if path.exists():
        existing = list(pd.read_csv(path, nrows=0).columns)
        if existing != list(df.columns):
            missing = [c for c in df.columns if c not in existing]
            extra = [c for c in existing if c not in df.columns]
            raise SystemExit(
                "ERROR: refusing to append to {}\n"
                "  its columns differ from this run's, so appending would misalign every row.\n"
                "  new columns: {}\n  columns only in the file: {}\n"
                "  Archive the existing file under a name recording its configuration "
                "(e.g. results/<experiment>_<what-it-was>.csv) and re-run.".format(
                    path, missing or "none", extra or "none")
            )
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def save_predictions(path: Path, arrays: dict, panels: dict, meta: dict = None) -> None:
    """Write one seed's full predictive arrays to a compressed `.npz`.

    `arrays` holds the data (`x_test`, `y_test`, `x_train`, `y_train`, ...) and
    `panels[method]` the un-standardised `mean`, `var_aleatoric`,
    `var_epistemic` and, where the method has them, `samples`. `meta` values
    are stored as 0-d arrays (strings and numbers both round-trip).

    Exists so that a figure is a function of a SAVED prediction, not of a
    re-run: any layout (one panel per method, a combined grid, with or without
    individual-run lines) can be redrawn later from the same numbers the
    metrics were computed from (DEC-007), without touching a model or the
    cache. Keys are `<method>__<field>`; the method list is stored under
    `methods` so a reader does not have to parse key names.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: np.asarray(v) for k, v in arrays.items()}
    out["methods"] = np.array(list(panels))
    for method, p in panels.items():
        for field, value in p.items():
            if value is not None:
                out["{}__{}".format(method, field)] = np.asarray(value)
    for k, v in (meta or {}).items():
        out["meta__{}".format(k)] = np.asarray(v)
    np.savez_compressed(path, **out)


def load_predictions(path: Path):
    """Inverse of `save_predictions`: `(arrays, panels, meta)`."""
    with np.load(Path(path), allow_pickle=False) as f:
        methods = [str(m) for m in f["methods"]]
        arrays, panels, meta = {}, {m: {} for m in methods}, {}
        for key in f.files:
            if key == "methods":
                continue
            if key.startswith("meta__"):
                value = f[key]
                meta[key[len("meta__"):]] = value.item() if value.ndim == 0 else value
            elif "__" in key:
                method, field = key.split("__", 1)
                panels[method][field] = f[key]
            else:
                arrays[key] = f[key]
    return arrays, panels, meta
