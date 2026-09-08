"""E1 plumbing: the figure data and the metric data must come from one array.

DEC-007's whole point is that the band drawn in `figures/e1_toy_*.png` and the
`mpiw95` written to `results/e1_toy_metrics.csv` describe the same numbers. The
script achieves that by slicing both out of a single `Prediction`; this test
re-derives the metrics from the returned panel arrays and asserts they match
the rows, so a future refactor cannot reintroduce a second `predict()` call
without failing here.

Runs at `--quick` epochs on two cheap methods - `map` (zero epistemic term, the
degenerate case) and `mcd` (the batch-dependent one). No number here is a
result.
"""
import numpy as np
import pytest

from experiments.e1_toy import (
    QUICK_EPOCHS, build_method, parse_overrides, run_seed, summarise, variant_label,
)
from src.data import EXTRAPOLATION, IN_RANGE, N_TEST, OVERALL, REGIONS
from src.metrics import compute_metrics


@pytest.fixture(scope="module")
def run():
    return run_seed(seed=0, method_names=["map", "mcd"], quick=True, use_cache=False)


def test_one_row_per_method_and_region(run):
    rows, _, _ = run
    assert len(rows) == 2 * len(REGIONS)
    assert {r["region"] for r in rows} == set(REGIONS)
    assert {r["method"] for r in rows} == {"map", "mcd"}
    assert all(r["seed"] == 0 for r in rows)


def test_region_row_counts_partition_the_grid(run):
    rows, _, _ = run
    per_region = {r["region"]: r["n_points"] for r in rows if r["method"] == "map"}
    assert per_region[OVERALL] == N_TEST
    assert per_region[IN_RANGE] + per_region[EXTRAPOLATION] == N_TEST


def test_metrics_are_reproducible_from_the_panel_arrays(run):
    """The DEC-007 invariant: figure data and table data are the same numbers."""
    rows, panels, data = run
    for row in rows:
        p = panels[row["method"]]
        mask = data.regions[row["region"]]
        expected = compute_metrics(
            y=data.y_test[mask], mean=p["mean"][mask],
            var_aleatoric=p["var_aleatoric"][mask], var_epistemic=p["var_epistemic"][mask],
        )
        for metric, value in expected.items():
            assert row[metric] == pytest.approx(value), (row["method"], row["region"], metric)


def test_map_reports_an_exactly_zero_epistemic_term(run):
    """`map` is the lower reference point: no epistemic term at all, stated as
    zeros rather than folded into the aleatoric one (`base.py`'s protocol).
    """
    _, panels, _ = run
    assert np.all(panels["map"]["var_epistemic"] == 0.0)
    assert np.all(panels["map"]["var_aleatoric"] > 0.0)


def test_predictions_are_in_original_y_units(run):
    """Un-standardised before scoring (DEC-003): the predictive mean must track
    sin(x) inside the training range, not the standardised target.
    """
    _, panels, data = run
    in_range = data.regions[IN_RANGE]
    assert np.abs(panels["map"]["mean"][in_range]).max() < 3.0
    assert panels["map"]["mean"][in_range].std() > 0.1


def test_summarise_produces_mean_and_std_columns(run):
    import pandas as pd
    rows, _, _ = run
    summary = summarise(pd.DataFrame(rows))
    assert len(summary) == 2 * len(REGIONS)
    for column in ("rmse_mean", "rmse_std", "ll_mean", "picp95_mean",
                   "mpiw95_mean", "sigma_fitted_mean"):
        assert column in summary.columns
    # one seed only, so every std is undefined rather than zero
    assert summary["rmse_std"].isna().all()


# --- sensitivity-sweep plumbing (DEC-014) ------------------------------------

def test_parse_overrides_types_values_by_literal():
    out = parse_overrides(["mcd:dropout_p=0.2", "ensemble:M=10",
                           "mcd:input_dropout=True", "map:activation=relu"])
    assert out == {
        "mcd": {"dropout_p": 0.2, "input_dropout": True},
        "ensemble": {"M": 10},
        "map": {"activation": "relu"},   # not a literal, kept as a string
    }
    assert isinstance(out["ensemble"]["M"], int)
    assert isinstance(out["mcd"]["dropout_p"], float)


def test_parse_overrides_rejects_malformed_input():
    for bad in ["dropout_p=0.2", "mcd:dropout_p", "nosuchmethod:x=1"]:
        with pytest.raises(SystemExit):
            parse_overrides([bad])


def test_no_overrides_is_an_empty_variant_label():
    """A baseline run must leave `variant` empty, or every row would look swept."""
    assert variant_label(parse_overrides([])) == ""
    assert variant_label(parse_overrides(None)) == ""


def test_variant_label_is_order_independent():
    a = variant_label(parse_overrides(["ensemble:M=10", "mcd:dropout_p=0.2"]))
    b = variant_label(parse_overrides(["mcd:dropout_p=0.2", "ensemble:M=10"]))
    assert a == b != ""


def test_overrides_reach_the_constructed_method_and_defaults_are_untouched():
    swept = build_method("mcd", quick=False, overrides={"mcd": {"dropout_p": 0.25, "T": 7}})
    assert (swept.dropout_p, swept.T) == (0.25, 7)
    # a method not named by --set keeps every repo default
    assert build_method("mcd", quick=False, overrides={"ensemble": {"M": 9}}).dropout_p == 0.1
    assert build_method("mcd", quick=False).dropout_p == 0.1


def test_overrides_win_over_quick_defaults():
    m = build_method("mcd", quick=True, overrides={"mcd": {"T": 3}})
    assert m.T == 3
    assert m.epochs == QUICK_EPOCHS      # untouched quick override still applies
