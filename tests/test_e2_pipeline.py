"""E2 plumbing: the rows written to the CSV are the metrics of the predictions made.

The E1 analogue (`tests/test_e1_pipeline.py`) exists because the figure and the
table had to describe the same numbers. E2 draws no figures, so the invariant
here is narrower but the same in kind: `run_split` must write rows that can be
re-derived from the very arrays it predicted, in original target units, one row
per (method, dataset, split) and **no region column** — E1's regions have no
counterpart on a real test split.

Runs at `--quick` epochs on `yacht` (the smallest dataset) with two cheap
methods: `map`, whose epistemic term is identically zero, and `mcd`, whose
predictions depend on the evaluation batch. No number produced here is a result.
"""
import numpy as np
import pytest

from experiments.e2_uci import BOUND_TOLERANCE, QUICK_EPOCHS, build_method, run_split, summarise
from src.metrics import METRIC_COLUMNS, compute_metrics
from src.uci import UCI_ROOT, load_dataset

pytestmark = pytest.mark.skipif(
    not UCI_ROOT.is_dir(),
    reason="data/uci_splits/ absent; run scripts/fetch_data.py")

METHODS_UNDER_TEST = ["map", "mcd"]
SPLIT = 0


@pytest.fixture(scope="module")
def run():
    dataset = load_dataset("yacht")
    rows, gp_rows, predictions = run_split(
        dataset, SPLIT, METHODS_UNDER_TEST, quick=True, use_cache=False)
    return dataset, rows, gp_rows, predictions


def test_one_row_per_method_with_no_region_column(run):
    _, rows, _, _ = run
    assert len(rows) == len(METHODS_UNDER_TEST)
    assert {r["method"] for r in rows} == set(METHODS_UNDER_TEST)
    assert all(r["dataset"] == "yacht" and r["split"] == SPLIT for r in rows)
    assert all("region" not in r for r in rows)


def test_seed_is_the_split_index(run):
    """DEC-017: one training run per split, and the row is reproducible from its
    own `seed` column."""
    _, rows, _, _ = run
    assert all(r["seed"] == SPLIT for r in rows)


def test_row_shape_columns_match_the_loaded_split(run):
    dataset, rows, _, _ = run
    for row in rows:
        assert row["n_train"] == 277
        assert row["n_test"] == 31
        assert row["d"] == dataset.d
        assert row["n_train"] + row["n_test"] == dataset.n


def test_metrics_are_reproducible_from_the_returned_predictions(run):
    """The DEC-007 invariant in E2's form: the written row is the metrics of the
    single `Prediction` that was made, not of a second one."""
    dataset, rows, _, predictions = run
    from src.uci import load_split
    y_test = load_split(dataset, SPLIT).y_test
    for row in rows:
        p = predictions[row["method"]]
        expected = compute_metrics(
            y=y_test, mean=p["mean"],
            var_aleatoric=p["var_aleatoric"], var_epistemic=p["var_epistemic"])
        for metric, value in expected.items():
            assert row[metric] == pytest.approx(value), (row["method"], metric)


def test_every_metric_column_is_present_and_finite(run):
    _, rows, _, _ = run
    for row in rows:
        for metric in METRIC_COLUMNS:
            assert metric in row
            assert np.isfinite(row[metric]), (row["method"], metric)


def test_map_reports_an_exactly_zero_epistemic_term(run):
    """`map` is the lower reference point: no epistemic term at all, stated as
    zeros rather than folded into the aleatoric one (`base.py`'s protocol)."""
    _, _, _, predictions = run
    assert np.all(predictions["map"]["var_epistemic"] == 0.0)
    assert np.all(predictions["map"]["var_aleatoric"] > 0.0)


def test_predictions_are_in_original_target_units(run):
    """DEC-003: metrics are scored after un-standardising. Yacht's target has
    mean ~10.5 and sd ~15, so a prediction left in standardised units would sit
    near zero and be obvious here."""
    dataset, _, _, predictions = run
    from src.uci import load_split
    data = load_split(dataset, SPLIT)
    for p in predictions.values():
        assert p["mean"].shape == (data.n_test,)
        assert abs(p["mean"].mean() - data.y_test.mean()) < 3.0 * data.y_test.std() + 5.0
        assert p["mean"].mean() > 1.0  # standardised predictions would be ~0


def test_no_gp_kernel_rows_when_gp_was_not_run(run):
    """The GP diagnostic side-file is written only for the method it describes."""
    _, _, gp_rows, _ = run
    assert gp_rows == []


def test_gp_kernel_rows_record_bounds_and_flag_values_sitting_on_them():
    """N-012: a kernel hyperparameter fixed by its bound rather than by the data
    makes the `gp` row unreportable, so the flag must fire on the log-space
    margin actually written to the file."""
    dataset = load_dataset("yacht")
    _, gp_rows, _ = run_split(dataset, SPLIT, ["gp"], quick=True, use_cache=False)
    assert gp_rows, "fitting gp must produce one row per kernel hyperparameter"
    for row in gp_rows:
        assert row["dataset"] == "yacht" and row["split"] == SPLIT
        assert row["log_lower"] <= row["log_value"] <= row["log_upper"]
        expected_margin = min(row["log_value"] - row["log_lower"],
                              row["log_upper"] - row["log_value"])
        assert row["log_margin_to_nearest_bound"] == pytest.approx(expected_margin)
        assert row["on_bound"] == (expected_margin <= BOUND_TOLERANCE)
        assert row["value"] == pytest.approx(np.exp(row["log_value"]))
        assert row["fitted_here"] is True  # use_cache=False, so L-BFGS really ran


def test_summarise_gives_mean_and_std_per_method_and_dataset(run):
    """DEC-006/DEC-015: mean +/- std over splits, grouped per dataset — no
    cross-dataset aggregate (DEC-016)."""
    import pandas as pd
    _, rows, _, _ = run
    df = pd.DataFrame(rows)
    summary = summarise(df)
    assert set(summary.columns) >= {"method", "dataset"}
    for metric in METRIC_COLUMNS:
        assert "{}_mean".format(metric) in summary.columns
        assert "{}_std".format(metric) in summary.columns
    assert len(summary) == len(METHODS_UNDER_TEST)  # one row per (method, dataset)


def test_report_only_keeps_baseline_rows_of_the_latest_run(tmp_path):
    """`--report-only` must never average a superseded run, a `--quick` run or a
    `--set` variant into the reported table. Results are appended and never
    overwritten, so all three can legitimately sit in the same file."""
    import pandas as pd
    from experiments.e2_uci import load_reported_rows

    path = tmp_path / "metrics.csv"
    pd.DataFrame([
        # superseded baseline run of yacht
        dict(run_id="20260101T000000Z", method="map", dataset="yacht", rmse=9.9,
             quick=False, variant=""),
        # current baseline run of yacht
        dict(run_id="20260202T000000Z", method="map", dataset="yacht", rmse=0.5,
             quick=False, variant=""),
        # a quick run and a variant run, both unreportable
        dict(run_id="20260303T000000Z", method="map", dataset="yacht", rmse=8.8,
             quick=True, variant=""),
        dict(run_id="20260303T000000Z", method="mcd", dataset="yacht", rmse=1.1,
             quick=False, variant="mcd:dropout_p=0.05"),
        # a different dataset, run separately — must survive alongside yacht
        dict(run_id="20260105T000000Z", method="map", dataset="energy", rmse=0.4,
             quick=False, variant=""),
    ]).to_csv(path, index=False)

    kept = load_reported_rows(path)
    assert sorted(kept.dataset) == ["energy", "yacht"]
    assert set(kept.run_id) == {"20260202T000000Z", "20260105T000000Z"}
    assert kept[kept.dataset == "yacht"].rmse.iloc[0] == 0.5


def test_report_only_refuses_a_file_with_no_baseline_rows(tmp_path):
    import pandas as pd
    import pytest as _pytest
    from experiments.e2_uci import load_reported_rows

    path = tmp_path / "metrics.csv"
    pd.DataFrame([dict(run_id="20260303T000000Z", method="map", dataset="yacht",
                       rmse=8.8, quick=True, variant="")]).to_csv(path, index=False)
    with _pytest.raises(SystemExit):
        load_reported_rows(path)


def test_quick_overrides_reach_the_constructed_method_and_defaults_are_untouched():
    """`--quick` must cut the epoch count (it is a plumbing check, not a result),
    and `--set` must win over it (DEC-014)."""
    assert build_method("map", quick=True).epochs == QUICK_EPOCHS
    assert build_method("map", quick=False).epochs > QUICK_EPOCHS
    overridden = build_method("mcd", quick=True, overrides={"mcd": {"dropout_p": 0.05}})
    assert overridden.dropout_p == 0.05
    assert overridden.epochs == QUICK_EPOCHS
