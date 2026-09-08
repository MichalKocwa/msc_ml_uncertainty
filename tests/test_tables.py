"""`src/tables.py`: the presentation layer must not change what the CSV says.

The notebook shows these tables and the thesis will paste their LaTeX, so the
failure that matters here is silent: a superseded run averaged into a current
one, a `--quick` row treated as a result, a mean and a standard deviation
swapped, or a LaTeX table that does not compile. None of those would look wrong
on screen.

The numbers are checked against hand-built frames, not against the real results,
so these tests keep passing when the results are re-run.
"""
import numpy as np
import pandas as pd
import pytest

from src import tables


def _rows(**overrides):
    row = dict(run_id="20260101T000000Z", method="map", region="in_range",
               seed=0, rmse=1.0, ll=-1.0, picp95=0.95, mpiw95=2.0,
               sigma_fitted=0.1, cal_err=0.01, cal_bias=0.0,
               quick=False, variant="")
    row.update(overrides)
    return row


def test_reported_rows_drops_quick_and_variant_runs(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame([
        _rows(rmse=1.0),
        _rows(rmse=9.9, quick=True),
        _rows(rmse=8.8, variant="mcd:dropout_p=0.05"),
    ]).to_csv(path, index=False)

    kept = tables.reported_rows(path, ["region"])
    assert len(kept) == 1
    assert kept["rmse"].iloc[0] == 1.0


def test_reported_rows_keeps_the_latest_run_per_group(tmp_path):
    """Results are appended, never overwritten, so an old run sits in the same
    file — it must not be averaged into the current one."""
    path = tmp_path / "m.csv"
    pd.DataFrame([
        _rows(run_id="20260101T000000Z", region="in_range", rmse=9.9),
        _rows(run_id="20260202T000000Z", region="in_range", rmse=1.0),
        # a group run separately and earlier must survive on its own timeline
        _rows(run_id="20260105T000000Z", region="overall", rmse=2.0),
    ]).to_csv(path, index=False)

    kept = tables.reported_rows(path, ["region"])
    assert sorted(kept["rmse"]) == [1.0, 2.0]


def test_reported_rows_rejects_a_file_with_nothing_reportable(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame([_rows(quick=True)]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        tables.reported_rows(path, ["region"])


def test_format_summary_is_mean_then_std_at_the_documented_precision():
    df = pd.DataFrame([
        _rows(method="map", seed=0, rmse=1.0, sigma_fitted=0.10),
        _rows(method="map", seed=1, rmse=2.0, sigma_fitted=0.20),
    ])
    table = tables.format_summary(tables.summarise(df, ["method"]))
    # mean 1.5, sample std (ddof=1) of {1, 2} is 0.7071...
    assert table.loc["map", "RMSE"] == "1.500 ± 0.707"
    # sigma_fitted prints four decimals, per METRIC_DIGITS
    assert table.loc["map", "sigma_fitted"] == "0.1500 ± 0.0707"


def test_format_summary_orders_methods_and_omits_absent_ones():
    df = pd.DataFrame([
        _rows(method="gp", seed=0), _rows(method="gp", seed=1),
        _rows(method="map", seed=0), _rows(method="map", seed=1),
    ])
    table = tables.format_summary(tables.summarise(df, ["method"]))
    assert list(table.index) == ["map", "gp"]  # METHOD_ORDER, not alphabetical


def test_a_single_repetition_reports_the_std_as_not_available():
    """One seed gives no spread; printing 0.000 would claim a precision that was
    never measured."""
    df = pd.DataFrame([_rows(method="map", seed=0)])
    table = tables.format_summary(tables.summarise(df, ["method"]))
    assert table.loc["map", "RMSE"].endswith("± n/a")


def test_stage_tables_follows_the_requested_group_order():
    df = pd.DataFrame([
        _rows(method="map", seed=s, region=r)
        for r in ("overall", "in_range") for s in (0, 1)
    ])
    got = tables.stage_tables(df, "region", tables.E1_REGION_ORDER)
    assert list(got) == ["in_range", "overall"]  # requested order, not CSV order


def test_band_expansion_ratio_is_extrapolation_over_in_range():
    df = pd.DataFrame([
        _rows(method="map", seed=0, region="in_range", mpiw95=1.0),
        _rows(method="map", seed=0, region="extrapolation", mpiw95=1.0),
        _rows(method="gp", seed=0, region="in_range", mpiw95=1.0),
        _rows(method="gp", seed=0, region="extrapolation", mpiw95=3.0),
    ])
    out = tables.e1_band_expansion(df)
    assert out.loc["gp", "iloraz"] == 3.0
    assert out.loc["map", "iloraz"] == 1.0
    assert list(out.index) == ["gp", "map"]  # sorted by expansion, widest first


def test_repetitions_counts_seeds_for_e1_and_splits_for_e2():
    e1 = pd.DataFrame([_rows(seed=s, region="in_range") for s in range(3)])
    assert tables.repetitions(e1, "region", "in_range") == 3

    e2 = pd.DataFrame([
        dict(method="map", dataset="yacht", split=s, rmse=1.0) for s in range(5)])
    assert tables.repetitions(e2, "dataset", "yacht") == 5


def test_latex_is_booktabs_and_escapes_what_would_not_compile():
    df = pd.DataFrame([_rows(method="map", seed=0), _rows(method="map", seed=1)])
    table = tables.format_summary(tables.summarise(df, ["method"]))
    latex = tables.to_latex(table, caption="Podpis", label="tab:x")

    for token in (r"\begin{table}", r"\toprule", r"\midrule", r"\bottomrule",
                  r"\end{tabular}", r"\caption{Podpis}", r"\label{tab:x}"):
        assert token in latex
    assert "±" not in latex and r"$\pm$" in latex
    assert r"sigma\_fitted" in latex          # underscores escaped in headers
    assert latex.count(r"\\") == table.shape[0] + 1  # one header row plus the body
    # column spec matches the column count: one l for the index, r for each metric
    assert r"\begin{tabular}{l" + "r" * table.shape[1] + "}" in latex


def test_latex_row_count_matches_the_table():
    df = pd.DataFrame([
        _rows(method=m, seed=s) for m in ("map", "gp") for s in (0, 1)])
    table = tables.format_summary(tables.summarise(df, ["method"]))
    latex = tables.to_latex(table, "c", "l")
    body = [line for line in latex.splitlines() if line.startswith("    ")]
    # header + two method rows
    assert len([b for b in body if b.endswith(r"\\")]) == 3


def test_e1_noise_check_reads_against_the_known_true_sigma():
    """E1 is the only stage with a ground-truth noise level to compare against
    (DEC-005); the variance ratio is the sd ratio squared."""
    df = pd.DataFrame([
        _rows(method="mcd", seed=s, region="in_range", sigma_fitted=0.2)
        for s in (0, 1)
    ] + [
        _rows(method="map", seed=s, region="in_range", sigma_fitted=0.1)
        for s in (0, 1)
    ])
    df["noise_sigma"] = 0.1
    out = tables.e1_noise_check(df)
    assert out.loc["mcd", "iloraz (sd)"] == 2.0
    assert out.loc["mcd", "iloraz (wariancja)"] == 4.0
    assert out.loc["map", "iloraz (sd)"] == 1.0


def test_mcd_inflation_is_measured_against_the_other_five_methods():
    """The E2 form of N-001: relative only, since UCI has no known noise level
    (DEC-010's amendment)."""
    rows = []
    for method, sigma in [("mcd", 3.0), ("map", 1.0), ("ensemble", 1.0),
                          ("bbb", 1.0), ("laplace", 1.0), ("gp", 1.0)]:
        rows.append(dict(method=method, dataset="yacht", split=0, sigma_fitted=sigma))
    out = tables.mcd_noise_inflation(pd.DataFrame(rows))
    assert out.loc["yacht", "iloraz (sd)"] == 3.0
    assert out.loc["yacht", "iloraz (wariancja)"] == 9.0
    assert out.loc["yacht", "pozostałe pięć (średnia)"] == 1.0


def test_metric_headers_and_digits_cover_every_reported_metric():
    """A metric added to `src/metrics.py` must not silently vanish from the
    tables or print at pandas' default precision."""
    from src.metrics import METRIC_COLUMNS
    assert set(METRIC_COLUMNS) <= set(tables.METRIC_HEADERS)
    assert set(METRIC_COLUMNS) <= set(tables.METRIC_DIGITS)


def test_formatted_values_round_trip_to_the_underlying_numbers():
    """Guards the whole point of the module: what is displayed is what was
    measured, to the precision displayed."""
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, size=20)
    df = pd.DataFrame([_rows(method="map", seed=i, rmse=v) for i, v in enumerate(values)])
    table = tables.format_summary(tables.summarise(df, ["method"]))
    shown_mean, shown_std = table.loc["map", "RMSE"].split(" ± ")
    assert float(shown_mean) == pytest.approx(values.mean(), abs=5e-4)
    assert float(shown_std) == pytest.approx(values.std(ddof=1), abs=5e-4)
