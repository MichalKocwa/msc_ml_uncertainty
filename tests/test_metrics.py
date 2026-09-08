"""The four metrics plus the diagnostic column, against hand-computed values.

The important one is `test_log_likelihood_in_original_units_matches_the_dec_003_correction`:
DEC-003 says models are fitted on standardised data but scored in original y
units, and the two differ by exactly `-log(sigma_y)` per point. E1 takes the
route of un-standardising first and scoring directly; this test asserts that
route agrees with the correction term the decision is written in terms of, so
the CSV cannot silently be off by a constant.
"""
import math

import numpy as np
import pytest
from scipy.stats import norm

from src.metrics import (
    CALIBRATION_LEVELS, METRIC_COLUMNS, Z95, calibration_curve, calibration_error,
    compute_metrics, fitted_sigma, log_likelihood, mpiw, picp, rmse,
)


def test_z95_is_the_exact_two_sided_95_percent_quantile():
    """DEC-008: metrics use 1.959964, not 2.0, so that '@95' is literally true."""
    assert Z95 == pytest.approx(1.959963984540054, abs=1e-12)
    assert norm.cdf(Z95) - norm.cdf(-Z95) == pytest.approx(0.95)


def test_rmse_against_a_hand_computed_value():
    y = np.array([1.0, 2.0, 3.0])
    mean = np.array([1.0, 4.0, 3.0])
    assert rmse(y, mean) == pytest.approx(math.sqrt(4.0 / 3.0))


def test_log_likelihood_includes_the_normalising_constant():
    """A residual of zero at unit variance must give exactly -0.5*log(2*pi).

    `backbone.gaussian_nll` drops that constant because it does not move an
    optimiser's argmin; a REPORTED log-likelihood that dropped it would be off
    by 0.919 and incomparable to any published table.
    """
    y = np.zeros(4)
    assert log_likelihood(y, y, np.ones(4)) == pytest.approx(-0.5 * math.log(2 * math.pi))


def test_log_likelihood_matches_scipy():
    rng = np.random.default_rng(0)
    y, mean, var = rng.normal(size=50), rng.normal(size=50), rng.uniform(0.1, 2.0, size=50)
    expected = np.mean(norm.logpdf(y, loc=mean, scale=np.sqrt(var)))
    assert log_likelihood(y, mean, var) == pytest.approx(expected)


def test_log_likelihood_in_original_units_matches_the_dec_003_correction():
    """LL(original) == LL(standardised) - log(sigma_y), exactly."""
    rng = np.random.default_rng(1)
    y_std, mean_std = rng.normal(size=200), rng.normal(size=200)
    var_std = rng.uniform(0.1, 1.5, size=200)
    sigma_y, mu_y = 2.75, -0.4

    ll_standardised = log_likelihood(y_std, mean_std, var_std)
    ll_original = log_likelihood(
        y_std * sigma_y + mu_y, mean_std * sigma_y + mu_y, var_std * sigma_y ** 2)
    assert ll_original == pytest.approx(ll_standardised - math.log(sigma_y))


def test_picp_counts_points_inside_the_interval():
    mean, var = np.zeros(4), np.ones(4)
    # boundaries are inclusive; +/-Z95 sits exactly on the edge
    y = np.array([0.0, Z95, -Z95, Z95 + 1e-9])
    assert picp(y, mean, var) == pytest.approx(0.75)


def test_picp_is_close_to_nominal_for_a_correctly_specified_gaussian():
    rng = np.random.default_rng(2)
    y = rng.normal(size=200_000)
    mean, var = np.zeros(200_000), np.ones(200_000)
    assert picp(y, mean, var) == pytest.approx(0.95, abs=0.01)


def test_mpiw_is_twice_z_times_the_standard_deviation():
    var = np.array([1.0, 4.0])
    assert mpiw(var) == pytest.approx(Z95 * (2.0 * 1.0 + 2.0 * 2.0) / 2.0)


def test_fitted_sigma_is_the_root_mean_aleatoric_variance():
    assert fitted_sigma(np.array([0.01, 0.09])) == pytest.approx(math.sqrt(0.05))
    # constant aleatoric term (every method as configured here) recovers sigma exactly
    assert fitted_sigma(np.full(10, 0.01)) == pytest.approx(0.1)


def test_compute_metrics_uses_the_total_variance_and_reports_every_column():
    y = np.array([0.0, 0.0])
    mean = np.array([0.0, 0.0])
    var_a, var_e = np.array([0.25, 0.25]), np.array([0.75, 0.75])
    out = compute_metrics(y, mean, var_a, var_e)
    assert set(out) == set(METRIC_COLUMNS)
    assert out["mpiw95"] == pytest.approx(2.0 * Z95 * 1.0)          # total sd is 1.0
    assert out["sigma_fitted"] == pytest.approx(0.5)                 # aleatoric only
    assert out["ll"] == pytest.approx(-0.5 * math.log(2 * math.pi))


def test_zero_variance_is_rejected_rather_than_producing_infinity():
    """`map` reports var_epistemic as exact zeros; only a zero TOTAL is fatal."""
    y = mean = np.zeros(3)
    with pytest.raises(ValueError):
        compute_metrics(y, mean, np.zeros(3), np.zeros(3))
    compute_metrics(y, mean, np.full(3, 0.01), np.zeros(3))  # must not raise


# --- calibration (DEC-013) ---------------------------------------------------

def test_calibration_curve_is_exact_for_a_correctly_specified_gaussian():
    """Empirical coverage must track the nominal levels on the diagonal."""
    rng = np.random.default_rng(3)
    n = 400_000
    y = rng.normal(size=n)
    mean, var = np.zeros(n), np.ones(n)
    coverage = calibration_curve(y, mean, var)
    assert np.allclose(coverage, CALIBRATION_LEVELS, atol=0.005)


def test_calibration_curve_passes_through_picp95():
    """The 0.95 entry of the curve must BE PICP@95, not merely resemble it.

    They are computed by different code paths (`norm.ppf(0.975)` inside the
    curve, the `Z95` constant in `picp`), so this pins them together: a table
    reporting both must not be able to show two different numbers for the same
    quantity.
    """
    rng = np.random.default_rng(4)
    y, mean = rng.normal(size=500), rng.normal(size=500) * 0.2
    var = rng.uniform(0.5, 1.5, size=500)
    idx = CALIBRATION_LEVELS.index(0.95)
    assert calibration_curve(y, mean, var)[idx] == pytest.approx(picp(y, mean, var))


def test_calibration_error_is_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(5)
    n = 400_000
    y = rng.normal(size=n)
    err, bias = calibration_error(y, np.zeros(n), np.ones(n))
    assert err == pytest.approx(0.0, abs=0.005)
    assert bias == pytest.approx(0.0, abs=0.005)


def test_cal_bias_sign_distinguishes_overconfidence_from_undercoverage():
    """The reason `cal_bias` exists alongside `cal_err`: direction.

    Too-narrow intervals under-cover (negative bias); too-wide ones over-cover
    (positive). `cal_err` alone gives both the same score, and for this
    comparison that is most of the point.
    """
    rng = np.random.default_rng(6)
    n = 200_000
    y = rng.normal(size=n)
    mean = np.zeros(n)

    _, bias_narrow = calibration_error(y, mean, np.full(n, 0.25))   # sd 0.5, too narrow
    _, bias_wide = calibration_error(y, mean, np.full(n, 4.0))      # sd 2.0, too wide
    assert bias_narrow < -0.1
    assert bias_wide > 0.1

    err_narrow, _ = calibration_error(y, mean, np.full(n, 0.25))
    err_wide, _ = calibration_error(y, mean, np.full(n, 4.0))
    assert err_narrow > 0.1 and err_wide > 0.1   # both penalised by the absolute score


def test_a_displaced_mean_is_miscalibrated_even_at_the_right_width():
    """The failure mode `PICP@95` alone cannot separate from a narrow band.

    Right predictive width, wrong location: coverage collapses at every level,
    which is what `map` does in E1's extrapolation region.
    """
    rng = np.random.default_rng(7)
    n = 200_000
    y = rng.normal(size=n) + 3.0            # truth sits 3 sd away from the prediction
    err, bias = calibration_error(y, np.zeros(n), np.ones(n))
    assert bias < -0.3
    assert err > 0.3


def test_compute_metrics_reports_the_calibration_columns():
    rng = np.random.default_rng(8)
    n = 5000
    y = rng.normal(size=n)
    out = compute_metrics(y, np.zeros(n), np.full(n, 0.5), np.full(n, 0.5))
    assert set(out) == set(METRIC_COLUMNS)
    assert out["cal_err"] == pytest.approx(0.0, abs=0.02)
