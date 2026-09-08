"""The E1 dataset: shapes, standardisation, region split, determinism.

These guard the properties every metric silently assumes. A standardiser fitted
on the wrong statistics, or a region mask that drops points, would not raise -
it would just produce a plausible, wrong table.
"""
import numpy as np
import pytest

from src.data import (
    EXTRAPOLATION, IN_RANGE, N_TEST, N_TRAIN, NOISE_SIGMA, OVERALL, TEST_RANGE, TRAIN_RANGE,
    make_toy_data, region_masks, true_function,
)


@pytest.fixture(scope="module")
def data():
    return make_toy_data(seed=0)


def test_configuration_matches_the_recorded_decisions(data):
    """N=250 (DEC-002), 500 grid points on [-2, 8] (DEC-004/006), sigma=0.1 (DEC-005)."""
    assert data.x_train.shape == (N_TRAIN, 1) == (250, 1)
    assert data.y_train.shape == (N_TRAIN,)
    assert data.x_test.shape == (N_TEST, 1) == (500, 1)
    assert data.y_test.shape == (N_TEST,)
    assert data.noise_sigma == NOISE_SIGMA == 0.1
    assert (TRAIN_RANGE, TEST_RANGE) == ((0.0, 6.0), (-2.0, 8.0))
    assert data.x_train.min() >= 0.0 and data.x_train.max() <= 6.0
    assert data.x_test.min() == pytest.approx(-2.0)
    assert data.x_test.max() == pytest.approx(8.0)


def test_standardiser_is_fitted_on_training_statistics_only(data):
    """Train standardises to exactly zero mean / unit variance; test does not.

    If the test set also standardised to N(0,1) the statistics would have been
    computed on the pooled data, leaking the extrapolation region's spread into
    the transform.
    """
    assert data.x_train_std.mean() == pytest.approx(0.0, abs=1e-12)
    assert data.x_train_std.std(ddof=0) == pytest.approx(1.0)
    assert data.y_train_std.mean() == pytest.approx(0.0, abs=1e-12)
    assert data.y_train_std.std(ddof=0) == pytest.approx(1.0)
    assert data.x_test_std.mean() != pytest.approx(0.0, abs=1e-6)


def test_unstandardise_inverts_standardise(data):
    s = data.standardiser
    assert np.allclose(s.unstandardise_mean(data.y_train_std), data.y_train)
    # a variance carries y_std squared, not y_std
    var_std = np.full(5, 0.25)
    assert np.allclose(s.unstandardise_var(var_std), var_std * s.y_std ** 2)


def test_regions_partition_the_grid_exactly(data):
    in_range, extrap, overall = (data.regions[k] for k in (IN_RANGE, EXTRAPOLATION, OVERALL))
    assert not np.any(in_range & extrap)
    assert np.all(in_range | extrap)
    assert overall.all()
    assert in_range.sum() + extrap.sum() == N_TEST
    x = data.x_test_flat
    assert x[in_range].min() >= 0.0 and x[in_range].max() <= 6.0
    assert np.all((x[extrap] < 0.0) | (x[extrap] > 6.0))


def test_no_grid_point_sits_on_a_region_boundary(data):
    """The closed/open convention in `region_masks` must not matter at this grid."""
    x = data.x_test_flat
    assert np.min(np.abs(x - 0.0)) > 1e-6
    assert np.min(np.abs(x - 6.0)) > 1e-6


def test_region_masks_keep_boundary_points_in_range():
    x = np.array([-1.0, 0.0, 3.0, 6.0, 7.0])
    masks = region_masks(x)
    assert masks[IN_RANGE].tolist() == [False, True, True, True, False]
    assert masks[EXTRAPOLATION].tolist() == [True, False, False, False, True]


def test_same_seed_gives_identical_data_and_different_seeds_do_not():
    a, b, c = make_toy_data(0), make_toy_data(0), make_toy_data(1)
    assert np.array_equal(a.x_train, b.x_train)
    assert np.array_equal(a.y_train, b.y_train)
    assert np.array_equal(a.y_test, c.y_test) is False
    # the grid itself is deterministic; only its noise moves with the seed
    assert np.array_equal(a.x_test, c.x_test)


def test_noise_level_is_as_specified(data):
    """Residuals about the true function should have sd ~= sigma on 250 points."""
    residuals = data.y_train - true_function(data.x_train).ravel()
    assert residuals.std(ddof=1) == pytest.approx(NOISE_SIGMA, rel=0.2)
