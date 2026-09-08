"""The UCI loader: split integrity, column selection and train-only standardisation.

These are the properties a wrong number in `results/e2_uci_metrics.csv` would
most plausibly come from, and none of them is visible in the metrics themselves:
a train/test overlap inflates every method's score at once, a mis-read
`index_features` column silently regresses on the wrong variable, and
standardising on pooled statistics leaks the test rows' spread into the
transform. All three would produce plausible-looking numbers.

The split checks run over **all 20 splits of all three datasets** (60 in total),
not just split 0 — the shapes were originally verified on split 0 alone, which
would not have caught a single malformed file among the other 19.

Reads the fetched data, so it needs `data/uci_splits/` (`scripts/fetch_data.py`);
the whole module skips if that is absent rather than failing, since the data is
gitignored and a fresh clone legitimately does not have it.
"""
import numpy as np
import pytest

from src.uci import UCI_DATASETS, UCI_ROOT, load_dataset, load_reference, load_split

pytestmark = pytest.mark.skipif(
    not UCI_ROOT.is_dir(),
    reason="data/uci_splits/ absent; run scripts/fetch_data.py")

# N, d, N_train, N_test as verified from the fetched files. N_train/N_test are
# asserted for EVERY split below, not only the first.
EXPECTED = {
    "yacht": dict(n=308, d=6, n_train=277, n_test=31),
    "energy": dict(n=768, d=8, n_train=691, n_test=77),
    "concrete": dict(n=1030, d=8, n_train=927, n_test=103),
}


@pytest.fixture(scope="module")
def datasets():
    return {name: load_dataset(name) for name in UCI_DATASETS}


@pytest.mark.parametrize("name", UCI_DATASETS)
def test_dataset_has_the_expected_shape(datasets, name):
    dataset = datasets[name]
    expected = EXPECTED[name]
    assert dataset.X.shape == (expected["n"], expected["d"])
    assert dataset.y.shape == (expected["n"],)
    assert dataset.n_splits == 20


@pytest.mark.parametrize("name", UCI_DATASETS)
def test_feature_and_target_columns_are_the_ones_the_index_files_name(datasets, name):
    """`index_features.txt` / `index_target.txt` are COLUMN indices into
    `data.txt`; reading them as anything else would regress on the wrong
    variable while every shape stayed right."""
    dataset = datasets[name]
    raw = np.loadtxt(UCI_ROOT / name / "data.txt")
    assert dataset.target_column not in set(dataset.feature_columns.tolist())
    assert np.array_equal(dataset.y, raw[:, dataset.target_column])
    for j, column in enumerate(dataset.feature_columns):
        assert np.array_equal(dataset.X[:, j], raw[:, column])


@pytest.mark.parametrize("name", UCI_DATASETS)
def test_every_split_partitions_the_dataset_with_no_overlap(datasets, name):
    """The property that matters most: no row is both trained on and tested on.

    Checked for all 20 splits, together with the train/test sizes — the sizes
    must be constant across splits, not merely correct for split 0.

    Disjointness is asserted at the level of ROW INDICES, which is what the
    protocol fixes and what the loader guarantees. It is deliberately not
    asserted on row VALUES: `concrete` contains duplicated observations, so a
    test row can be value-identical to a training row while the two are
    different rows of the file (N-018). The next test measures that separately
    rather than letting it look like a leak here.
    """
    dataset = datasets[name]
    expected = EXPECTED[name]
    for split in range(dataset.n_splits):
        data = load_split(dataset, split)
        assert data.n_train == expected["n_train"], split
        assert data.n_test == expected["n_test"], split
        assert data.n_train + data.n_test == dataset.n, split
        assert data.d == expected["d"], split

        train_idx = np.loadtxt(UCI_ROOT / name / "index_train_{}.txt".format(split)).astype(int)
        test_idx = np.loadtxt(UCI_ROOT / name / "index_test_{}.txt".format(split)).astype(int)
        assert not set(train_idx) & set(test_idx), (name, split)
        assert len(set(train_idx) | set(test_idx)) == dataset.n, (name, split)

        # ...and that those indices are the rows that actually reached the
        # method, not merely that the index files are well formed.
        assert np.array_equal(data.x_train, dataset.X[train_idx]), (name, split)
        assert np.array_equal(data.y_test, dataset.y[test_idx]), (name, split)


@pytest.mark.parametrize("name", UCI_DATASETS)
def test_duplicate_observations_are_measured_not_assumed_absent(datasets, name):
    """How many test rows are value-identical to a training row (N-018).

    `yacht` and `energy` contain no duplicated rows at all, so the count must be
    exactly zero and any non-zero value would be a real leak. `concrete` ships
    25 exactly duplicated `(X, y)` rows in `data.txt`, so 70 of its 2060 test
    rows across the 20 splits repeat a training row verbatim — a property of the
    dataset Gal actually trained on, inherited with the splits and reported
    rather than fixed (N-016 forbids the deduplication that would remove it).

    Pinned as an exact count so that a change in the fetched data, or a loader
    bug that genuinely mixed the two sides, would still fail here.
    """
    expected_duplicates = {"yacht": 0, "energy": 0, "concrete": 70}
    dataset = datasets[name]
    total = 0
    for split in range(dataset.n_splits):
        data = load_split(dataset, split)
        train_rows = {tuple(row) for row in np.column_stack([data.x_train, data.y_train])}
        total += sum(
            tuple(row) in train_rows
            for row in np.column_stack([data.x_test, data.y_test]))
    assert total == expected_duplicates[name]


@pytest.mark.parametrize("name", UCI_DATASETS)
def test_standardisation_uses_training_statistics_only(datasets, name):
    """DEC-003. Training rows standardise to 0/1; test rows must NOT.

    A test set that also came out at exactly mean 0 and std 1 would be the
    signature of pooled statistics — the leak this test exists to catch.
    """
    dataset = datasets[name]
    data = load_split(dataset, 0)

    assert np.allclose(data.x_train_std.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(data.x_train_std.std(axis=0, ddof=0), 1.0, atol=1e-10)
    assert data.y_train_std.mean() == pytest.approx(0.0, abs=1e-10)
    assert data.y_train_std.std(ddof=0) == pytest.approx(1.0, abs=1e-10)

    assert not np.allclose(data.x_test_std.mean(axis=0), 0.0, atol=1e-6)
    assert not np.allclose(data.x_test_std.std(axis=0, ddof=0), 1.0, atol=1e-6)

    s = data.standardiser
    assert np.allclose(s.x_mean, data.x_train.mean(axis=0))
    assert np.allclose(s.x_std, data.x_train.std(axis=0, ddof=0))
    assert s.y_mean == pytest.approx(data.y_train.mean())
    assert s.y_std == pytest.approx(data.y_train.std(ddof=0))


@pytest.mark.parametrize("name", UCI_DATASETS)
def test_no_feature_has_zero_variance_in_any_training_split(datasets, name):
    """Standardisation divides by the per-feature training std, so a constant
    feature in any of the 60 training splits would produce inf/NaN inputs."""
    dataset = datasets[name]
    for split in range(dataset.n_splits):
        std = load_split(dataset, split).x_train.std(axis=0, ddof=0)
        assert std.min() > 1e-3, (name, split, float(std.min()))


@pytest.mark.parametrize("name", UCI_DATASETS)
def test_unstandardise_round_trips_and_scales_variance_by_y_std_squared(datasets, name):
    """`unstandardise_var` scales by `y_std ** 2`, not `y_std`: a variance is in
    squared target units. Getting it wrong leaves the bands right and the
    log-likelihood wrong by a constant."""
    data = load_split(datasets[name], 0)
    s = data.standardiser

    assert np.allclose(s.unstandardise_mean(data.y_train_std), data.y_train)
    assert np.allclose(s.standardise_x(data.x_test), data.x_test_std)

    var_std = np.array([0.25, 1.0, 4.0])
    assert np.allclose(s.unstandardise_var(var_std), var_std * s.y_std ** 2)


@pytest.mark.parametrize("name", UCI_DATASETS)
def test_loading_the_same_split_twice_gives_identical_arrays(datasets, name):
    """The split is read from fixed index files, so it carries no randomness —
    hence no seeding in `src/uci.py`. This pins that."""
    a, b = load_split(datasets[name], 3), load_split(datasets[name], 3)
    assert np.array_equal(a.x_train, b.x_train)
    assert np.array_equal(a.y_test, b.y_test)


@pytest.mark.parametrize("name", UCI_DATASETS)
def test_out_of_range_split_is_rejected(datasets, name):
    with pytest.raises(ValueError):
        load_split(datasets[name], datasets[name].n_splits)


@pytest.mark.parametrize("name", UCI_DATASETS)
def test_reference_results_have_one_value_per_split(name):
    """The published per-split numbers are the external sanity check; there must
    be exactly 20 of each, and they are never written into our results CSV."""
    reference = load_reference(name)
    assert reference["rmse"].shape == (20,)
    assert reference["ll"].shape == (20,)
    assert np.all(reference["rmse"] > 0)
    assert np.all(np.isfinite(reference["ll"]))
