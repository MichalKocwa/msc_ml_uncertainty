"""UCI regression datasets for E2, on the Hernandez-Lobato & Adams (2015) splits.

Reads the files `scripts/fetch_data.py` copies out of
`yaringal/DropoutUncertaintyExps` at its pinned commit — the data matrix plus
the twenty 90/10 train/test index files that Gal & Ghahramani (2016) and
Lakshminarayanan et al. (2017) report on. The vendored `data.txt` is plain
text and is the supported path: `concrete` and `energy` are `.xls`/`.xlsx`
upstream at UCI, no Excel reader is installed, and `CLAUDE.md` forbids adding
one.

**Preprocessing is standardisation and nothing else (N-016).** HLA specify
exactly one step — "*the input features and the targets have zero mean and
unit variance in the training set. The normalization on the targets is removed
for prediction*" — which is DEC-003 verbatim. Log transforms, outlier removal,
one-hot encoding and feature selection are all deliberately absent: any of them
would forfeit the comparison against the published per-split results in
`data/uci_splits/<name>/results/`, which is E2's only external validation. Two
consequences are reported rather than fixed (N-016): `yacht`'s target is
strongly right-skewed and non-negative, which a homoscedastic Gaussian
likelihood fits badly; `energy` has one binary and three four-level features
that are standardised as continuous, exactly as upstream does.

**This module does not seed anything, deliberately.** A split is read from
fixed index files, so there is no randomness in it to control. `set_seed` runs
where it does work — inside each method's `fit` — and in E2 the seed therefore
governs only model initialisation and batch order, not the data. That differs
from E1, where the seed drew the data as well; the consequence for how E2's
spread must be read is recorded in DEC-017.

Kept separate from `src/data.py`'s toy problem: `make_toy_data`, `REGIONS`,
`IN_RANGE` and `true_function` are E1-specific and must not grow UCI branches.
`Standardiser` is the one piece of that module reused here, so both stages
standardise through the same code.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.data import Standardiser

REPO_ROOT = Path(__file__).resolve().parent.parent
UCI_ROOT = REPO_ROOT / "data" / "uci_splits"

# DEC-011: yacht, energy and concrete only. The list is fixed there on measured
# cost (N-012) and matches `CLAUDE.md`'s stage table; adding to it is a scope
# decision needing a new DEC, not an edit here.
UCI_DATASETS: Tuple[str, ...] = ("yacht", "energy", "concrete")

# The published reference run these datasets ship: one hidden layer, 40 x 100
# epochs (`n_epochs.txt` = 40), 20 lines per file, one per split.
# `test_MC_rmse_*` — the MC-dropout-averaged prediction — is the column that
# reproduces the upstream README table; `test_rmse_*` is the single
# deterministic-pass RMSE and is not the comparison to make.
REFERENCE_RMSE_FILE = "results/test_MC_rmse_100_xepochs_1_hidden_layers.txt"
REFERENCE_LL_FILE = "results/test_ll_100_xepochs_1_hidden_layers.txt"


@dataclass(frozen=True)
class UciDataset:
    """One dataset's full matrix, already separated into features and target.

    `X` is `(N, d)` and `y` is `(N,)` — the shapes every method's `fit`/`predict`
    expects, matching `ToyData`'s convention so the two stages feed the methods
    identically.
    """
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_columns: np.ndarray
    target_column: int
    n_splits: int

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])


@dataclass(frozen=True)
class UciSplit:
    """One (dataset, split) pair, raw and standardised.

    Mirrors `src.data.ToyData` field for field where the two overlap, so the
    experiment scripts differ only in how they build this object.
    """
    name: str
    split: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_train_std: np.ndarray
    y_train_std: np.ndarray
    x_test_std: np.ndarray
    standardiser: Standardiser

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.x_test.shape[0])

    @property
    def d(self) -> int:
        return int(self.x_train.shape[1])


def dataset_dir(name: str, root: Path = UCI_ROOT) -> Path:
    path = root / name
    if not path.is_dir():
        raise FileNotFoundError(
            "no fetched data for {!r} at {}. Run `.venv/Scripts/python.exe "
            "scripts/fetch_data.py` from the repository root.".format(name, path))
    return path


def _load_indices(path: Path, n_rows: int) -> np.ndarray:
    """Read one index file as row indices, checking they really are indices.

    The files store integers as text but `np.loadtxt` returns float64, so the
    integrality check happens before the cast rather than being assumed by it —
    a non-integral value here would otherwise be silently truncated into a valid
    but wrong row number.
    """
    raw = np.loadtxt(path, dtype=np.float64, ndmin=1)
    if not np.all(raw == np.floor(raw)):
        raise ValueError("{} contains a non-integral row index".format(path))
    idx = raw.astype(np.int64)
    if idx.min() < 0 or idx.max() >= n_rows:
        raise ValueError("{} indexes outside [0, {})".format(path, n_rows))
    if len(np.unique(idx)) != len(idx):
        raise ValueError("{} contains a repeated row index".format(path))
    return idx


def load_dataset(name: str, root: Path = UCI_ROOT) -> UciDataset:
    """Load `data.txt` and split it into features and target by the index files.

    `index_features.txt` and `index_target.txt` hold **column** indices, not row
    indices — the one place in this layout where two kinds of index file look
    alike and mean different things. `energy` already carries a single target
    here (the original UCI file has two; the vendored version keeps one and
    `index_target` = 8 points at it), so there is nothing to choose.
    """
    path = dataset_dir(name, root)
    data = np.loadtxt(path / "data.txt", dtype=np.float64)
    feature_columns = _load_indices(path / "index_features.txt", data.shape[1])
    target_raw = np.loadtxt(path / "index_target.txt", dtype=np.float64, ndmin=1)
    if target_raw.size != 1:
        raise ValueError("{} does not name exactly one target column".format(path))
    target_column = int(target_raw[0])
    if target_column in set(feature_columns.tolist()):
        raise ValueError("{}: the target column is also listed as a feature".format(path))

    n_splits = int(np.loadtxt(path / "n_splits.txt", dtype=np.float64))
    return UciDataset(
        name=name,
        X=data[:, feature_columns],
        y=data[:, target_column],
        feature_columns=feature_columns,
        target_column=target_column,
        n_splits=n_splits,
    )


def load_split(dataset: UciDataset, split: int, root: Path = UCI_ROOT) -> UciSplit:
    """Build split `split` of `dataset`, standardised on TRAINING rows only.

    Takes an already-loaded `UciDataset` rather than a name so that the twenty
    splits of one dataset re-read `data.txt` once, not twenty times.

    The train/test indices are checked here to be disjoint and to cover every
    row exactly once. They are the literature's splits and are expected to hold,
    so this is a guard against a corrupted or mis-copied file rather than a
    property under test — but it costs nothing per split and it fails loudly
    instead of quietly scoring a method on rows it trained on.
    """
    if not 0 <= split < dataset.n_splits:
        raise ValueError("split {} outside [0, {}) for {}".format(
            split, dataset.n_splits, dataset.name))
    path = dataset_dir(dataset.name, root)
    train_idx = _load_indices(path / "index_train_{}.txt".format(split), dataset.n)
    test_idx = _load_indices(path / "index_test_{}.txt".format(split), dataset.n)
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("{} split {}: train and test indices overlap".format(
            dataset.name, split))
    if np.union1d(train_idx, test_idx).size != dataset.n:
        raise ValueError("{} split {}: indices do not cover every row".format(
            dataset.name, split))

    x_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
    x_test, y_test = dataset.X[test_idx], dataset.y[test_idx]

    # DEC-003 / N-015: training statistics only. Using pooled or test statistics
    # would leak the test rows' spread into the transform and flatter every
    # method at once. `ddof=0` matches `src/data.py`.
    standardiser = Standardiser(
        x_mean=x_train.mean(axis=0),
        x_std=x_train.std(axis=0, ddof=0),
        y_mean=float(y_train.mean()),
        y_std=float(y_train.std(ddof=0)),
    )

    return UciSplit(
        name=dataset.name,
        split=split,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_train_std=standardiser.standardise_x(x_train),
        y_train_std=standardiser.standardise_y(y_train),
        x_test_std=standardiser.standardise_x(x_test),
        standardiser=standardiser,
    )


def load_reference(name: str, root: Path = UCI_ROOT) -> Dict[str, np.ndarray]:
    """Gal & Ghahramani's own per-split test RMSE and log-likelihood.

    **A sanity check, never a result of this repository.** These numbers come
    from another repository's run and are used only to confirm that ours land in
    the same range; they are deliberately kept out of
    `results/e2_uci_metrics.csv` so that no reader can mistake a fetched number
    for a measured one. They are also on a different convention in two ways that
    must be stated wherever they are quoted: they are Gal's log-sum-exp mixture
    log-likelihood, not DEC-009's moment-matched Gaussian, and the published
    tables report mean +/- standard error where DEC-006 reports mean +/- std.
    """
    path = dataset_dir(name, root)
    return {
        "rmse": np.loadtxt(path / REFERENCE_RMSE_FILE, dtype=np.float64, ndmin=1),
        "ll": np.loadtxt(path / REFERENCE_LL_FILE, dtype=np.float64, ndmin=1),
    }
