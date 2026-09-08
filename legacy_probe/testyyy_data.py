"""The `ml_TESTYYY` sine recipe, vendored so this folder stays self-contained.

Copied from `C:/Users/michal/Desktop/ml_TESTYYY/src/data.py` and
`src/seeding.py`. Only the imports are rewritten (`from src.seeding import
set_seed` becomes the local copy below); every value, every call and the order
of the random draws are unchanged, so `make_toy_data(seed)` here returns the
same arrays as `make_toy_data(seed)` there. `check_matches_testyyy()` asserts
exactly that, and is run in the notebook's first cell.

**How this differs from the `legacy_data.py` recipe in the same folder** (which
is the `ml_uncertainty_msc` one):

    ml_uncertainty_msc              ml_TESTYYY (this file)
    x_train = linspace(0, 6, 250)   x_train = rng.uniform(0, 6, 250)   <- random, not a grid
    grid    = linspace(-2, 8, 1000) grid    = linspace(-2, 8, 500)
    RandomState(seed)               default_rng(seed)
    no standardisation in E1        Standardiser fitted on the training split
    3 variants (homo/hetero/gap)    1 variant

**A note on `set_seed` and dtype.** The faithful copy of `set_seed` below sets
torch's default dtype to float64, because that is what the current methods
need. The legacy methods in this folder build their tensors as float32
explicitly, so a float64 default makes `nn.Linear` weights float64 and the
first forward pass raises. The returned ARRAY VALUES do not depend on
`set_seed` at all — they come from `np.random.default_rng(seed)`, which does
not read torch or global NumPy state — so the notebook calls
`restore_float32_default()` immediately after building the data. That restores
the environment the legacy methods were written for without touching a single
number in the dataset.
"""
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

REFERENCE_REPO = Path(os.environ.get(
    "ML_TESTYYY_REPO", r"C:/Users/michal/Desktop/ml_TESTYYY"))

# ----------------------------------------------------------------- seeding.py
TORCH_THREADS_ENV = "ML_UNCERTAINTY_TORCH_THREADS"
DEFAULT_TORCH_THREADS = 1


def set_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    """Verbatim copy of `ml_TESTYYY/src/seeding.py::set_seed`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(os.environ.get(TORCH_THREADS_ENV, DEFAULT_TORCH_THREADS)))

    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def restore_float32_default() -> None:
    """Undo the parts of `set_seed` the legacy float32 methods cannot live with.

    Not part of the original module — see this file's docstring. Leaves the RNG
    state alone, so anything drawn after this call is still seed-determined.
    """
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(False)


# -------------------------------------------------------------------- data.py
TRAIN_RANGE: Tuple[float, float] = (0.0, 6.0)
TEST_RANGE: Tuple[float, float] = (-2.0, 8.0)
N_TRAIN = 250
N_TEST = 500
NOISE_SIGMA = 0.1

IN_RANGE = "in_range"
EXTRAPOLATION = "extrapolation"
OVERALL = "overall"
REGIONS = (IN_RANGE, EXTRAPOLATION, OVERALL)


@dataclass(frozen=True)
class Standardiser:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def standardise_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def standardise_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def unstandardise_mean(self, mean: np.ndarray) -> np.ndarray:
        return mean * self.y_std + self.y_mean

    def unstandardise_var(self, var: np.ndarray) -> np.ndarray:
        return var * self.y_std ** 2


@dataclass(frozen=True)
class ToyData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_train_std: np.ndarray
    y_train_std: np.ndarray
    x_test_std: np.ndarray
    standardiser: Standardiser
    regions: Dict[str, np.ndarray]
    seed: int
    noise_sigma: float

    @property
    def x_test_flat(self) -> np.ndarray:
        return self.x_test.ravel()


def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def region_masks(x: np.ndarray, train_range: Tuple[float, float] = TRAIN_RANGE) -> Dict[str, np.ndarray]:
    x = np.asarray(x).ravel()
    lo, hi = train_range
    in_range = (x >= lo) & (x <= hi)
    return {
        IN_RANGE: in_range,
        EXTRAPOLATION: ~in_range,
        OVERALL: np.ones_like(in_range, dtype=bool),
    }


def make_toy_data(
    seed: int,
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    noise_sigma: float = NOISE_SIGMA,
    train_range: Tuple[float, float] = TRAIN_RANGE,
    test_range: Tuple[float, float] = TEST_RANGE,
) -> ToyData:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    x_train = rng.uniform(train_range[0], train_range[1], size=(n_train, 1))
    y_train = true_function(x_train).ravel() + rng.normal(0.0, noise_sigma, size=n_train)

    x_test = np.linspace(test_range[0], test_range[1], n_test).reshape(-1, 1)
    y_test = true_function(x_test).ravel() + rng.normal(0.0, noise_sigma, size=n_test)

    standardiser = Standardiser(
        x_mean=x_train.mean(axis=0),
        x_std=x_train.std(axis=0, ddof=0),
        y_mean=float(y_train.mean()),
        y_std=float(y_train.std(ddof=0)),
    )

    return ToyData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_train_std=standardiser.standardise_x(x_train),
        y_train_std=standardiser.standardise_y(y_train),
        x_test_std=standardiser.standardise_x(x_test),
        standardiser=standardiser,
        regions=region_masks(x_test, train_range),
        seed=seed,
        noise_sigma=noise_sigma,
    )


def check_matches_testyyy(seed: int = 42) -> str:
    """Assert this copy reproduces `ml_TESTYYY/src/data.py::make_toy_data`.

    Returns a SKIPPED message if that tree is not reachable — this folder is
    meant to stay runnable after the reference is moved or deleted, and the
    stored notebook outputs record the check passing at build time.
    """
    import sys

    if not (REFERENCE_REPO / "src" / "data.py").is_file():
        return (f"CHECK SKIPPED: reference tree not found at {REFERENCE_REPO} "
                f"(set ML_TESTYYY_REPO to point at it)")

    if str(REFERENCE_REPO) not in sys.path:
        sys.path.insert(0, str(REFERENCE_REPO))
    from src.data import make_toy_data as reference_make_toy_data

    live = reference_make_toy_data(seed)
    mine = make_toy_data(seed)
    for field in ("x_train", "y_train", "x_test", "y_test",
                  "x_train_std", "y_train_std", "x_test_std"):
        np.testing.assert_array_equal(getattr(live, field), getattr(mine, field),
                                      err_msg=f"{field} drifted from the reference")
    # Compare the standardiser field by field: dataclass `__eq__` returns
    # NotImplemented across two different classes, and the reference's
    # `Standardiser` is a different class object from this file's copy even
    # when every value in it agrees.
    for field in ("x_mean", "x_std", "y_mean", "y_std"):
        np.testing.assert_array_equal(
            getattr(live.standardiser, field), getattr(mine.standardiser, field),
            err_msg=f"standardiser.{field} drifted from the reference")
    assert live.noise_sigma == mine.noise_sigma
    for region in mine.regions:
        np.testing.assert_array_equal(live.regions[region], mine.regions[region],
                                      err_msg=f"regions[{region}] drifted")
    return (f"OK: make_toy_data({seed}) matches the reference on all 7 arrays "
            f"+ standardiser (reference: {REFERENCE_REPO})")
