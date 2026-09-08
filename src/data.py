"""1D synthetic sine problem for E1.

`y = sin(x) + N(0, sigma^2)`, training inputs drawn uniformly from
`TRAIN_RANGE`, test inputs on a regular grid over `TEST_RANGE` that extends
beyond the training domain on both sides. The configuration is fixed by
DEC-002 (N = 250), DEC-004 (test range [-2, 8]), DEC-005 (sigma = 0.1) and
DEC-006 (500 grid points, fresh test noise per seed) — see
`docs/experiment_notes.md`. None of those values is a free parameter of this
module; the keyword arguments exist so a test can build a small fixture, not
so an experiment can quietly use a different setup.

**Standardisation.** Models are fitted on standardised X and y; predictions are
un-standardised before scoring, so every metric is in original y units
(DEC-003, the Hernandez-Lobato & Adams UCI convention). The statistics come
from the TRAINING set only — using test statistics would leak the
extrapolation region's spread into the transform and quietly flatter every
method at once.

**Seeding.** `set_seed` runs before any array is drawn, per CLAUDE.md's hard
rule. The same seed therefore fixes the training sample, the test noise and
(via each method's own `fit(..., seed)`) the model.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.seeding import set_seed

TRAIN_RANGE: Tuple[float, float] = (0.0, 6.0)
TEST_RANGE: Tuple[float, float] = (-2.0, 8.0)
N_TRAIN = 250
N_TEST = 500
NOISE_SIGMA = 0.1

# Region names, in the order they should appear in a table. `overall` is the
# whole test grid, i.e. the union of the other two — reported alongside them
# rather than instead of them, because a single figure over [-2, 8] averages
# interpolation and extrapolation together and hides exactly the behaviour E1
# is about (DEC-006).
IN_RANGE = "in_range"
EXTRAPOLATION = "extrapolation"
OVERALL = "overall"
REGIONS = (IN_RANGE, EXTRAPOLATION, OVERALL)


@dataclass(frozen=True)
class Standardiser:
    """Affine transform fitted on training statistics only.

    `unstandardise_var` scales by `y_std ** 2` rather than `y_std`: a variance
    is in squared y units. Getting this wrong would leave the predictive bands
    right and the log-likelihood wrong by a constant, which is precisely the
    kind of error a metrics table does not make visible.
    """
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
    """One seed's worth of the E1 problem, raw and standardised.

    `x_*` are `(n, 1)` — the shape every method's `predict` expects, since the
    shared backbone's first layer is `nn.Linear(in_dim=1, ...)` and sklearn's
    GP expects `(n_samples, n_features)`. `y_*` are `(n,)`.
    """
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
    """The noiseless target, `sin(x)`. Used by the figures for the dashed
    reference curve, and by nothing that computes a metric — metrics score
    against the noisy observations `y_test`, not against this.
    """
    return np.sin(x)


def region_masks(x: np.ndarray, train_range: Tuple[float, float] = TRAIN_RANGE) -> Dict[str, np.ndarray]:
    """Boolean masks over a flat `x`, keyed by `REGIONS`.

    `in_range` is the closed training interval and `extrapolation` its
    complement, so the two partition the grid exactly and `overall` is their
    union. At the E1 grid (`linspace(-2, 8, 500)`) no point falls on a
    boundary — the step is 10/499, and neither 0 nor 6 is reached at an
    integer index — so the closed/open choice does not move a single point.
    It is written closed anyway, because a later change of grid size must not
    silently drop a boundary point from both regions.
    """
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
    """Build one seed's training sample and test grid.

    Test inputs are a regular grid rather than a second uniform sample: the
    figures and the metrics read off the same array (DEC-007), and a regular
    grid makes the band shape legible at the edges, where a uniform sample
    would leave the extrapolation regions unevenly covered by chance. Test
    targets carry fresh noise per seed, so the log-likelihood is scored
    against observations rather than against the noiseless function.
    """
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
