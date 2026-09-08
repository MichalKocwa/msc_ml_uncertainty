"""Exact Gaussian process — the reference point, not fitted to the shared backbone.

The only method here whose posterior is exact rather than approximate, which
is what makes it the reference the others are read against.

`normalize_y=False`: `y` is standardised externally (`src/data.py`), the same
way for every method, so GP and NN log-likelihoods are comparable.
`GaussianProcessRegressor`'s own `normalize_y` would double up on that and
silently break the comparison.

`var_epistemic` = `return_std=True` variance with the `WhiteKernel`'s
constant noise term subtracted back out. This works because, for a query
point x* that is not itself a training point, `WhiteKernel(x*, x*)`
contributes exactly `noise_level` to the diagonal of `kernel_(X*)` and
`WhiteKernel(x*, X_train) = 0` (distinct points) — so `noise_level` is the
*only* effect the white-noise term has on the returned predictive
variance. Subtracting it recovers `Var[f(x*)]`, i.e. epistemic-only.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from src.methods import cache
from src.methods.base import Prediction


class GPMethod:
    name = "gp"

    def __init__(self, n_restarts_optimizer: int = 5):
        self.n_restarts_optimizer = n_restarts_optimizer
        self.gp: GaussianProcessRegressor = None
        self.noise_level_: float = None
        self.n_parameters: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int, use_cache: bool = True) -> "GPMethod":
        """`use_cache` (src/methods/cache.py): unlike the NN methods, the
        whole fitted `GaussianProcessRegressor` is cached directly (via
        `torch.save`'s pickle fallback for non-tensor objects) rather than
        just a state dict — there is no separate "architecture" to rebuild it
        into, the object itself is the trained state. Matters at larger N,
        where the O(N^3) fit stops being instant.
        """
        cache_path = None
        if use_cache:
            config = dict(n_restarts_optimizer=self.n_restarts_optimizer)
            cache_path = cache.cache_path("gp", config, X, y, seed)
            cached = cache.load(cache_path)
            if cached is not None:
                self.gp = cached["gp"]
                self.noise_level_ = cached["noise_level_"]
                self.n_parameters = cached["n_parameters"]
                return self

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=seed,
        )
        self.gp.fit(X, y)
        # kernel_ == Sum(Product(ConstantKernel, RBF), WhiteKernel) after fit; k2 is the WhiteKernel summand.
        self.noise_level_ = float(self.gp.kernel_.k2.noise_level)
        self.n_parameters = len(self.gp.kernel_.theta)

        if use_cache:
            cache.save(cache_path, {"gp": self.gp, "noise_level_": self.noise_level_, "n_parameters": self.n_parameters})
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        mean, std = self.gp.predict(X, return_std=True)
        var_epistemic = np.clip(std ** 2 - self.noise_level_, 0.0, None)
        var_aleatoric = np.full_like(mean, self.noise_level_)
        return Prediction(mean=mean, var_aleatoric=var_aleatoric, var_epistemic=var_epistemic, samples=None)
