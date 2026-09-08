"""Monte Carlo dropout (gal2016).

`AlwaysOnDropout` before the hidden->mean layer only, not input->hidden — see
`src/methods/backbone.py`'s `MLP` docstring for why. `T` stochastic forward
passes at predict time give both the mean prediction and the aleatoric/
epistemic split, via the same estimator as BBB and deep ensembles (divisor
`T-1`, applied consistently across all three methods).
"""
import numpy as np
import torch

from src.config import layerwise_penalty_coefficients
from src.methods.backbone import (
    DEFAULT_ACTIVATION, DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_EPOCHS, DEFAULT_GAMMA, DTYPE, HomoscedasticMLP, count_parameters,
    train_homoscedastic_mlp,
)
from src.methods.base import Prediction
from src.seeding import set_seed


class MCDropoutMethod:
    name = "mcd"

    def __init__(
        self,
        hidden: int = 50,
        gamma: float = DEFAULT_GAMMA,
        dropout_p: float = 0.1,
        T: int = 100,  # number of stochastic passes — NOT justified by anything yet; a T-stability sweep is what would justify it
        epochs: int = DEFAULT_EPOCHS,
        lr: float = 1e-2,
        batch_size: int = DEFAULT_BATCH_SIZE,
        activation: str = DEFAULT_ACTIVATION,  # "tanh" (default) | "relu" — activation ablation, and required to
        # match gal2016's own relu-based protocol: pass activation="relu" explicitly at such a call site,
        # do not rely on this default
        fixed_sigma2: float = None,  # synthetic data only — see HomoscedasticMLP
        layerwise_prior_omega: float = None,  # layer-scaled prior variant — overrides gamma, see config.layerwise_prior_sigma
        depth: int = DEFAULT_DEPTH,  # depth ablation — see backbone.py's DEFAULT_DEPTH
        input_dropout: bool = False,  # gal2016-parity runs ONLY (he drops the input layer too) — see MLP's docstring
        penalty_override=None,  # gal2016-parity runs ONLY: scalar or {parameter_name: coefficient}, replaces the
        # gamma-derived penalty entirely (HomoscedasticMLP.prior_penalty). Exists because gal2016's L2 coefficient
        # `lengthscale^2 (1-p) / (2 N tau)` is a function of his (p, tau) grid, not of our prior — it cannot be
        # expressed through `gamma`. Every other call site leaves it None and keeps the flat N(0, gamma^2 I).
    ):
        self.hidden = hidden
        self.gamma = gamma
        self.dropout_p = dropout_p
        self.T = T
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation
        self.fixed_sigma2 = fixed_sigma2
        self.layerwise_prior_omega = layerwise_prior_omega
        self.depth = depth
        self.input_dropout = input_dropout
        self.penalty_override = penalty_override
        self.model: HomoscedasticMLP = None
        self.n_parameters: int = 0
        self._seed: int = None

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int, use_cache: bool = True,
            epoch_callback=None) -> "MCDropoutMethod":
        """`epoch_callback(epoch, model)` is forwarded to the training loop:
        observation only, and it disables the model cache. It lets a diagnostic
        read a whole epoch trajectory off one training run instead of refitting
        once per grid point.
        """

        self._seed = seed
        factory = lambda: HomoscedasticMLP(
            in_dim=X.shape[1], hidden=self.hidden, dropout_p=self.dropout_p, always_on=True,
            activation=self.activation, fixed_sigma2=self.fixed_sigma2, depth=self.depth,
            input_dropout=self.input_dropout,
        )
        penalty_override = self.penalty_override
        if self.layerwise_prior_omega is not None:
            probe = HomoscedasticMLP(in_dim=X.shape[1], hidden=self.hidden, activation=self.activation, depth=self.depth)
            penalty_override = layerwise_penalty_coefficients(
                probe.mlp.named_parameters(), self.layerwise_prior_omega, X.shape[0],
            )
        cache_key_extra = dict(
            depth=self.depth, hidden=self.hidden, activation=self.activation, dropout_p=self.dropout_p, always_on=True,
            fixed_sigma2=self.fixed_sigma2, layerwise_prior_omega=self.layerwise_prior_omega,
        )
        # Added to the key only when enabled, so that every model cached before
        # this flag existed still hits: `input_dropout=False` is the same
        # architecture it always was, and a new key would silently retrain all
        # of them (a cost, not a correctness problem, but an avoidable one).
        if self.input_dropout:
            cache_key_extra["input_dropout"] = True
        # (`penalty_override` needs no such guard: `train_homoscedastic_mlp`
        # already folds `penalty_coefficient_override` into its own cache key.)
        self.model = train_homoscedastic_mlp(
            factory, X, y, seed=seed, gamma=self.gamma, epochs=self.epochs, lr=self.lr,
            batch_size=self.batch_size, penalty_coefficient_override=penalty_override,
            use_cache=use_cache, cache_key_extra=cache_key_extra, epoch_callback=epoch_callback,
        )
        self.n_parameters = count_parameters(self.model)
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        """Seeded from `self._seed` (the `fit()` seed), not ambient RNG
        state — the `T` dropout masks below are sampled fresh on every call
        (`AlwaysOnDropout`), so without this, two `predict()` calls on the
        same fitted model would silently disagree, and a cache-hit `fit()`
        (`src/methods/cache.py`, which skips training and so does not
        consume RNG state the way a real training run would) would leave
        `predict()` reading whatever RNG state happened to be ambient —
        violating "every result reproducible from the seed alone" (hard rule,
        CLAUDE.md). Worth a regression test: fit twice with the same seed, once
        through the cache and once not, and assert the predictions match.
        """
        set_seed(self._seed)
        self.model.eval()  # irrelevant to AlwaysOnDropout, but keeps e.g. BatchNorm-free assumptions explicit
        X_t = torch.as_tensor(X, dtype=DTYPE)
        means = np.empty((self.T, X.shape[0]), dtype=np.float64)
        var_t = np.empty((self.T, X.shape[0]), dtype=np.float64)
        with torch.no_grad():
            for t in range(self.T):
                mu, log_var = self.model(X_t)
                means[t] = mu.numpy().ravel()
                var_t[t] = float(torch.exp(log_var))

        mean = means.mean(axis=0)
        var_aleatoric = var_t.mean(axis=0)
        var_epistemic = means.var(axis=0, ddof=1)
        return Prediction(mean=mean, var_aleatoric=var_aleatoric, var_epistemic=var_epistemic, samples=means)
