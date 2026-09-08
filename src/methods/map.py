"""Deterministic MAP baseline.

Not a sixth method — the lower reference point for epistemic uncertainty.
Same architecture as every other NN method, `dropout_p=0`, single
deterministic forward pass. `var_epistemic` is reported as an explicit
zero array, per the protocol's requirement that a method unable to
separate the two variance terms say so rather than folding everything into
`var_aleatoric`.
"""
import numpy as np
import torch

from src.config import layerwise_penalty_coefficients
from src.methods.backbone import (
    DEFAULT_ACTIVATION, DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_EPOCHS, DEFAULT_GAMMA, DTYPE, HomoscedasticMLP, count_parameters,
    train_homoscedastic_mlp,
)
from src.methods.base import Prediction


class MAPMethod:
    name = "map"

    def __init__(
        self, hidden: int = 50, gamma: float = DEFAULT_GAMMA, epochs: int = DEFAULT_EPOCHS, lr: float = 1e-2,
        batch_size: int = DEFAULT_BATCH_SIZE,
        activation: str = DEFAULT_ACTIVATION,  # "tanh" (default) | "relu" — activation ablation, see backbone.py's DEFAULT_ACTIVATION
        fixed_sigma2: float = None,  # synthetic data only — see HomoscedasticMLP
        layerwise_prior_omega: float = None,  # layer-scaled prior variant — overrides gamma, see config.layerwise_prior_sigma
        depth: int = DEFAULT_DEPTH,  # depth ablation — see backbone.py's DEFAULT_DEPTH
    ):
        self.hidden = hidden
        self.gamma = gamma
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation
        self.fixed_sigma2 = fixed_sigma2
        self.layerwise_prior_omega = layerwise_prior_omega
        self.depth = depth
        self.model: HomoscedasticMLP = None
        self.n_parameters: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int, use_cache: bool = True,
            epoch_callback=None) -> "MAPMethod":
        """`epoch_callback(epoch, model)` is forwarded to the training loop:
        observation only, and it disables the model cache. It lets a diagnostic
        read a whole epoch trajectory off one training run instead of refitting
        once per grid point.
        """

        factory = lambda: HomoscedasticMLP(
            in_dim=X.shape[1], hidden=self.hidden, dropout_p=0.0, activation=self.activation,
            fixed_sigma2=self.fixed_sigma2, depth=self.depth,
        )
        penalty_override = None
        if self.layerwise_prior_omega is not None:
            probe = HomoscedasticMLP(in_dim=X.shape[1], hidden=self.hidden, activation=self.activation, depth=self.depth)
            penalty_override = layerwise_penalty_coefficients(
                probe.mlp.named_parameters(), self.layerwise_prior_omega, X.shape[0],
            )
        cache_key_extra = dict(
            depth=self.depth, hidden=self.hidden, activation=self.activation, dropout_p=0.0, always_on=False,
            fixed_sigma2=self.fixed_sigma2, layerwise_prior_omega=self.layerwise_prior_omega,
        )
        self.model = train_homoscedastic_mlp(
            factory, X, y, seed=seed, gamma=self.gamma, epochs=self.epochs, lr=self.lr,
            batch_size=self.batch_size, penalty_coefficient_override=penalty_override,
            use_cache=use_cache, cache_key_extra=cache_key_extra, epoch_callback=epoch_callback,
        )
        self.n_parameters = count_parameters(self.model)
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=DTYPE)
        with torch.no_grad():
            mu, log_var = self.model(X_t)
        mean = mu.numpy().ravel()
        var_aleatoric = np.full_like(mean, float(torch.exp(log_var)))
        var_epistemic = np.zeros_like(mean)
        return Prediction(mean=mean, var_aleatoric=var_aleatoric, var_epistemic=var_epistemic, samples=None)
