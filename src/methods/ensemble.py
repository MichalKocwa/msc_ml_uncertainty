"""Deep ensembles (lakshminarayanan2017).

`M` independent members, each with its own weight initialisation and its
own minibatch order — both driven off a single per-member seed derived
from the base seed, so the whole ensemble is reproducible from one seed
(hard rule in CLAUDE.md) while members still differ from each other.
Same mean/epistemic estimator as MC dropout, with `T = M`. No adversarial
training — a deliberate, documented deviation from the paper's optional extra.
"""
import numpy as np
import torch

from src.config import layerwise_penalty_coefficients
from src.methods.backbone import (
    DEFAULT_ACTIVATION, DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_EPOCHS, DEFAULT_GAMMA, DTYPE, HomoscedasticMLP, count_parameters,
    train_homoscedastic_mlp,
)
from src.methods.base import Prediction


class DeepEnsembleMethod:
    name = "ensemble"

    def __init__(
        self,
        M: int = 5,
        hidden: int = 50,
        gamma: float = DEFAULT_GAMMA,
        epochs: int = DEFAULT_EPOCHS,
        lr: float = 1e-2,
        batch_size: int = DEFAULT_BATCH_SIZE,  # shared across every NN method — same optimisation regime as map/mcd/laplace/bbb, not a separate axis of variation
        activation: str = DEFAULT_ACTIVATION,  # "tanh" (default) | "relu" — activation ablation, see backbone.py's DEFAULT_ACTIVATION
        fixed_sigma2: float = None,  # synthetic data only — see HomoscedasticMLP
        layerwise_prior_omega: float = None,  # layer-scaled prior variant — overrides gamma, see config.layerwise_prior_sigma
        depth: int = DEFAULT_DEPTH,  # depth ablation — see backbone.py's DEFAULT_DEPTH
    ):
        self.M = M
        self.hidden = hidden
        self.gamma = gamma
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation
        self.fixed_sigma2 = fixed_sigma2
        self.layerwise_prior_omega = layerwise_prior_omega
        self.depth = depth
        self.models = []
        self.n_parameters: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int, use_cache: bool = True) -> "DeepEnsembleMethod":
        self.models = []
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
        for m in range(self.M):
            # deterministic per-member seed from a single base seed (init AND batch order both derive from it)
            member_seed = int(np.random.SeedSequence([seed, m]).generate_state(1)[0])
            factory = lambda: HomoscedasticMLP(
                in_dim=X.shape[1], hidden=self.hidden, dropout_p=0.0, activation=self.activation,
                fixed_sigma2=self.fixed_sigma2, depth=self.depth,
            )
            model = train_homoscedastic_mlp(
                factory, X, y, seed=member_seed, gamma=self.gamma,
                epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
                penalty_coefficient_override=penalty_override,
                use_cache=use_cache, cache_key_extra=cache_key_extra,
            )
            self.models.append(model)
        self.n_parameters = self.M * count_parameters(self.models[0])
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        X_t = torch.as_tensor(X, dtype=DTYPE)
        means = np.empty((self.M, X.shape[0]), dtype=np.float64)
        var_m = np.empty((self.M, X.shape[0]), dtype=np.float64)
        with torch.no_grad():
            for m, model in enumerate(self.models):
                model.eval()
                mu, log_var = model(X_t)
                means[m] = mu.numpy().ravel()
                var_m[m] = float(torch.exp(log_var))

        mean = means.mean(axis=0)
        var_aleatoric = var_m.mean(axis=0)
        var_epistemic = means.var(axis=0, ddof=1)
        return Prediction(mean=mean, var_aleatoric=var_aleatoric, var_epistemic=var_epistemic, samples=means)
