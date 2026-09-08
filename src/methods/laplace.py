"""Laplace approximation (`laplace-torch`, Daxberger et al. 2021).

Full-network Laplace (`subset_of_weights="all"`) around the same MAP model
`MAPMethod` trains. Subnetwork Laplace is not usable here: `sigma_noise` only
accepts a scalar, and the `kron` Hessian structure does not exist for it at
all. Both restrictions disappear in full-network mode with the homoscedastic
backbone.

Note this departs from Daxberger et al.'s own recommended default, which is
*last-layer* KFAC with a marginal-likelihood-tuned prior precision. Full-network
is used instead because every other method in this comparison places a
distribution over all weights, and a last-layer variant would be answering a
different question. The networks here have a few hundred parameters, so the
full curvature matrix is affordable.

Predictive mean is `la()`'s own linearised `f_mu`, not a separately computed
forward pass; for the linearised (GLM) predictive these coincide, since the
expansion is taken at the MAP weights. `var_aleatoric` is read from the exact
`log_sigma2` learned during MAP training rather than round-tripped through
`sigma_noise`'s sqrt/square, so it is bit-identical to `MAPMethod`'s. The
consequence is worth stating in the text: `laplace` and `map` differ **only**
in `var_epistemic`.

**GLM predictive, not Monte Carlo sampling from the posterior.** Daxberger et
al. are explicit that Monte Carlo integration performs poorly for Laplace
approximations built on GGN or Fisher curvature — the sampled predictive is
inconsistent with the curvature approximation, and the linearised predictive
removes that mismatch. `predict()` therefore uses `pred_type="glm"` and no
sampling, which is also why `Prediction.samples` is `None` for this method.

**Does `optimize_prior_precision(method="marglik")` also tune `sigma_noise`?**
No — it makes `log_prior_precision` differentiable and optimises that alone.
The module-level `laplace.marglik_training` helper does tune both jointly, but
it retrains a network from scratch, so it would not share weights with
`MAPMethod` and the "differs only in var_epistemic" property above would be
lost. It is not used here.

**`prior_precision_mode="fixed"` is the default, not `"marglik"`.** Marglik
tuning picks whatever prior precision maximises the marginal likelihood, which
is not `1/gamma**2` for the shared `gamma`. That would leave Laplace
approximating a posterior under a *different* prior than BBB/MAP/ensemble, so
the comparison would stop being "different approximations of the same
posterior" and become "different models". `"fixed"` sets
`prior_precision = prior_parametrisations(gamma, n).prior_precision` and never
calls `optimize_prior_precision` — the same gamma as everywhere else by
construction, not by coincidence.

`"marglik"` and `"unregularised"` are kept as explicit, opt-in configurations
for a prior-precision ablation. Both are numerically fragile at the low end:
an exactly-zero prior precision leaves the posterior precision singular, and
small values can make it indefinite outright (Cholesky fails).

**Two known sources of structure in `var_epistemic`, both worth checking before
trusting a plot.**

1. *Rank deficiency.* When N is at or below the network's parameter count, the
   GGN Hessian is rank-deficient (rank <= N), so in `full`/`kron` mode much of
   parameter space is purely prior-determined and the posterior precision is
   badly conditioned. A `full`/`kron` linearised-variance evaluation can hit
   that ill-conditioning at specific inputs, producing narrow, grid-density-
   independent spikes well inside the training range. Keeping N above the
   parameter count removes the null space that causes them. This is a property
   of the method at small N, not a bug to be patched.
2. *Activation kinks.* The linearised predictive `f_var(x) = J(x)^T Sigma J(x)`
   is genuinely discontinuous wherever `J(x)` is, which under ReLU is every
   activation boundary. This is why `DEFAULT_ACTIVATION` is `"tanh"` (see
   `backbone.py`); passing `activation="relu"` reproduces the discontinuity by
   design, which is what makes it an ablation rather than a defect.

**Environment note:** this `laplace-torch` commit's `kron` backend calls
`KFACLinearOperator._compute_kfac()`, a method renamed to
`compute_kronecker_factors()` in `curvlinops-for-pytorch>=3.0`. `laplace-torch`
pins no upper bound, so a plain install pulls the incompatible 3.x; hence the
`curvlinops-for-pytorch==2.0.1` pin in `requirements.txt`.
"""
import numpy as np
import torch
from laplace import Laplace
from torch.utils.data import DataLoader, TensorDataset

from src.config import layerwise_penalty_coefficients, layerwise_prior_precisions, prior_parametrisations
from src.methods.backbone import (
    DEFAULT_ACTIVATION, DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_EPOCHS, DEFAULT_GAMMA, DTYPE, LOGVAR_CLAMP, HomoscedasticMLP,
    count_parameters, train_homoscedastic_mlp,
)
from src.methods.base import Prediction

# Prior precision for the "unregularised" ablation: near zero, rather than
# equal to 1/gamma**2 or tuned by marginal likelihood. Not exactly zero, which
# would leave the posterior precision singular; small enough that the resulting
# var_epistemic is deliberately large, which is the ablation's whole point.
# The exact value is arbitrary and has to stay large enough for the Cholesky
# factorisation to succeed — if it fails, raise it and say so, rather than
# quietly picking the smallest value that happens to work.
UNREGULARISED_PRIOR_PRECISION = 1e-3


class LaplaceMethod:
    name = "laplace"

    def __init__(
        self,
        hidden: int = 50,
        gamma: float = DEFAULT_GAMMA,
        epochs: int = DEFAULT_EPOCHS,
        lr: float = 1e-2,
        hessian_structure: str = "full",  # "full" | "kron" | "diag" — the covariance-structure ablation
        prior_precision_mode: str = "fixed",  # "fixed" (default: prior_precision = 1/gamma**2, matches every other method) | "marglik" | "unregularised" (both ablation-only)
        marglik_steps: int = 100,
        batch_size: int = DEFAULT_BATCH_SIZE,  # for the MAP backbone's training only — see fit(); la.fit()'s own loader stays full-batch, it accumulates curvature, not gradient steps
        activation: str = DEFAULT_ACTIVATION,  # "tanh" (default) | "relu" — activation ablation; tanh's continuous Jacobian is why this method's var_epistemic is continuous under the default (see module docstring)
        fixed_sigma2: float = None,  # synthetic data only — see HomoscedasticMLP
        layerwise_prior_omega: float = None,  # layer-scaled prior variant — overrides gamma for BOTH the MAP backbone's training penalty and `Laplace(prior_precision=...)`, see config.layerwise_prior_sigma
        depth: int = DEFAULT_DEPTH,  # depth ablation — see backbone.py's DEFAULT_DEPTH
    ):
        if hessian_structure not in ("full", "kron", "diag"):
            raise ValueError(f"hessian_structure must be 'full', 'kron' or 'diag', got {hessian_structure!r}")
        if prior_precision_mode not in ("fixed", "marglik", "unregularised"):
            raise ValueError(
                f"prior_precision_mode must be 'fixed', 'marglik' or 'unregularised', got {prior_precision_mode!r}"
            )
        self.hidden = hidden
        self.gamma = gamma
        self.epochs = epochs
        self.lr = lr
        self.hessian_structure = hessian_structure
        self.prior_precision_mode = prior_precision_mode
        self.marglik_steps = marglik_steps
        self.batch_size = batch_size
        self.activation = activation
        self.fixed_sigma2 = fixed_sigma2
        self.layerwise_prior_omega = layerwise_prior_omega
        self.depth = depth

        self.la: Laplace = None
        self._noise_var: torch.Tensor = None  # exp(log_sigma2), reused verbatim for var_aleatoric
        self.n_parameters: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int, use_cache: bool = True) -> "LaplaceMethod":
        """`use_cache` covers only the MAP backbone (the expensive, 2000-epoch
        part) via `train_homoscedastic_mlp`'s own cache — the subsequent
        `la.fit()` below is cheap (curvature only, no gradient descent) and
        is not itself cached, to avoid depending on `laplace-torch`'s
        `Laplace` object being safely serialisable (`src/methods/cache.py`).
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
        map_model = train_homoscedastic_mlp(
            factory, X, y, seed=seed, gamma=self.gamma, epochs=self.epochs, lr=self.lr,
            batch_size=self.batch_size, penalty_coefficient_override=penalty_override,
            use_cache=use_cache, cache_key_extra=cache_key_extra,
        )
        map_model.eval()
        mean_net = map_model.mlp
        log_var = map_model.log_sigma2.clamp(*LOGVAR_CLAMP).detach()
        self._noise_var = torch.exp(log_var)
        sigma_noise = self._noise_var.sqrt()

        n = X.shape[0]
        if self.layerwise_prior_omega is not None:
            # order must match mean_net.parameters()'s own iteration order — Laplace's `prior_precision`
            # vector is positional (per parameter GROUP), not name-keyed
            precisions = layerwise_prior_precisions(mean_net.named_parameters(), self.layerwise_prior_omega)
            matched_prior_precision = torch.tensor(
                [precisions[name] for name, _ in mean_net.named_parameters()], dtype=DTYPE,
            )
        else:
            matched_prior_precision = prior_parametrisations(self.gamma, n).prior_precision  # = 1/gamma**2, same gamma as every other method
        if self.prior_precision_mode == "unregularised":
            prior_precision_init = UNREGULARISED_PRIOR_PRECISION
        else:  # "fixed" or "marglik" (marglik's value is the optimizer's starting point)
            prior_precision_init = matched_prior_precision

        self.la = Laplace(
            mean_net,
            "regression",
            subset_of_weights="all",
            hessian_structure=self.hessian_structure,
            sigma_noise=sigma_noise,
            prior_precision=prior_precision_init,
        )

        X_t = torch.as_tensor(X, dtype=DTYPE)
        y_t = torch.as_tensor(y, dtype=DTYPE).reshape(-1, 1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=n)
        self.la.fit(loader)

        if self.prior_precision_mode == "marglik":
            self.la.optimize_prior_precision(pred_type="glm", method="marglik", n_steps=self.marglik_steps)
        # "fixed" and "unregularised": prior_precision is never optimised away from what was set above

        self.n_parameters = count_parameters(map_model)
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        X_t = torch.as_tensor(X, dtype=DTYPE)
        f_mu, f_var = self.la(X_t, pred_type="glm", diagonal_output=True)

        mean = f_mu.detach().numpy().ravel()
        var_epistemic = np.clip(f_var.detach().numpy().ravel(), 0.0, None)
        var_aleatoric = np.full_like(mean, float(self._noise_var))
        return Prediction(mean=mean, var_aleatoric=var_aleatoric, var_epistemic=var_epistemic, samples=None)
