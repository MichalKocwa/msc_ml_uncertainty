"""Bayes by Backprop (Blundell et al. 2015) written from scratch in plain torch.

**Sandbox implementation.** This module exists only inside `bbb_experiments/`
and is not a replacement for `src/methods/bbb.py` (which wraps
`bayesian-torch`). It was written so that every knob of the method is visible
and changeable in one file, to find out *what* stops mean-field VI from
raising its uncertainty outside the training range on the E1 sine.

Model: `f(x) = W_L phi(... phi(W_1 x + b_1) ...) + b_L`, every weight and bias
with an independent Gaussian variational posterior `q(w) = N(mu, sigma^2)`,
`sigma = softplus(rho)`, and prior `p(w) = N(0, sigma_p^2)`. Observation model
`y ~ N(f(x), sigma_n^2)` with `sigma_n^2` either learned (a free `log_sigma2`
parameter, as in the shared backbone) or fixed (Foong et al. 2019's 1D
protocol).

Objective (negative ELBO, minimised):

    L = -E_q[ sum_i log N(y_i | f(x_i), sigma_n^2) ] + KL(q || p)

The expectation is a Monte Carlo estimate over `elbo_samples` reparameterised
draws. On a minibatch of size B the data term is rescaled by N/B so that its
expectation over batches equals the full-data sum, and the KL is added once
per step. This is the same objective as Blundell's `pi_i = 1/M` weighting up to
a global factor M, which Adam is invariant to.

Two ways to draw the data term:

- `local_reparam=False` — sample one full weight matrix per forward pass
  (Blundell et al. 2015, what `bayesian-torch`'s `LinearReparameterization`
  does). All points in the batch share the same weights.
- `local_reparam=True` — sample the *pre-activations* instead (Kingma,
  Salimans & Welling 2015, "local reparameterization trick"): for a layer with
  input `h`, `a ~ N(h mu^T + mu_b, h^2 sigma^2^T + sigma_b^2)`. Same marginal
  distribution of each activation, much lower gradient variance, so the
  variational sigmas actually get trained rather than sitting at their
  initialisation. Used at training time only; `predict` always samples whole
  weight sets so that the returned `samples` are genuine functions.

The KL is closed-form Gaussian-Gaussian, never sampled.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import Prediction
from src.seeding import set_seed

LOG_2PI = math.log(2.0 * math.pi)


class BayesLinear(nn.Module):
    """One mean-field Gaussian linear layer.

    `rho_init` sets the initial posterior std through `softplus(rho_init)`:
    -3 -> 0.049 (bayesian-torch's default), -1 -> 0.31, 0 -> 0.69. Whether the
    optimiser ever moves the sigmas far from where they start is exactly one
    of the questions this sandbox asks, so the initialisation is a knob here
    rather than a constant.

    `mu` is initialised like `nn.Linear` (Kaiming-uniform on weights, uniform
    on biases) so a `rho -> -inf` network would be an ordinary MLP at init.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma_w: float,
                 prior_sigma_b: float, rho_init: float, local_reparam: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.local_reparam = local_reparam
        self.register_buffer("prior_sigma_w", torch.tensor(float(prior_sigma_w)))
        self.register_buffer("prior_sigma_b", torch.tensor(float(prior_sigma_b)))

        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.rho_w = nn.Parameter(torch.full((out_features, in_features), float(rho_init)))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.rho_b = nn.Parameter(torch.full((out_features,), float(rho_init)))

        # same init as nn.Linear.reset_parameters, so the mean network starts
        # where a deterministic MLP would
        nn.init.kaiming_uniform_(self.mu_w, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.mu_b, -bound, bound)

    @property
    def sigma_w(self) -> torch.Tensor:
        return F.softplus(self.rho_w)

    @property
    def sigma_b(self) -> torch.Tensor:
        return F.softplus(self.rho_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_reparam and self.training:
            mean = F.linear(x, self.mu_w, self.mu_b)
            var = F.linear(x ** 2, self.sigma_w ** 2, self.sigma_b ** 2)
            return mean + torch.sqrt(var + 1e-16) * torch.randn_like(mean)
        w = self.mu_w + self.sigma_w * torch.randn_like(self.mu_w)
        b = self.mu_b + self.sigma_b * torch.randn_like(self.mu_b)
        return F.linear(x, w, b)

    def kl(self) -> torch.Tensor:
        return _gaussian_kl(self.mu_w, self.sigma_w, self.prior_sigma_w) + \
            _gaussian_kl(self.mu_b, self.sigma_b, self.prior_sigma_b)


def _gaussian_kl(mu: torch.Tensor, sigma: torch.Tensor, prior_sigma: torch.Tensor) -> torch.Tensor:
    """`sum KL( N(mu, sigma^2) || N(0, prior_sigma^2) )` over every element."""
    return torch.sum(
        torch.log(prior_sigma / sigma) + (sigma ** 2 + mu ** 2) / (2.0 * prior_sigma ** 2) - 0.5
    )


_ACTIVATIONS = {"tanh": torch.tanh, "relu": torch.relu}


class BBBNet(nn.Module):
    """`in_dim -> [BayesLinear(hidden) -> act] x depth -> BayesLinear(1)` plus a
    global observation-noise `log_sigma2` (learned parameter, or fixed buffer).

    `prior_sigma` is either a single float (flat `N(0, gamma^2 I)` on every
    parameter — the project's shared prior) or the string `"layerwise"`, which
    applies Foong et al. (2019)'s scaling `N(0, omega^2 / fan_in)` on weights
    and `N(0, 1)` on biases with `omega = layerwise_omega`.
    """

    def __init__(self, in_dim: int, hidden: int, depth: int, activation: str,
                 prior_sigma, rho_init: float, local_reparam: bool,
                 fixed_sigma2: Optional[float], layerwise_omega: float = 4.0):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS)}, got {activation!r}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self._act = _ACTIVATIONS[activation]

        widths = [in_dim] + [hidden] * depth + [1]
        layers = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            if prior_sigma == "layerwise":
                sw, sb = layerwise_omega / math.sqrt(fan_in), 1.0
            else:
                sw = sb = float(prior_sigma)
            layers.append(BayesLinear(fan_in, fan_out, sw, sb, rho_init, local_reparam))
        self.layers = nn.ModuleList(layers)

        if fixed_sigma2 is None:
            self.log_sigma2 = nn.Parameter(torch.zeros(()))
        else:
            if fixed_sigma2 <= 0:
                raise ValueError(f"fixed_sigma2 must be positive, got {fixed_sigma2}")
            self.register_buffer("log_sigma2", torch.tensor(math.log(fixed_sigma2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers[:-1]:
            h = self._act(layer(h))
        return self.layers[-1](h)

    def kl(self) -> torch.Tensor:
        return sum(layer.kl() for layer in self.layers)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def posterior_sigma_summary(self) -> Dict[str, float]:
        """Mean posterior std per layer (weights), plus the fraction of
        variational parameters whose variance is below 1% of their prior's —
        the same diagnostic as `src.methods.bbb.frac_posterior_var_below_prior`,
        with the same caveat: it reads 100% at `rho_init = -3` before any
        training, so only movement away from 1.0 is informative.
        """
        out = {}
        n_all = n_below = 0
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                out[f"mean_sigma_w_layer{i}"] = float(layer.sigma_w.mean())
                for sigma, prior in ((layer.sigma_w, layer.prior_sigma_w), (layer.sigma_b, layer.prior_sigma_b)):
                    n_all += sigma.numel()
                    n_below += int((sigma ** 2 < 0.01 * prior ** 2).sum())
        out["frac_var_below_1pct_prior"] = n_below / n_all
        return out


@dataclass
class BBBConfig:
    """Everything that defines one run. Defaults reproduce the shared backbone
    setting of `src/methods/bbb.py` as closely as this implementation allows
    (tanh, 50 hidden, gamma = 1, rho_init = -3, minibatch 128, Adam 1e-2,
    4000 epochs, one ELBO sample, no local reparameterisation, learned noise).
    """
    name: str = "repo_like"
    hidden: int = 50
    depth: int = 1
    activation: str = "tanh"
    prior_sigma: object = 1.0          # float, or "layerwise"
    layerwise_omega: float = 4.0       # only used with prior_sigma == "layerwise"; Foong et al. 2019's 1D value
    rho_init: float = -3.0
    local_reparam: bool = False
    fixed_sigma2: Optional[float] = None  # in STANDARDISED y units; None = learn it
    elbo_samples: int = 1
    epochs: int = 4000
    lr: float = 1e-2
    batch_size: Optional[int] = 128    # None = full batch
    kl_warmup_frac: float = 0.0        # fraction of epochs over which the KL weight ramps 0 -> 1; 0 = off
    T: int = 100                       # predictive samples
    notes: str = ""

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class TrainingTrace:
    epoch: List[int] = field(default_factory=list)
    neg_elbo: List[float] = field(default_factory=list)
    nll: List[float] = field(default_factory=list)
    kl: List[float] = field(default_factory=list)
    sigma_n: List[float] = field(default_factory=list)


class BBBRegressor:
    """Fit/predict wrapper with the `src.methods.base.UncertaintyMethod` shape."""

    name = "bbb_sandbox"

    def __init__(self, config: BBBConfig):
        self.config = config
        self.model: Optional[BBBNet] = None
        self.n_parameters = 0
        self.trace = TrainingTrace()
        self._seed: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int, log_every: int = 0) -> "BBBRegressor":
        c = self.config
        self._seed = seed
        set_seed(seed)  # before construction — CLAUDE.md hard rule; also sets float64 + 1 thread
        self.model = BBBNet(
            in_dim=X.shape[1], hidden=c.hidden, depth=c.depth, activation=c.activation,
            prior_sigma=c.prior_sigma, rho_init=c.rho_init, local_reparam=c.local_reparam,
            fixed_sigma2=c.fixed_sigma2, layerwise_omega=c.layerwise_omega,
        )
        self.n_parameters = self.model.n_parameters()

        n = X.shape[0]
        X_t = torch.as_tensor(X, dtype=torch.float64)
        y_t = torch.as_tensor(y, dtype=torch.float64).reshape(-1, 1)
        opt = torch.optim.Adam(self.model.parameters(), lr=c.lr, weight_decay=0.0)
        generator = torch.Generator().manual_seed(seed)
        bs = n if c.batch_size is None else min(c.batch_size, n)
        warmup_epochs = int(round(c.kl_warmup_frac * c.epochs))

        self.model.train()
        for epoch in range(1, c.epochs + 1):
            kl_weight = 1.0 if warmup_epochs == 0 else min(1.0, epoch / warmup_epochs)
            perm = torch.randperm(n, generator=generator)
            epoch_nll = epoch_kl = 0.0
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb, yb = X_t[idx], y_t[idx]
                opt.zero_grad()
                log_var = self.model.log_sigma2
                nll = 0.0
                for _ in range(c.elbo_samples):
                    mu = self.model(xb)
                    nll = nll + torch.sum(0.5 * LOG_2PI + 0.5 * log_var + 0.5 * (yb - mu) ** 2 / torch.exp(log_var))
                nll = nll / c.elbo_samples * (n / len(idx))
                kl = self.model.kl()
                loss = nll + kl_weight * kl
                loss.backward()
                opt.step()
                epoch_nll += float(nll.detach()) * len(idx) / n
                epoch_kl = float(kl.detach())
            if log_every and (epoch % log_every == 0 or epoch == 1 or epoch == c.epochs):
                self.trace.epoch.append(epoch)
                self.trace.nll.append(epoch_nll)
                self.trace.kl.append(epoch_kl)
                self.trace.neg_elbo.append(epoch_nll + epoch_kl)
                self.trace.sigma_n.append(float(torch.exp(0.5 * self.model.log_sigma2)))
        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        """`T` full weight draws (never the local trick, so each row of
        `samples` is one function). Reseeded from the fit seed so the draws
        depend on the seed alone, as `src/methods/bbb.py` does.
        """
        c = self.config
        set_seed(self._seed)
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float64)
        means = np.empty((c.T, X.shape[0]))
        with torch.no_grad():
            for t in range(c.T):
                means[t] = self.model(X_t).numpy().ravel()
            sigma2 = float(torch.exp(self.model.log_sigma2))
        return Prediction(
            mean=means.mean(axis=0),
            var_aleatoric=np.full(X.shape[0], sigma2),
            var_epistemic=means.var(axis=0, ddof=1),
            samples=means,
        )

    def diagnostics(self) -> Dict[str, float]:
        d = self.model.posterior_sigma_summary()
        with torch.no_grad():
            d["kl_final"] = float(self.model.kl())
            d["sigma_n_std_units"] = float(torch.exp(0.5 * self.model.log_sigma2))
        return d
