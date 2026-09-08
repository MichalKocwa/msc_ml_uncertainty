"""Bayes by Backprop (blundell2015, via `bayesian-torch` / Intel Labs).

`dnn_to_bnn(model.mlp, ...)` replaces the mean network's `nn.Linear` layers
with `LinearReparameterization` in place. `log_sigma2` lives outside
`model.mlp` — it is a plain `nn.Parameter` on `HomoscedasticMLP`, never an
`nn.Module` — so `dnn_to_bnn`'s module-tree walk and `get_kl_loss`'s
`m.modules()` walk structurally cannot reach it. That is the intended
behaviour: the noise level is estimated from data, not given a variational
posterior. It is the same trap as the prior penalty
(`HomoscedasticMLP.prior_penalty`) in a different guise, a library-level KL
term instead of our own loss term, and `bayesian-torch`'s Module/Parameter
split avoids it by construction.

`LinearReparameterization.forward` samples fresh weights via the
reparameterisation trick on *every* call, in train and eval mode alike (there
is no `self.training` branch) — so unlike dropout, no `AlwaysOnDropout`-style
workaround is needed for predict-time sampling to actually happen.

**`bayesian-torch`'s `"prior_sigma"` key really is a standard deviation**,
despite being passed to `LinearReparameterization` as `prior_variance` and
stored as `sigma_p`. Confirmed by reading `kl_div`'s closed-form Gaussian KL,
where it appears as a standard deviation, not by trusting the parameter name.
So it maps directly onto `config.py`'s `prior_sigma = gamma`.

**float64 and `LinearReparameterization`.** Its `__init__` has no `dtype`
argument at all — the `torch.Tensor(...)` parameter allocations pick up
whatever `torch`'s *global* default dtype is at construction time. That is why
`set_seed()` sets `torch.set_default_dtype(torch.float64)` (see `backbone.py`'s
`DTYPE`) *before* `factory()` runs: this yields genuinely float64-initialised
`mu_weight`/`rho_weight` and sampling buffers, rather than a `.double()` cast
after float32 initialisation. The two are not equivalent — a post-hoc cast
draws the initial random values from a different point in the same seed's RNG
stream.

**What to expect, and what to watch for.** Mean-field VI is expected to
underestimate posterior spread: the KL direction favours mode-seeking
solutions, and the diagonal covariance ignores weight correlations. Foong et
al. (2019) sharpen this into a testable claim for exactly this architecture —
for a single-hidden-layer network with mean-field output weights, the
predictive variance is a convex function of the input, so it *cannot* express
increased uncertainty in between two separated regions of data. Narrow bands
here are therefore the method's documented failure mode surfacing, not
necessarily a bug in this wrapper.

A second, distinct mechanism can produce the same symptom and should not be
confused with it: "overpruning" (Trippe & Turner 2017), where the variational
variance of some units collapses towards zero to buy a cheap KL term, so those
units contribute no variance anywhere — in a gap or outside the data alike.
`frac_posterior_var_below_prior` below is a diagnostic for that, with the
important caveat documented on the function itself.

`fixed_sigma2`, `layerwise_prior_omega` and `elbo_samples` exist so that Foong
et al.'s own 1D configuration (fixed observation noise, layer-scaled prior,
many-sample ELBO, long training) can be reproduced and compared against the
defaults used here.
"""
import math

import numpy as np
import torch
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

from src.config import layerwise_prior_sigma, prior_parametrisations
from src.methods import cache
from src.methods.backbone import (
    DEFAULT_ACTIVATION, DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_EPOCHS, DEFAULT_GAMMA, DTYPE, HomoscedasticMLP, count_parameters,
    gaussian_nll,
)
from src.methods.base import Prediction
from src.seeding import set_seed


class BBBMethod:
    name = "bbb"

    def __init__(
        self,
        hidden: int = 50,
        gamma: float = DEFAULT_GAMMA,
        T: int = 100,  # number of predictive samples — NOT justified by anything yet; a T-stability sweep is what would justify it
        posterior_rho_init: float = -3.0,  # bayesian-torch's own default, inherited and not re-derived here
        epochs: int = DEFAULT_EPOCHS,
        lr: float = 1e-2,
        batch_size: int = DEFAULT_BATCH_SIZE,  # shared across every NN method — pi_i=1/M weighting handles M>1 correctly (see _train_bbb)
        activation: str = DEFAULT_ACTIVATION,  # "tanh" (default) | "relu" — activation ablation, see backbone.py's DEFAULT_ACTIVATION
        fixed_sigma2: float = None,  # synthetic data only — see HomoscedasticMLP
        layerwise_prior_omega: float = None,  # layer-scaled prior variant — overrides gamma (see _apply_layerwise_prior below); N(0,1) on biases, N(0,omega^2/fan_in) on weights, applied through bayesian-torch's own per-layer prior_weight_sigma/prior_bias_sigma buffers, not a reimplementation
        depth: int = DEFAULT_DEPTH,  # depth ablation — see backbone.py's DEFAULT_DEPTH
        elbo_samples: int = 1,  # averages the data term over K reparameterisation-trick samples per step instead of 1 (Blundell's ELBO is an expectation; more samples reduce the estimator's variance during training, at K x the forward/backward cost)
    ):
        self.hidden = hidden
        self.gamma = gamma
        self.T = T
        self.posterior_rho_init = posterior_rho_init
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation
        self.fixed_sigma2 = fixed_sigma2
        self.layerwise_prior_omega = layerwise_prior_omega
        self.depth = depth
        self.elbo_samples = elbo_samples
        self.model: HomoscedasticMLP = None
        self.n_parameters: int = 0
        self._seed: int = None

    def _bnn_prior_parameters(self) -> dict:
        prior_sigma = prior_parametrisations(self.gamma, 1).prior_sigma  # prior_sigma = gamma, independent of N
        return dict(
            prior_mu=0.0,
            prior_sigma=prior_sigma,
            posterior_mu_init=0.0,
            posterior_rho_init=self.posterior_rho_init,
            type="Reparameterization",  # plain reparameterisation, not Flipout — blundell2015 as published
            moped_enable=False,  # no MOPED empirical-Bayes initialisation; the prior is the shared N(0, gamma^2 I)
        )

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int, use_cache: bool = True,
            epoch_callback=None) -> "BBBMethod":
        """`epoch_callback(epoch, model)` is forwarded to the training loop:
        observation only, and it disables the model cache. It lets a diagnostic
        read a whole epoch trajectory off one training run instead of refitting
        once per grid point.
        """

        self._seed = seed
        bnn_prior_parameters = self._bnn_prior_parameters()

        def factory():
            model = HomoscedasticMLP(
                in_dim=X.shape[1], hidden=self.hidden, dropout_p=0.0, activation=self.activation,
                fixed_sigma2=self.fixed_sigma2, depth=self.depth,
            )
            dnn_to_bnn(model.mlp, bnn_prior_parameters)  # must run after set_seed (see backbone.py's model_factory note)
            if self.layerwise_prior_omega is not None:
                _apply_layerwise_prior(model.mlp, self.layerwise_prior_omega)
            return model

        cache_key_extra = dict(
            depth=self.depth, hidden=self.hidden, activation=self.activation, posterior_rho_init=self.posterior_rho_init,
            fixed_sigma2=self.fixed_sigma2, layerwise_prior_omega=self.layerwise_prior_omega,
            gamma=self.gamma,
        )
        self.model = _train_bbb(
            factory, X, y, seed=seed, epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
            elbo_samples=self.elbo_samples, use_cache=use_cache, cache_key_extra=cache_key_extra,
            epoch_callback=epoch_callback,
        )
        self.n_parameters = count_parameters(self.model)
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        """Seeded from `self._seed` (the `fit()` seed) — see
        `MCDropoutMethod.predict`'s docstring for why this matters:
        `LinearReparameterization.forward` resamples fresh weights on every
        call, so `predict()`'s `T` samples otherwise depend on ambient RNG
        state, not the seed alone.
        """
        set_seed(self._seed)
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


def frac_posterior_var_below_prior(model, threshold: float = 0.01) -> dict:
    """Fraction of variational parameters whose posterior variance is below
    `threshold` x their own prior variance.

    **Deliberately NOT called an "overpruning" fraction**, though that is the
    mechanism it is reached for (Trippe & Turner 2017: posterior variance driven
    to ~0 to buy a cheap KL term). It cannot measure that,
    because it cannot distinguish a variance that COLLAPSED from one that
    NEVER MOVED — see the initialisation note below. The name states what
    is computed; whether a given value means overpruning is an
    interpretation that has to be argued from the training regime, not read
    off this number.

    `sigma_post = log1p(exp(rho))` — `LinearReparameterization`'s own
    parameterisation, read off `rho_weight`/`rho_bias`; `sigma_prior` from
    the layer's `prior_weight_sigma`/`prior_bias_sigma` buffers, so the
    comparison stays correct under the layer-scaled prior too (where the
    prior is not one number across layers). Compared as variances, i.e.
    `sigma_post**2 < threshold * sigma_prior**2`.

    Returns both conventions, deliberately: `weights` (weight matrices only)
    and `all` (weights and biases). Which convention a published or previously
    quoted figure used is rarely stated, so reporting both is what makes a
    cross-check possible at all.

    **The metric reads 100% at initialisation, and cannot tell "collapsed by
    training" from "never moved".** At `posterior_rho_init = -3.0` the
    untrained posterior sigma is `log1p(exp(-3)) = 0.0486`, i.e. a variance
    of 0.236% of a `gamma = 1` prior's — already under the 1% threshold.
    Checked on an `epochs=0` fit at the shared defaults: the fraction is
    exactly 1.0 for both conventions. So a HIGH value means "the variational
    variances
    are at or below where they started", which at short epoch counts is
    mostly "training has barely moved them", not necessarily the
    Trippe & Turner pruning mechanism. Only a value well below 1.0 is
    unambiguous, and only differences at equal epoch counts are meaningful.

    `model` is a `HomoscedasticMLP` whose `.mlp` has already been converted
    by `dnn_to_bnn` (i.e. a fitted `BBBMethod.model`). Raises if it has not:
    a silently-zero fraction on a non-Bayesian network would look like a
    perfectly healthy posterior.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    mlp = model.mlp
    layers = [mlp.linear1, *mlp.extra_hidden, mlp.mean_head]
    if not all(hasattr(layer, "rho_weight") for layer in layers):
        raise TypeError(
            "expected a BBB model whose layers are LinearReparameterization "
            "(dnn_to_bnn already applied); got plain nn.Linear layers"
        )

    n_weights = n_weights_pruned = n_biases = n_biases_pruned = 0
    with torch.no_grad():
        for layer in layers:
            post_var = torch.log1p(torch.exp(layer.rho_weight)) ** 2
            prior_var = layer.prior_weight_sigma ** 2
            n_weights += post_var.numel()
            n_weights_pruned += int((post_var < threshold * prior_var).sum())
            if getattr(layer, "rho_bias", None) is not None:
                post_var_b = torch.log1p(torch.exp(layer.rho_bias)) ** 2
                prior_var_b = layer.prior_bias_sigma ** 2
                n_biases += post_var_b.numel()
                n_biases_pruned += int((post_var_b < threshold * prior_var_b).sum())

    n_all = n_weights + n_biases
    return dict(
        weights=n_weights_pruned / n_weights,
        all=(n_weights_pruned + n_biases_pruned) / n_all,
        n_weights=n_weights,
        n_all=n_all,
    )


def _apply_layerwise_prior(mlp, omega: float) -> None:
    """Overwrite `dnn_to_bnn`'s flat per-layer `prior_sigma` with the
    layer-scaled prior of Foong et al. (2019) — `N(0,1)` on biases, `N(0,omega^2/fan_in)`
    on weights, applied via `LinearReparameterization`'s own already-separate
    `prior_weight_sigma`/`prior_bias_sigma` buffers (confirmed by reading its
    source: `init_parameters` fills both from the single `prior_variance`
    passed to `dnn_to_bnn`, but the two buffers are independently addressable
    plain tensors, never learned — `.fill_()` on them is exactly what the
    library's own constructor does, not an undocumented internal). Must run
    *after* `dnn_to_bnn` has already converted `mlp`'s `nn.Linear` layers to
    `LinearReparameterization` in place.
    """
    for layer in (mlp.linear1, *mlp.extra_hidden, mlp.mean_head):  # extra_hidden is empty at depth=1
        weight_sigma = layerwise_prior_sigma(is_bias=False, fan_in=layer.in_features, omega=omega)
        layer.prior_weight_sigma.fill_(weight_sigma)
        if layer.prior_bias_sigma is not None:
            bias_sigma = layerwise_prior_sigma(is_bias=True, fan_in=layer.in_features, omega=omega)
            layer.prior_bias_sigma.fill_(bias_sigma)


def _train_bbb(
    model_factory, X, y, seed, epochs, lr, batch_size, elbo_samples: int = 1,
    use_cache: bool = True, cache_key_extra: dict = None, epoch_callback=None,
):
    """Adam, `weight_decay=0`. Minibatch ELBO with uniform KL weighting
    `pi_i = 1/M`: `loss_i = sum_nll(batch_i) + KL/M`,
    `M` = number of minibatches per epoch. `sum_nll` (not mean-reduced) is
    `gaussian_nll`'s mean-reduced output rescaled by batch size — Blundell's
    minibatch ELBO (eq. 8) is stated in terms of the per-batch *sum* of log
    likelihoods, and `pi_i` sums to 1 over the M minibatches in an epoch.

    `elbo_samples`: the data term is a
    Monte Carlo estimate of an expectation over the weight posterior
    (`LinearReparameterization.forward` resamples fresh weights via the
    reparameterisation trick on every call) — averaging over `elbo_samples`
    independent forward passes reduces that estimator's variance, at
    `elbo_samples`x the per-step forward/backward cost. `kl_term` is the
    closed-form Gaussian KL, not itself a Monte Carlo estimate, so it is
    computed once per step regardless of `elbo_samples`.

    `epoch_callback(epoch, model)` — same contract as
    `backbone.train_homoscedastic_mlp`'s: observation only, and it disables
    the cache. It may raise to stop training early; nothing here catches
    it, so the caller both decides when to stop and keeps whatever model
    reference the callback was handed.
    """
    n = X.shape[0]
    cache_path = None
    if epoch_callback is not None:
        use_cache = False
    if use_cache:
        config = dict(
            epochs=epochs, lr=lr, batch_size=batch_size, elbo_samples=elbo_samples,
            **(cache_key_extra or {}),
        )
        cache_path = cache.cache_path("bbb", config, X, y, seed)
        cached = cache.load(cache_path)
        if cached is not None:
            model = model_factory()
            model.load_state_dict(cached["state_dict"])
            return model

    set_seed(seed)
    model = model_factory()

    X_t = torch.as_tensor(X, dtype=DTYPE)
    y_t = torch.as_tensor(y, dtype=DTYPE).reshape(-1, 1)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    bs = n if batch_size is None else min(batch_size, n)
    n_batches = math.ceil(n / bs)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            opt.zero_grad()
            data_term = 0.0
            for _ in range(elbo_samples):
                mu, log_var = model(X_t[idx])
                data_term = data_term + gaussian_nll(mu, log_var, y_t[idx]) * len(idx)
            data_term = data_term / elbo_samples
            kl_term = get_kl_loss(model.mlp) / n_batches
            loss = data_term + kl_term
            loss.backward()
            opt.step()
        if epoch_callback is not None:
            epoch_callback(epoch + 1, model)

    if use_cache:
        cache.save(cache_path, {"state_dict": model.state_dict()})
    return model
