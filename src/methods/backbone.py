"""Shared backbone for every NN-based method.

Homoscedastic single-head MLP: `DEFAULT_DEPTH` hidden layers of 50
`DEFAULT_ACTIVATION` units, plus one global learned scalar `log_sigma2`.

Single-head, rather than the two-head heteroscedastic network of the thesis's
Section 2.1: `laplace-torch`'s `sigma_noise` only accepts a scalar ("Only
homoscedastic output noise supported", baselaplace.py), and its `kron` Hessian
structure does not exist for subnetwork Laplace at all — so a per-point
variance head is not implementable with that library. `log_sigma2` plays the
same role as `WhiteKernel` does in the GP: one constant estimated from data,
with identical status across every network method (map, mcd, ensemble, bbb,
laplace).

The architecture is identical for every NN method by construction. The
comparison is between posterior approximations, not between networks.
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import prior_parametrisations
from src.methods import cache
from src.seeding import set_seed

# Numerical guard on `log_sigma2`, and NOTHING ELSE — it must never be the
# thing that determines a fitted noise level.
#
# A bound that binds is a noise-model hyperparameter masquerading as an
# overflow guard: it floors the aleatoric term of every method at once, and it
# can fabricate a convergence plateau (a flat validation-NLL tail that is the
# clamp, not the model). So if a fitted `sigma^2` is ever found sitting on
# either bound, the run is not reportable — widen the bound and re-run rather
# than quoting the number.
#
# exp(-12) = 6.1e-6, far below any plausible noise level on standardised
# targets, so the lower bound should be inactive in practice. Verify, do not
# assume: `HomoscedasticMLP` takes `logvar_clamp` so a diagnostic can ask
# whether the bound is binding without changing the constant every method
# shares.
LOGVAR_CLAMP = (-12.0, 6.0)

# float64 everywhere. These networks have a few hundred parameters and the
# datasets a few thousand rows, so the cost is negligible, and it removes an
# entire class of numerical questions from the comparison. The one that
# actually bites: the Laplace `full`/`kron` GGN posterior precision is badly
# conditioned whenever N approaches or falls below the parameter count, and
# float32's ~7 significant digits are not enough to invert it reliably.
#
# `set_seed()` sets this globally, so every model constructed after it picks it
# up automatically — including `bayesian_torch`'s `LinearReparameterization`,
# which has no `dtype` constructor argument at all (see bbb.py).
DTYPE = torch.float64

# Shared minibatch size across every NN method — a controlled variable, not one
# that varies per method. Batch size sets the number of optimizer steps per
# epoch (`ceil(N/batch_size)`), so a method-specific value would make the
# optimisation regime a second axis of variation alongside the posterior
# approximation, and the comparison would no longer isolate what it claims to.
#
# Minibatch rather than full-batch: deep ensembles (lakshminarayanan2017) draw
# part of their diversity from minibatch order, not only from initialisation,
# so full-batch would be a deviation from the method as published rather than
# just a slower version of it.
DEFAULT_BATCH_SIZE = 128

# Shared prior std across every NN method — must be identical everywhere by
# construction, so it lives here rather than as an independently-typed literal
# in five constructors. `src/config.py` translates it into each library's own
# parametrisation so the five cannot drift apart.
#
# **Do not tune this per method.** The thesis's claim is that all methods
# approximate the *same* posterior and differ only in how; gamma is part of
# that posterior's definition. Lowering it because one method's diagnostic plot
# looks better is choosing a different Bayesian model for that method's
# convenience, not a neutral fix. 1.0 is the standard unit Gaussian prior,
# chosen independently of how the results come out.
DEFAULT_GAMMA = 1.0

# Shared hidden-layer activation across every NN method.
#
# tanh rather than relu, for two reasons that hold independently of any
# measurement:
#   - Under the linearised Laplace predictive `f_var = J^T Sigma J`, a relu
#     network's Jacobian is a step function of which hidden units are active,
#     so `var_epistemic` is genuinely discontinuous at every activation-boundary
#     kink. tanh's continuous Jacobian removes that at the source rather than
#     smoothing it after the fact.
#   - It matches the architecture Foong et al. (2019) use in their 1D
#     experiments (single hidden layer, 50 tanh units), which is the setup this
#     project's synthetic problem is modelled on.
#
# Not a free choice, and the thesis should say so: gal2016 observes that a relu
# network's uncertainty keeps growing away from the data while a tanh network's
# stays bounded, because the two correspond to different covariance functions.
# Extrapolation bands for the sampling methods are therefore expected to look
# more conservative under tanh, and the activation is not a neutral
# implementation detail once uncertainty is the quantity of interest.
#
# "relu" stays available per method, for an activation ablation and for any run
# that must match gal2016's own relu-based protocol (see mcd.py).
DEFAULT_ACTIVATION = "tanh"

# Shared training length across every NN method (DEC-012).
#
# 4000 = 40 x 100, the budget of the reference run in
# yaringal/DropoutUncertaintyExps, whose published per-split results
# `scripts/fetch_data.py` fetches for E2 to validate against. Matching it
# removes one of the four differences between our MC dropout and the numbers in
# that repository's table; the other three (fixed `dropout_p` against their
# tuned grid, a gamma-derived penalty against their `tau`, and the
# moment-matched predictive of DEC-009 against their log-sum-exp mixture)
# remain, so the comparison is still "same order of magnitude", never equality.
#
# It is shared, not per method and not per stage: a training budget that varied
# between E1 and E2 would make the two chapters incomparable, and one that
# varied between methods would make the optimisation regime a second axis of
# variation alongside the posterior approximation.
#
# The previous value, 2000, was inherited and had no surviving justification
# (N-005) — so this is not a departure from a well-founded default, it is a
# replacement of an unfounded one by a citable one.
DEFAULT_EPOCHS = 4000

# Shared hidden-layer count across every NN method. This constant exists so
# that a depth ablation can vary depth explicitly per method instance, not
# because the default is expected to move. One hidden layer is the setting
# Foong et al. (2019) prove their mean-field result for, and the setting their
# 1D figures use, so it is the one that makes their prediction testable here.
#
# Note when reading a depth ablation: their universality result for >= 2 hidden
# layers is an EXISTENCE proof, so depth cannot be promised as a fix for
# mean-field VI's in-between uncertainty — that is a reason not to promise it,
# not a reason not to measure it.
DEFAULT_DEPTH = 1


class AlwaysOnDropout(nn.Module):
    """Dropout that samples regardless of `model.train()`/`model.eval()`.

    `torch.nn.Dropout` is a no-op under `model.eval()`, which is exactly the
    mode predictions are normally made in — with plain `nn.Dropout`, MC
    dropout sampling silently does not happen at predict time unless the
    caller remembers to leave the model in `.train()` mode. This module
    makes that impossible to forget: it always applies dropout when `p > 0`,
    independent of the module's training flag.
    """

    def __init__(self, p: float):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return x
        return F.dropout(x, p=self.p, training=True)


class MLP(nn.Module):
    """`input -> [Linear(hidden) -> activation -> Dropout] x depth -> Linear(1)`,
    activation tanh by default (see `DEFAULT_ACTIVATION`).

    Single mean head. `dropout_p=0.0` (the map/ensemble/laplace default)
    makes every dropout position identity, so this one module is the
    shared architecture across all five NN-based methods.

    **`depth`.** At `DEFAULT_DEPTH = 1` the `extra_hidden` list is empty and
    the forward pass reduces exactly to
    `input -> Linear(h) -> act -> Dropout -> Linear(1)`, on the same
    parameters in the same order — so `depth` can be introduced without
    perturbing the single-layer configuration every method shares.

    **Dropout placement at `depth > 1`: after EVERY hidden activation, not
    only the last.** This is standard multi-layer MC dropout; putting dropout
    only before the output layer would leave a `depth=2` network with the same
    single stochastic layer as `depth=1`, which would test nothing. It does
    mean MC dropout's total injected noise grows with depth — a property of
    the method being tested, not a confound introduced here, and a reason to
    report in-range fit quality alongside uncertainty shape in any depth
    sweep. Input->hidden dropout stays removed at every depth, for the reason
    below.

    **No input->hidden dropout by default** — a deliberate deviation from
    gal2016's literal protocol (dropout before *every* weighted layer). At
    `d=1` (the synthetic 1D problem) input dropout zeroes the network's *only*
    feature in `dropout_p` of forward passes, at both train and predict time
    (`AlwaysOnDropout` samples regardless of mode). During training this
    corrupts a fraction of batches to `x=0` outright, and the network
    compensates by inflating `log_sigma2` to cover the resulting loss spikes —
    the fitted noise level then reflects the dropout rate rather than the
    data-generating noise. Removed for all datasets rather than special-cased
    by dimensionality: the mechanism (dropped features absorbed into the noise
    estimate) is the same at any `d`, just proportionally weaker when other
    features survive the mask.

    **`input_dropout=True` puts it back.** Needed only to reproduce gal2016's
    own protocol, whose reference implementation applies dropout to the inputs
    before the first dense layer; omitting it would leave an unforced
    deviation standing if our MC dropout failed to match his published
    numbers. The `d=1` argument above does not transfer to the UCI sets, where
    a dropped feature leaves the rest of the input intact. The default stays
    `False`: `nn.Identity` consumes no RNG draws and holds no parameters, so
    disabling it is the same forward pass on the same weights in the same
    order.
    """

    def __init__(
        self, in_dim: int, hidden: int = 50, dropout_p: float = 0.0, always_on: bool = False,
        activation: str = DEFAULT_ACTIVATION, depth: int = DEFAULT_DEPTH,
        input_dropout: bool = False,
    ):
        super().__init__()
        if activation not in ("relu", "tanh"):
            raise ValueError(f"activation must be 'relu' or 'tanh', got {activation!r}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        drop_cls = AlwaysOnDropout if always_on else nn.Dropout
        self.depth = depth
        self.drop_input = drop_cls(dropout_p) if input_dropout else nn.Identity()
        self.linear1 = nn.Linear(in_dim, hidden, dtype=DTYPE)
        # `depth - 1` further hidden->hidden layers; empty at DEFAULT_DEPTH=1, which is
        # what makes depth=1 bit-identical to the pre-depth version of this class
        self.extra_hidden = nn.ModuleList(
            [nn.Linear(hidden, hidden, dtype=DTYPE) for _ in range(depth - 1)]
        )
        self.drop_hidden = drop_cls(dropout_p)
        self.mean_head = nn.Linear(hidden, 1, dtype=DTYPE)
        self.activation = activation
        self._activation_fn = torch.tanh if activation == "tanh" else torch.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop_hidden(self._activation_fn(self.linear1(self.drop_input(x))))
        for layer in self.extra_hidden:
            h = self.drop_hidden(self._activation_fn(layer(h)))
        return self.mean_head(h)


class HomoscedasticMLP(nn.Module):
    """`MLP` plus one global `log_sigma2`, clamped to `LOGVAR_CLAMP` = [-12, 6] (post-standardisation).

    `log_sigma2` is learned by default (`fixed_sigma2=None`). Passing
    `fixed_sigma2` registers it as a plain buffer instead of an
    `nn.Parameter` — no gradient, never touched by the optimizer — fixed to
    `log(fixed_sigma2)` for the model's lifetime. Only meaningful where the
    true noise level is known by construction, i.e. on synthetic data; it is
    what Foong et al. (2019) do in their 1D experiments. Fixing it on real
    (UCI) data would not be a fair comparison, since the true noise there is
    exactly what every method has to estimate.

    `logvar_clamp` defaults to the shared `LOGVAR_CLAMP` and exists so that a
    diagnostic can ask whether that bound is BINDING — whether the flat tail of
    a loss curve is the model converging or `sigma^2` sitting on the clamp
    floor — without changing the constant every method shares. Passing anything
    other than the default changes the model, so it must never be silently
    different between two methods being compared.
    """

    def __init__(
        self, in_dim: int, hidden: int = 50, dropout_p: float = 0.0, always_on: bool = False,
        activation: str = DEFAULT_ACTIVATION, fixed_sigma2: float = None, depth: int = DEFAULT_DEPTH,
        logvar_clamp: Tuple[float, float] = LOGVAR_CLAMP, input_dropout: bool = False,
    ):
        super().__init__()
        self.logvar_clamp = tuple(logvar_clamp)
        self.mlp = MLP(in_dim, hidden=hidden, dropout_p=dropout_p, always_on=always_on,
                       activation=activation, depth=depth, input_dropout=input_dropout)
        if fixed_sigma2 is None:
            self.log_sigma2 = nn.Parameter(torch.zeros((), dtype=DTYPE))
        else:
            if fixed_sigma2 <= 0:
                raise ValueError(f"fixed_sigma2 must be positive, got {fixed_sigma2}")
            self.register_buffer("log_sigma2", torch.tensor(math.log(fixed_sigma2), dtype=DTYPE))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mlp(x)
        log_var = self.log_sigma2.clamp(*self.logvar_clamp)
        return mu, log_var

    def prior_penalty(self, coefficient) -> torch.Tensor:
        """`coefficient * sum(theta**2)` over `self.mlp.parameters()` — weights
        and biases alike, since the prior is stated generically over theta with
        no weight/bias distinction.

        `coefficient` is a scalar (the flat `N(0, gamma^2 I)` prior, default) or
        a `{parameter_name: coefficient}` dict keyed by
        `self.mlp.named_parameters()`'s names (the layer-scaled prior —
        `config.layerwise_prior_sigma` computes the per-name values). Both paths
        only ever touch `self.mlp.parameters()`.

        **`log_sigma2` must never be regularised**: it is a noise-model
        parameter estimated from data, not a weight the prior speaks about.
        This is deliberately a method rather than a free function taking a
        parameter list, because that shape lets a call site pass
        `model.parameters()` by mistake and silently penalise `log_sigma2` too.
        Hard-coding `self.mlp` here removes the call site that could get it
        wrong: there is one way to compute this model's prior penalty.
        """
        if isinstance(coefficient, dict):
            return sum(coefficient[name] * p.pow(2).sum() for name, p in self.mlp.named_parameters())
        return coefficient * sum(p.pow(2).sum() for p in self.mlp.parameters())


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean-reduced Gaussian NLL, up to the constant 0.5*log(2*pi) term.

    `log_var` may be a scalar (homoscedastic) or per-point tensor; broadcasts
    either way.
    """
    var = torch.exp(log_var)
    return torch.mean(0.5 * log_var + 0.5 * (y - mu) ** 2 / var)


def train_homoscedastic_mlp(
    model_factory,
    X: "np.ndarray",
    y: "np.ndarray",
    seed: int,
    gamma: float,
    epochs: int,
    lr: float = 1e-2,
    batch_size: int = None,
    penalty_coefficient_override=None,
    use_cache: bool = True,
    cache_key_extra: dict = None,
    epoch_callback=None,
) -> HomoscedasticMLP:
    """Shared training loop for map / mcd / ensemble members.

    `model_factory` is a zero-arg callable that constructs a fresh
    `HomoscedasticMLP` — it must be called *after* `set_seed`, not before,
    since `nn.Linear` draws its initial weights at construction time. Taking
    a factory instead of an already-built model makes that ordering
    impossible to get backwards by accident.

    Adam, `weight_decay=0`, explicit `prior_penalty_coefficient * sum(theta**2)`
    term in the loss (recommended path — see `src/config.py`). `batch_size=None`
    trains full-batch; otherwise minibatches are reshuffled every epoch from
    the per-call `seed`, which is how deep-ensemble members get both a
    different initialisation and a different batch order (section 4.5) from
    a single base seed.

    `penalty_coefficient_override` (scalar or `{name: coefficient}` dict,
    see `HomoscedasticMLP.prior_penalty`): when given, replaces the
    `gamma`-derived coefficient entirely — `gamma` is still a required
    argument (so the ordinary call sites stay unchanged) but is then unused.
    Only for the layer-scaled prior variant, not the default training path.

    `epoch_callback(epoch, model)`, if given, is called after every epoch,
    purely to OBSERVE (it must not touch the model or the RNG). It exists so a
    diagnostic can read a whole training trajectory — e.g. the `log_sigma2`
    history, which the final model does not preserve — off one run instead of
    refitting once per epoch count. Because training is deterministic given the
    seed, one instrumented run of `E` epochs sees exactly the models that `E`
    separate runs of `1..E` epochs would produce. Passing a callback DISABLES
    the cache: a cache hit skips training and would silently produce no epochs
    at all.

    `use_cache`/`cache_key_extra` (see `src/methods/cache.py`): `model_factory`
    is an opaque closure, so whatever it bakes in beyond this function's own
    arguments (`hidden`, `activation`, `dropout_p`, `always_on`,
    `fixed_sigma2`) must be passed explicitly via `cache_key_extra` by the
    caller, or two different architectures trained with otherwise-identical
    arguments would collide in the cache. The cache stores only
    `model.state_dict()` — on a hit, `model_factory()` still runs (rebuilds
    the right architecture, including any deterministic prior buffers a
    caller like `BBBMethod` sets up) and the trained weights are loaded into
    it; the model's random initial values are irrelevant on a hit, since
    `load_state_dict` overwrites them.
    """
    n = X.shape[0]
    cache_path = None
    if epoch_callback is not None:
        use_cache = False
    if use_cache:
        config = dict(
            gamma=gamma, epochs=epochs, lr=lr, batch_size=batch_size,
            penalty_coefficient_override=penalty_coefficient_override,
            **(cache_key_extra or {}),
        )
        cache_path = cache.cache_path("homoscedastic_mlp", config, X, y, seed)
        cached = cache.load(cache_path)
        if cached is not None:
            model = model_factory()
            model.load_state_dict(cached["state_dict"])
            return model

    set_seed(seed)
    model = model_factory()
    if penalty_coefficient_override is not None:
        penalty_coef = penalty_coefficient_override
    else:
        penalty_coef = prior_parametrisations(gamma, n).prior_penalty_coefficient

    X_t = torch.as_tensor(X, dtype=DTYPE)
    y_t = torch.as_tensor(y, dtype=DTYPE).reshape(-1, 1)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    bs = n if batch_size is None else min(batch_size, n)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            opt.zero_grad()
            mu, log_var = model(X_t[idx])
            loss = gaussian_nll(mu, log_var, y_t[idx]) + model.prior_penalty(penalty_coef)
            loss.backward()
            opt.step()
        if epoch_callback is not None:
            epoch_callback(epoch + 1, model)

    if use_cache:
        cache.save(cache_path, {"state_dict": model.state_dict()})
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
