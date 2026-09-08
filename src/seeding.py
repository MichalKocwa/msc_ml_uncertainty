"""Single place to seed every source of randomness used across experiments.

Every experiment script must call `set_seed` before constructing data or
models, so that a reported result is reproducible from the seed alone
(hard rule in CLAUDE.md).

**The seed alone is not enough, and this is where that is fixed.** `torch`'s
intra-op thread count changes the order of float64 reductions. The
per-operation difference is at the rounding floor, but an optimiser running
tens of thousands of steps amplifies it, so the same seed under a different
thread count can give a measurably different network. `set_seed` therefore
pins the thread count as well, for the same reason it pins the default
dtype: it is the one function every method calls before building a model, so
it is the only place where this cannot be forgotten.

Escape hatch: set the environment variable `ML_UNCERTAINTY_TORCH_THREADS` to
run under a different count deliberately — only to measure the effect. A
normal run must leave it unset, and any result table should record the count
it ran under.
"""
import os
import random

import numpy as np
import torch

TORCH_THREADS_ENV = "ML_UNCERTAINTY_TORCH_THREADS"
DEFAULT_TORCH_THREADS = 1


def set_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs and pin CuDNN/algorithm determinism.

    Also sets the global default torch dtype to float64 (see
    src/methods/backbone.py's `DTYPE`): every NN method calls this before
    constructing its model, so
    layers built with no explicit `dtype` (`bayesian_torch`'s
    `LinearReparameterization`, used by BBB, has no `dtype` constructor
    argument at all) still come out float64. Must run *before* model
    construction for the same reason weight init must happen after
    `set_seed` — a layer already built in float32 would need a `.double()`
    cast, and casting after construction draws the initial random weights
    from a different point in the RNG stream than constructing directly in
    float64 — the two are not equivalent.

    `deterministic_algorithms=False` is an escape hatch for operations that
    have no deterministic GPU implementation; leave it on by default.
    """
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
