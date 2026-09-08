"""The architecture grid. THE ONLY place in this folder that names a value.

Nothing outside `arch_sweep/` imports this module, and this module imports
nothing from `arch_sweep/`. That is the whole isolation contract: a number
written here can reach a sweep row and a sweep figure, and nowhere else. In
particular it can never become a repo default -- the methods are still
constructed from `src/methods/`, and every value below is passed in as an
explicit keyword argument at the call site, exactly the way `--set` does it in
`experiments/e1_toy.py` (DEC-014).

`depth`, `hidden` and `activation` are the three architecture knobs the shared
backbone exposes (`src/methods/backbone.py`). Everything else -- gamma, the
prior, the epoch budget, the batch size, dropout_p, T, M -- stays at the repo
default, because varying them would make this a different experiment.
"""

DEPTHS = (1, 2, 3)
# Width is fixed at 50, the repo default. `hidden=100` was dropped from this
# sweep after measuring what it costs the one method that cannot absorb it:
# `laplace` runs `subset_of_weights="all"` with `hessian_structure="full"`, so
# it materialises a dense P x P Hessian and factorises it. At depth 3, width 100
# that is 20,502 parameters -- a 3.13 GB float64 Hessian and an O(P^3)
# factorisation that `set_seed`'s deliberate `torch.set_num_threads(1)` cannot
# parallelise. A probe run at 20 epochs produced no output in over ten minutes.
# Width therefore stays out of this sweep, and the finding itself belongs in the
# cost discussion: full-network Laplace is the method whose scaling forbids the
# wider architectures, not a method that merely does worse on them.
HIDDEN = (50,)
ACTIVATIONS = ("tanh", "relu")

# Methods whose architecture this sweep can vary: everything built on the shared
# backbone. `gp` is deliberately absent -- it has no depth, width or activation.
NN_METHODS = ("map", "mcd", "ensemble", "bbb", "laplace")

# Run once per seed at its own defaults and drawn into every comparison figure.
# Without it the panels show only how the networks differ from each other, not
# how far any of them is from the one exact posterior in the comparison.
REFERENCE_METHODS = ("gp",)

# Seeds. Stage 1 is a single seed: 12 configurations x 5 methods is already 60
# fits, and its purpose is to show the SHAPE of each architecture's predictive
# band. Stage 2 re-runs the shortlist over METRIC_SEEDS so the numbers that
# reach the thesis carry a spread.
FIGURE_SEED = 0            # matches experiments/e1_toy.py's FIGURE_SEED
METRIC_SEEDS = tuple(range(20))   # matches its DEFAULT_SEEDS


def grid():
    """The configurations, in a stable order. `config_id` keys every output."""
    out = []
    for depth in DEPTHS:
        for hidden in HIDDEN:
            for activation in ACTIVATIONS:
                out.append({
                    "config_id": f"d{depth}_h{hidden}_{activation}",
                    "depth": depth,
                    "hidden": hidden,
                    "activation": activation,
                })
    return out


def baseline_config_id() -> str:
    """The repo default (`DEFAULT_DEPTH=1`, `hidden=50`, `DEFAULT_ACTIVATION`).

    Named so a reader of the results can find the row that reproduces E1 and
    check this sweep against it before believing anything else in the table.
    """
    return "d1_h50_tanh"
