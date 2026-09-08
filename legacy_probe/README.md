# `legacy_probe/` — the pre-rewrite methods, on two different sine setups

Throwaway diagnostic folder, parked in `ml_TESTYYY/` so that the
`ml_uncertainty_msc` repository stays untouched. It is self-contained: every
module it needs is vendored here, and deleting the whole directory breaks
nothing anywhere.

It answers one question: **how much of the "the old plots looked better"
impression is the old METHODS, and how much is the SETUP they ran on?** The
method code is held fixed across both notebooks; only the data and the way it
is fed in change.

Nothing here is a thesis result.

## Contents

| file | what it is |
| --- | --- |
| `legacy_models.py` | **verbatim** `git show 28b7615^:src/models.py` from `ml_uncertainty_msc` (MLP with mean+logvar heads, `train_mlp`, `make_gp`) |
| `legacy_methods.py` | **reconstructed from bytecode** — see the warning below |
| `legacy_plotting.py` | verbatim `plot_uncertainty_for_1d_data` from `28b7615^:src/plotting.py`, plus an optional `out_path` |
| `legacy_data.py` | local copy of `ml_uncertainty_msc`'s `_make_sin_dataset`, with `n_train` as a parameter |
| `testyyy_data.py` | local copy of `ml_TESTYYY`'s `make_toy_data` + its `set_seed` |
| `build_notebooks.py` | generates and executes both notebooks |
| `ensemble_ablation.py` | one-at-a-time attribution of the legacy-vs-current deep-ensemble gap |
| `sine_legacy_n50.ipynb` | legacy methods, `ml_uncertainty_msc` recipe, N=50, raw arrays |
| `sine_testyyy_protocol.ipynb` | legacy methods, exactly `ml_TESTYYY`'s E1 data and protocol |
| `figures/` | the same plots as PNGs (`n50_*`, `protocol_*`) |

## The two setups

|  | `sine_legacy_n50` | `sine_testyyy_protocol` |
| --- | --- | --- |
| source | `ml_uncertainty_msc/src/data.py` | `ml_TESTYYY/src/data.py` |
| seed | 42 | **0** — `e1_toy.py`'s `FIGURE_SEED` |
| training inputs | `linspace(0, 6, 50)` — regular grid | `rng.uniform(0, 6, 250)` — random draws |
| N train | 50 | 250 |
| evaluation grid | `linspace(-2, 8, 1000)` | `linspace(-2, 8, 500)` |
| sigma | 0.1 constant | 0.1 constant |
| RNG | `np.random.RandomState(42)` | `np.random.default_rng(0)` |
| fitted on | raw arrays | **standardised** `x_train_std`, `y_train_std`; predictions un-standardised before plotting |

Both notebooks define `X_train`, `y_train`, `X_eval`, `SEED` and a `show()`
helper, so the per-method cells are literally the same text in both — only what
those names are bound to differs. Both run identical method settings:
`epochs=1000`, dropout 0.2, `T=200`, 10 ensemble members, Laplace last-layer
full Hessian with `marglik` prior precision.

`sine_testyyy_protocol.ipynb` reproduces `e1_toy.py`'s data handling exactly,
but not its whole experiment: that script runs seeds 0..19 and reports
`mean ± std` of metrics, and it uses the CURRENT methods. This notebook is one
seed, no metrics, legacy methods.

## Provenance warning

`src/methods.py` — the module holding MC dropout, the ensemble wrappers,
Bayes by Backprop and the Laplace wrapper — **was never committed to git**
(`git log --all -- src/methods.py` in `ml_uncertainty_msc` is empty). It
survived only as a stale `src/__pycache__/methods.cpython-313.pyc`
(mtime 2026-05-11 11:35), and `legacy_methods.py` is a hand-decompilation of
that bytecode.

Read off the disassembly and therefore reliable: class and function structure,
every numeric constant, every default argument, the control flow, the order of
operations. Guesswork and therefore not evidence of anything: local variable
spelling, comments, blank lines.

## Fidelity checks

Each notebook's first cell asserts that its vendored data recipe still matches
the tree it was copied from, and prints the result:

- `legacy_data.check_matches_live()` → `make_sin_homo(250)` vs
  `ml_uncertainty_msc`'s `load_sin_homo()`, all seven arrays.
- `testyyy_data.check_matches_testyyy(seed=0)` → `make_toy_data(0)` vs
  `ml_TESTYYY`'s own, all seven arrays plus the standardiser and the region
  masks.

Both are pinned by absolute path (`REFERENCE_REPO` in each module, overridable
with `ML_UNCERTAINTY_REPO` / `ML_TESTYYY_REPO`). This matters: both trees
contain a package called `src`, and an unqualified `import src.data` would pick
up whichever happens to be on `sys.path` — they are different files, and
`ml_TESTYYY`'s has no `load_sin_homo` in it at all. If a reference tree is
missing, the check reports SKIPPED rather than failing, since this folder is
meant to stay runnable after the references move.

## One thing that is deliberately not faithful

`make_toy_data` calls the project's `set_seed`, which sets torch's default dtype
to float64. The legacy methods build float32 tensors explicitly, so a float64
default makes `nn.Linear` weights float64 and the first forward pass raises. The
notebook calls `testyyy_data.restore_float32_default()` immediately after
building the data. No data value depends on this: the arrays come from
`np.random.default_rng(seed)`, which reads neither torch nor global NumPy state.

## `ensemble_ablation.py`

The legacy deep ensemble fans out on extrapolation; the current one does not
(MPIW extrapolation / in-range = 1.08, PICP@95 = 0.62 outside the training
range, over 20 seeds). Five things differ at once, so this script changes them
one at a time on `ml_TESTYYY`'s own seed-0 data, under its own protocol, with
the legacy training loop held fixed. Result:

| step | MPIW ratio |
| --- | --- |
| legacy as-is (relu, 2x64, no prior, 1000ep, M=10) | 1.88 |
| + tanh | 1.84 |
| **+ one hidden layer** | **0.81** |
| + hidden 50 | 0.85 |
| + 4000 epochs | 0.98 |
| + prior gamma=1 | 1.13 |
| + M=5 | 1.09 |
| + homoscedastic  [= the current method] | 1.03 |

**Depth is the whole story.** Dropping from two hidden layers to one takes the
ratio from 1.84 to 0.81; activation, ensemble size, epoch budget and the prior
are worth a few hundredths each. Controls run from the legacy end agree: M=5
alone gives 2.00, gamma=1 alone 1.61, 4000 epochs alone 2.57.

The epistemic term still grows outside the training range in every variant
(8-23x) — it is just so small in absolute terms at depth 1 that the total band
stays pinned at the aleatoric floor, `2 * 1.96 * 0.1 = 0.392`, which is exactly
the in-range MPIW in every row.

`DEFAULT_DEPTH = 1` is a deliberate, documented choice in `ml_TESTYYY`
(Foong et al. prove their mean-field result for one hidden layer). This is not
an argument against it — it is a note that the choice has a side effect on the
ensemble, and that `e1_toy.py`'s own sanity check expects the ensemble to
expand "a little" and gets 1.08.

## What is NOT comparable to current results

The legacy plotting draws a `+-2 sigma` band from a single **total** standard
deviation and compares it against the **clean** `sin(x)`. Both current pipelines
report `var_aleatoric` and `var_epistemic` separately and evaluate against noisy
observations. These are different quantities on different targets; do not put
the two side by side in the thesis without saying so.

## Reproduce

```bash
cd C:/Users/michal/Desktop/ml_TESTYYY/legacy_probe
C:/Users/michal/Desktop/ml_uncertainty_msc/.venv/Scripts/python.exe build_notebooks.py
```

Either project copy's venv works — both have torch, sklearn, laplace,
matplotlib, nbformat and nbconvert. Deterministic from the seeds above; the
ensemble's member *i* is always trained with `seed=i`, which is baked into the
legacy `train_ensemble` and does not follow `SEED`.
