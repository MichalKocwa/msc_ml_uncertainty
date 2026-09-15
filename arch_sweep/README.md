# `arch_sweep/` — architecture sensitivity on the E1 sine

A sweep over the shared backbone's `depth` and `activation`, applied identically
to every network-based method, on `ml_TESTYYY`'s own E1 data and protocol.

**This is a sensitivity sweep, not tuning, and not a result of E1.** The
distinction is DEC-014's: varying a setting for all methods at once and
reporting the whole sweep is a measurement; varying it and reporting only the
configuration whose picture looked best is selection on the evaluation set,
which E1 cannot support because its test grid *is* its evaluation set. Every
configuration's metrics are therefore recorded, not just the ones that looked
good.

## Isolation

Nothing here can change the main experiment.

- `sweep_config.py` is the only place that names an architecture value, and
  nothing outside this folder imports it.
- Methods are built through the repo's own `METHODS` registry with the
  architecture passed as explicit keyword arguments — the same route `--set`
  takes in `experiments/e1_toy.py`. No value here can become a repo default.
- All output goes to `arch_sweep/results/` and `arch_sweep/figures/`. The
  repository's own `results/` and `figures/` are never written to.
- `src/` is imported read-only: `make_toy_data`, `compute_metrics`,
  `plot_predictive_panel`, `append_rows`, `set_seed`, `METHODS`.

## Protocol

Inherited from `experiments/e1_toy.py` unchanged: `set_seed` before data and
models, fit on the standardised arrays, one `predict` per method per seed,
un-standardise before scoring, metrics against the noisy `y_test` in original
y units, split by the project's own region masks.

**Verified, not assumed.** The baseline configuration `d1_h50_tanh` reproduces
`results/e1_toy_metrics.csv`'s seed-0 rows *exactly* — `rmse`, `ll`, `picp95`
and `mpiw95` agree to every recorded digit for all six methods. Any difference
seen elsewhere in this folder is the architecture, not a drifted harness. Re-run
that check before trusting a future change to this code.

## The grid

`depth ∈ {1, 2, 3}` × `activation ∈ {tanh, relu}`, width fixed at 50.

Width was dropped after measuring what it costs `laplace`, which runs
`subset_of_weights="all"` with `hessian_structure="full"` and so materialises a
dense P×P Hessian:

| configuration | parameters | Hessian (float64) | fit time |
| --- | --- | --- | --- |
| `d1_h50` | 152 | 0.0002 GB | 25 s |
| `d2_h50` | 2 702 | 0.05 GB | 32 s |
| `d3_h50` | 5 252 | 0.21 GB | 39 s |
| `d2_h100` | 10 402 | 0.81 GB | 48 s *at 20 epochs* |
| `d3_h100` | 20 502 | **3.13 GB** | no output in >10 min |

`set_seed` pins `torch.set_num_threads(1)` for determinism, so the O(P³)
factorisation cannot be parallelised out of the way. **This is itself a finding
for the cost discussion:** full-network Laplace is the method whose scaling
forbids the wider architectures, not one that merely does worse on them.

## Running it

```bash
cd arch_sweep
..\.venv\Scripts\python.exe run_sweep.py --quick      # plumbing check, ~2 min
..\.venv\Scripts\python.exe run_sweep.py              # stage 1, seed 0, ~19 min
..\.venv\Scripts\python.exe run_sweep.py --seeds 0 1 2 --configs d2_h50_relu
```

`--row-limits` shares the figure's y axis within each row instead of across the
whole image — worth using if one configuration's band flattens the others.
Nothing is ever clipped either way: a squashed panel is honest, a cropped one
lies.

## Output

- `figures/arch_sweep_seed<N>.png` — one image, rows are configurations, columns
  are methods. Reading across a row compares methods at fixed architecture;
  reading down a column compares one method across architectures. `gp` is
  repeated in every row: it has no architecture, and a row is easier to read
  against the one exact posterior in the comparison than against nothing.
- `results/arch_sweep_metrics.csv` — appended, never overwritten, with
  `config_id`, `depth`, `hidden`, `activation` alongside the standard
  `METRIC_COLUMNS`.

## Stage 1 findings (seed 0 only — nothing here is stable yet)

One seed. `map`'s `ll = +0.08` at `d2_h50_relu` and `mcd`'s `PICP = 1.000` at
`d1_h50_relu` are exactly the kind of single-seed accident that a 20-seed run
exists to wash out. Read the direction, not the digits.

1. **Depth destroys BBB's extrapolation.** PICP@95 outside the training range
   falls from 0.840 at depth 1 to 0.075 / 0.060 at depths 2 and 3, and
   log-likelihood from +0.28 to −27 / −33. This *strengthens* the warning
   already recorded next to `DEFAULT_DEPTH`, that Foong et al.'s universality
   result for ≥2 hidden layers is an existence proof and cannot be promised as a
   fix for mean-field VI.
2. **Laplace's high expansion ratios under ReLU are an artefact, not a
   success.** `d2_h50_relu` and `d3_h50_relu` give ratios of 7.69 and 8.61 and
   PICP 1.000, but the band edges are visibly jagged in the extrapolation
   region — the discontinuous-Jacobian behaviour `src/methods/laplace.py`'s own
   docstring says `tanh` was chosen to avoid. Do not read those numbers as good
   uncertainty without looking at the panel.
3. **Nothing beats the GP reference.** It is the only estimator in the figure
   with a smooth, symmetric funnel on both sides, PICP 1.000 and the best
   extrapolation log-likelihood (+0.37).
4. **`map`'s ratio is exactly 1.00 in every configuration**, as it must be — it
   reports `var_epistemic` as explicit zeros, so architecture cannot move its
   interval width. A useful sanity check on the harness.
