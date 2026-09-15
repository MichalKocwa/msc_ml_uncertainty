"""Sweep of Bayes-by-Backprop configurations on the E1 sine, from scratch.

    .venv/Scripts/python.exe bbb_experiments/run_bbb.py                 # all configs, seed 0
    .venv/Scripts/python.exe bbb_experiments/run_bbb.py --configs relu_fixed_lrt --seeds 0 1 2 3 4
    .venv/Scripts/python.exe bbb_experiments/run_bbb.py --quick          # smoke test only

Question: which ingredient of mean-field VI makes its predictive band grow
outside the training range on this problem, given that the repo's `bbb` (E1,
N-014) shows a 1.01x band ratio, i.e. no growth at all?

Each named config in `CONFIGS` changes one or a few things relative to the
previous one, so the sweep reads as an ablation. The headline number is

    band ratio = MPIW@95(extrapolation) / MPIW@95(in-range)

alongside PICP@95 in extrapolation (a band that grows but still under-covers
is not a success), and the in-range RMSE/LL (a band that grows because the fit
broke is not one either).

**Nothing outside `bbb_experiments/` is written.** `src/` is imported
read-only (data generator, metrics, plotting helpers). This module's own BBB
implementation never touches `src.methods.cache`, so the shared `cache/` is
neither read nor added to.
"""
import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import NOISE_SIGMA, REGIONS, TRAIN_RANGE, make_toy_data, true_function  # noqa: E402
from src.metrics import METRIC_COLUMNS, compute_metrics  # noqa: E402
from src.style import apply_style, plot_predictive_panel  # noqa: E402

from bbb_vi import BBBConfig, BBBRegressor  # noqa: E402

RESULTS_DIR = HERE / "results"
FIGURE_DIR = HERE / "figures"
PRED_DIR = RESULTS_DIR / "predictions"

# `fixed_sigma2="true"` is resolved per seed to `(NOISE_SIGMA / y_std)^2`,
# i.e. the data-generating noise in standardised units — what Foong et al.
# (2019) do in their 1D experiment (DEC-005 note).
CONFIGS = {
    # 0. reproduce the repo setting inside this implementation, as a check that
    #    the sandbox reproduces the ~1.0x band ratio before anything is changed
    "repo_like": BBBConfig(
        name="repo_like",
        notes="tanh, minibatch 128, rho_init -3, learned noise, K=1 — mirrors src/methods/bbb.py",
    ),
    # 1. optimisation only: full batch, 4 ELBO samples, KL counted exactly once per step
    "tanh_fullbatch_K4": BBBConfig(
        name="tanh_fullbatch_K4", batch_size=None, elbo_samples=4,
        notes="as repo_like but full-batch and K=4",
    ),
    # 2. + local reparameterisation trick, + sigmas initialised nearer the prior
    "tanh_lrt_rho-1": BBBConfig(
        name="tanh_lrt_rho-1", batch_size=None, elbo_samples=4, local_reparam=True, rho_init=-1.0,
        notes="tanh, LRT, rho_init -1 (sigma 0.31)",
    ),
    # 3. + fixed observation noise at the true value (Foong et al. 2019)
    "tanh_lrt_fixed": BBBConfig(
        name="tanh_lrt_fixed", batch_size=None, elbo_samples=4, local_reparam=True, rho_init=-1.0,
        fixed_sigma2="true", notes="tanh, LRT, rho_init -1, fixed noise",
    ),
    # 4. activation: relu instead of tanh, otherwise as 1
    "relu_fullbatch_K4": BBBConfig(
        name="relu_fullbatch_K4", activation="relu", batch_size=None, elbo_samples=4,
        notes="relu, full-batch, K=4, rho_init -3, learned noise",
    ),
    # 5. relu + LRT + rho_init -1, learned noise
    "relu_lrt_rho-1": BBBConfig(
        name="relu_lrt_rho-1", activation="relu", batch_size=None, elbo_samples=4,
        local_reparam=True, rho_init=-1.0, notes="relu, LRT, rho_init -1, learned noise",
    ),
    # 6. relu + LRT + fixed noise
    "relu_lrt_fixed": BBBConfig(
        name="relu_lrt_fixed", activation="relu", batch_size=None, elbo_samples=4,
        local_reparam=True, rho_init=-1.0, fixed_sigma2="true",
        notes="relu, LRT, rho_init -1, fixed noise",
    ),
    # 7. relu + LRT + fixed noise + Foong's layer-scaled prior (omega = 4)
    "relu_lrt_fixed_layerwise": BBBConfig(
        name="relu_lrt_fixed_layerwise", activation="relu", batch_size=None, elbo_samples=4,
        local_reparam=True, rho_init=-1.0, fixed_sigma2="true", prior_sigma="layerwise",
        notes="relu, LRT, fixed noise, layerwise prior omega=4",
    ),
    # 8. relu + LRT + fixed noise + KL warm-up over the first 25% of epochs
    "relu_lrt_fixed_warmup": BBBConfig(
        name="relu_lrt_fixed_warmup", activation="relu", batch_size=None, elbo_samples=4,
        local_reparam=True, rho_init=-1.0, fixed_sigma2="true", kl_warmup_frac=0.25,
        notes="relu, LRT, fixed noise, KL warm-up 25%",
    ),
    # 9. tanh but with the layerwise prior — is it the activation or the prior?
    "tanh_lrt_fixed_layerwise": BBBConfig(
        name="tanh_lrt_fixed_layerwise", batch_size=None, elbo_samples=4,
        local_reparam=True, rho_init=-1.0, fixed_sigma2="true", prior_sigma="layerwise",
        notes="tanh, LRT, fixed noise, layerwise prior omega=4",
    ),
    # 10/11. can tanh be made to grow at all? two probes: a 3x wider flat prior,
    #        and a 4x wider hidden layer — each a different Bayesian model,
    #        so a positive result here would not transfer to the shared setup
    "tanh_lrt_fixed_prior3": BBBConfig(
        name="tanh_lrt_fixed_prior3", batch_size=None, elbo_samples=4,
        local_reparam=True, rho_init=-1.0, fixed_sigma2="true", prior_sigma=3.0,
        notes="tanh, LRT, fixed noise, prior sigma 3",
    ),
    "tanh_lrt_fixed_h200": BBBConfig(
        name="tanh_lrt_fixed_h200", hidden=200, batch_size=None, elbo_samples=4,
        local_reparam=True, rho_init=-1.0, fixed_sigma2="true",
        notes="tanh, LRT, fixed noise, 200 hidden",
    ),
}


def resolve_config(config: BBBConfig, y_std: float) -> BBBConfig:
    if config.fixed_sigma2 == "true":
        return dataclasses.replace(config, fixed_sigma2=(NOISE_SIGMA / y_std) ** 2)
    return config


def run_one(config: BBBConfig, seed: int, quick: bool):
    data = make_toy_data(seed)
    cfg = resolve_config(config, data.standardiser.y_std)
    if quick:
        cfg = dataclasses.replace(cfg, epochs=20, T=10)

    method = BBBRegressor(cfg)
    t0 = time.perf_counter()
    method.fit(data.x_train_std, data.y_train_std, seed=seed, log_every=max(1, cfg.epochs // 40))
    fit_seconds = time.perf_counter() - t0
    pred = method.predict(data.x_test_std)

    s = data.standardiser
    mean = s.unstandardise_mean(pred.mean)
    var_aleatoric = s.unstandardise_var(pred.var_aleatoric)
    var_epistemic = s.unstandardise_var(pred.var_epistemic)
    samples = s.unstandardise_mean(pred.samples)

    diag = method.diagnostics()
    rows = []
    for region in REGIONS:
        mask = data.regions[region]
        values = compute_metrics(
            y=data.y_test[mask], mean=mean[mask],
            var_aleatoric=var_aleatoric[mask], var_epistemic=var_epistemic[mask],
        )
        rows.append(dict(
            config=cfg.name, seed=seed, region=region, n_points=int(mask.sum()), **values,
            std_epistemic_mean=float(np.sqrt(var_epistemic[mask]).mean()),
            n_parameters=method.n_parameters, fit_seconds=round(fit_seconds, 2),
            **{k: round(v, 6) for k, v in diag.items()},
            activation=cfg.activation, local_reparam=cfg.local_reparam, rho_init=cfg.rho_init,
            fixed_noise=cfg.fixed_sigma2 is not None, prior=str(cfg.prior_sigma),
            batch_size=cfg.batch_size, elbo_samples=cfg.elbo_samples, epochs=cfg.epochs,
            kl_warmup_frac=cfg.kl_warmup_frac,
        ))

    panel = dict(mean=mean, var_aleatoric=var_aleatoric, var_epistemic=var_epistemic, samples=samples)
    return rows, panel, data, method.trace


def band_ratio(rows):
    by_region = {r["region"]: r for r in rows}
    return by_region["extrapolation"]["mpiw95"] / by_region["in_range"]["mpiw95"]


def save_panel_figure(name, seed, panel, data, trace, path):
    apply_style()
    fig, (ax, ax_tr) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    plot_predictive_panel(
        ax, "bbb", data.x_test_flat, panel["mean"], panel["var_aleatoric"], panel["var_epistemic"],
        data.x_train, data.y_train, true_function(data.x_test_flat), TRAIN_RANGE,
        show_legend=True, samples=panel["samples"], n_sample_lines=8,
    )
    ax.set_title(f"BBB sandbox — {name} (seed {seed})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-2.5, 2.5)

    ax_tr.plot(trace.epoch, trace.nll, label="E[-log lik] (sum over N)")
    ax_tr.plot(trace.epoch, trace.kl, label="KL(q || p)")
    ax_tr.plot(trace.epoch, trace.neg_elbo, label="-ELBO", color="k", lw=1)
    ax_tr.set_xlabel("epoch")
    ax_tr.set_yscale("symlog")
    ax_tr.set_title("training trace")
    ax_tr.legend()
    fig.savefig(path)
    plt.close(fig)


def save_profiles_figure(profiles, data, path):
    """`std_total(x)` of every config on one axis — the direct view of the
    question. Dashed line: the true noise sd, the floor any band must sit on.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = data.x_test_flat
    for name, (sd_total, sd_epi) in profiles.items():
        ax.plot(x, sd_total, label=name)
    ax.axhline(NOISE_SIGMA, color="k", ls="--", lw=0.8, label=f"true noise sd = {NOISE_SIGMA}")
    for edge in TRAIN_RANGE:
        ax.axvline(edge, color="#999999", lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("predictive sd (total), original y units")
    ax.set_title("Predictive standard deviation across configurations (seed 0)")
    ax.legend(ncol=2, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def summary_table(df: pd.DataFrame) -> str:
    """Markdown: one line per config, mean over seeds; the columns the
    question needs, in-range quality next to extrapolation growth.
    """
    lines = [
        "| config | seeds | in RMSE | in LL | in PICP | in MPIW | ext PICP | ext LL | ext MPIW | **ratio** | sigma_n fitted | frac var <1% prior |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, g in df.groupby("config", sort=False):
        inr = g[g.region == "in_range"]
        ext = g[g.region == "extrapolation"]
        ratio = (ext.set_index("seed")["mpiw95"] / inr.set_index("seed")["mpiw95"])
        lines.append(
            "| {} | {} | {:.3f} | {:+.3f} | {:.3f} | {:.3f} | {:.3f} | {:+.3f} | {:.3f} | **{:.2f}x** | {:.4f} | {:.2f} |".format(
                name, inr.seed.nunique(), inr.rmse.mean(), inr.ll.mean(), inr.picp95.mean(), inr.mpiw95.mean(),
                ext.picp95.mean(), ext.ll.mean(), ext.mpiw95.mean(), ratio.mean(),
                inr.sigma_fitted.mean(), inr.frac_var_below_1pct_prior.mean(),
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--quick", action="store_true", help="20 epochs, T=10: checks the script runs, never a result")
    parser.add_argument("--tag", default="sweep", help="output file stem")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{'quick_' if args.quick else ''}{args.tag}"

    all_rows, profiles = [], {}
    data0 = None
    t_start = time.perf_counter()
    for name in args.configs:
        for seed in args.seeds:
            rows, panel, data, trace = run_one(CONFIGS[name], seed, args.quick)
            for r in rows:
                r["run_id"] = run_id
            all_rows.extend(rows)
            ratio = band_ratio(rows)
            inr = rows[0]
            ext = rows[1]
            print("{:<26} seed {}  fit {:6.1f}s  in RMSE {:.3f}  in LL {:+.3f}  ext PICP {:.3f}  ratio {:.2f}x  sigma_n {:.4f}  frac<1% {:.2f}".format(
                name, seed, inr["fit_seconds"], inr["rmse"], inr["ll"], ext["picp95"], ratio,
                inr["sigma_fitted"], inr["frac_var_below_1pct_prior"]), flush=True)

            if seed == args.seeds[0]:
                data0 = data
                sd_total = np.sqrt(panel["var_aleatoric"] + panel["var_epistemic"])
                profiles[name] = (sd_total, np.sqrt(panel["var_epistemic"]))
                save_panel_figure(name, seed, panel, data, trace, FIGURE_DIR / f"{stem}_{name}_seed{seed}.png")
                np.savez(
                    PRED_DIR / f"{stem}_{name}_seed{seed}.npz",
                    x_test=data.x_test_flat, y_test=data.y_test, x_train=data.x_train.ravel(), y_train=data.y_train,
                    mean=panel["mean"], var_aleatoric=panel["var_aleatoric"], var_epistemic=panel["var_epistemic"],
                    samples=panel["samples"], trace_epoch=np.array(trace.epoch), trace_nll=np.array(trace.nll),
                    trace_kl=np.array(trace.kl),
                )

    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / f"{stem}_metrics.csv"
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)  # appended, never overwritten
    save_profiles_figure(profiles, data0, FIGURE_DIR / f"{stem}_std_profiles.png")

    table = summary_table(df)
    md_path = RESULTS_DIR / f"{stem}_summary.md"
    with open(md_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n## run {run_id} — seeds {args.seeds}, quick={args.quick}, wall {time.perf_counter() - t_start:.0f}s\n\n")
        fh.write("configs:\n```json\n")
        fh.write(json.dumps({n: CONFIGS[n].as_dict() for n in args.configs}, indent=1, default=str))
        fh.write("\n```\n\n")
        fh.write(table + "\n")
    print("\n" + table)
    print(f"\nwrote {csv_path}, {md_path}, figures in {FIGURE_DIR}")
    if args.quick:
        print("QUICK RUN — numbers above are not results.")


if __name__ == "__main__":
    main()
