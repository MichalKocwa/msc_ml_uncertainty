"""Redraw E1b figures from SAVED predictions - any layout, no refitting.

    .venv/Scripts/python.exe experiments/e1b_toy_depth2_figures.py                      # seed 0, both layouts, 5 member lines
    .venv/Scripts/python.exe experiments/e1b_toy_depth2_figures.py --layout separate
    .venv/Scripts/python.exe experiments/e1b_toy_depth2_figures.py --layout all --members 0
    .venv/Scripts/python.exe experiments/e1b_toy_depth2_figures.py --seed 3 --methods ensemble laplace gp
    .venv/Scripts/python.exe experiments/e1b_toy_depth2_figures.py --ylim -3 3 --prefix e1b_d2_zoom

Reads `results/e1b_toy_depth2_predictions/seed<N>.npz` written by
`experiments/e1b_toy_depth2.py` and calls the same `src.style.make_method_figures`
that script used, so a redraw shows exactly the numbers in the metrics table
(DEC-007) - the only things that can change here are layout, which methods are
shown, how many individual-run lines are drawn, the y-range and the file name.

`--ylim` is an explicit decision about the y-range and is printed with the
output; without it the limits cover every band with nothing clipped.
`--prefix` defaults to the experiment's own name and overwrites the figures the
run wrote; pass a different one to keep both.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import TRAIN_RANGE  # noqa: E402
from src.experiment_io import METHOD_ORDER, load_predictions  # noqa: E402
from src.style import make_method_figures  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "e1b_toy_depth2"
PREDICTIONS_DIR = REPO_ROOT / "results" / "{}_predictions".format(EXPERIMENT)
FIGURE_DIR = REPO_ROOT / "figures"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=["separate", "all", "both"], default="both")
    parser.add_argument("--methods", nargs="+", default=list(METHOD_ORDER))
    parser.add_argument("--members", type=int, default=5, help="individual-run lines per sampling method (0 = none)")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--prefix", default=EXPERIMENT)
    parser.add_argument("--no-title", action="store_true")
    parser.add_argument("--predictions", type=Path, default=None, help="explicit .npz path (default: seed<N>.npz)")
    args = parser.parse_args()

    path = args.predictions or PREDICTIONS_DIR / "seed{}.npz".format(args.seed)
    if not path.exists():
        raise SystemExit("no saved predictions at {}; run experiments/e1b_toy_depth2.py first".format(path))
    arrays, panels, meta = load_predictions(path)
    if meta.get("quick", False):
        print("*** these predictions are from a --quick run: NOT A RESULT ***")

    title = "" if args.no_title else (
        "E1b, {}, seed {}: predictive distributions (inner band $\\pm2\\sigma$ aleatoric, outer $\\pm2\\sigma$ total)".format(
            meta.get("arch", "?"), meta.get("seed", args.seed)))
    written = make_method_figures(
        panels, arrays["x_test"], arrays["x_train"], arrays["y_train"], arrays["y_true"],
        TRAIN_RANGE, FIGURE_DIR, args.prefix, suptitle=title, layout=args.layout,
        n_sample_lines=args.members, ylim=args.ylim, methods=args.methods,
    )
    print("source: {} (run {}, arch {})".format(path, meta.get("run_id"), meta.get("arch")))
    if args.ylim:
        print("y-range fixed explicitly to {} - a decision, record it if the figure is used".format(args.ylim))
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
