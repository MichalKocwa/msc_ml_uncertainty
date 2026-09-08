"""Fetch UCI benchmark data and the literature train/test splits (brief section 5.2).

Downloads the exact data files and 20-way split indices used by
Hernandez-Lobato & Adams (2015), reused by Gal & Ghahramani (2016) and
Lakshminarayanan et al. (2017), as vendored in yaringal/DropoutUncertaintyExps
and pinned to a specific commit. `data/uci_splits/` is gitignored (the
LICENSE on that repo is CC BY-NC 4.0 and ambiguous about redistribution —
see docs/datasets.md), so every environment re-fetches it; the committed
`data/uci_splits.checksums.json` is what makes that reproducible instead of
"trust whatever is at the URL today".

Usage:
    python scripts/fetch_data.py                  # fetch + verify against committed checksums
    python scripts/fetch_data.py --update-checksums  # maintainer only: regenerate the checksum file

Source:  https://github.com/yaringal/DropoutUncertaintyExps
Commit:  6eb4497628d12b0f300f4b4f6bdc386bebad565c (2018-08-09)
License: CC BY-NC 4.0 for Yarin Gal's contributions (see docs/datasets.md)
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/yaringal/DropoutUncertaintyExps.git"
COMMIT = "6eb4497628d12b0f300f4b4f6bdc386bebad565c"

# our canonical dataset name -> upstream UCI_Datasets/ directory name
#
# Exactly the three datasets `CLAUDE.md`'s stage table names for E2, and no
# more (DEC-011). `wine_quality_red`, `kin8nm` and `power_plant` were removed
# on cost grounds measured in this repository — see N-012 for the per-split
# timings. The short version: at 20 splits x 6 methods, these three cost ~1.9 h
# in total, whereas adding `wine-quality-red` (N=1599) alone would have added
# another 1.7 h, and the two large sets (N=8192, N=9568) are far beyond that
# again because `gp` scales as O(N^3).
#
# Upstream has more (bostonHousing, naval-propulsion-plant,
# protein-tertiary-structure, ...). Adding any of them is a scope decision that
# changes a table in the thesis, not a configuration tweak: record it as a DEC
# and update `CLAUDE.md`'s stage table first.
DATASETS = {
    "yacht": "yacht",
    "energy": "energy",
    "concrete": "concrete",
}

# Fetched for every dataset regardless of n_splits.
#
# `dropout_rates.txt` / `tau_values.txt` are the grid `experiment.py` reads
# in its hyperparameter search (`for dropout_rate in dropout_rates: for tau
# in tau_values:`), so P13's literature-validation run must take its grid
# from these files rather than from values typed into our own source — the
# whole point of that run is that the protocol is upstream's, not ours.
FIXED_FILES = [
    "data.txt",
    "index_features.txt",
    "index_target.txt",
    "n_splits.txt",
    "n_hidden.txt",
    "n_epochs.txt",
    "dropout_rates.txt",
    "tau_values.txt",
]

# Published per-split test metrics for the reference "1 hidden layer, 40
# epochs x100 multiplier" run (confirmed identical across all six datasets
# by inspecting experiment.py: the "100_xepochs_1_hidden_layers" filename
# suffix is exactly `--epochx`/`--hidden` from that run, i.e. actual
# training used 40*100=4000 epochs, not the "10x" the repo's README prose
# suggests). `test_MC_rmse` (MC-dropout-averaged), not the plain
# `test_rmse` (single deterministic pass), is the one that reproduces the
# RMSE column of the repo's own README table — confirmed by recomputing
# mean+-SE over these 20-line files and comparing (see docs/datasets.md).
# Both are fetched anyway since the plain pass is a legitimate additional
# diagnostic and the cost is a few KB.
RESULTS_SUFFIX = "100_xepochs_1_hidden_layers.txt"
RESULTS_FILES = [
    f"test_rmse_{RESULTS_SUFFIX}",
    f"test_MC_rmse_{RESULTS_SUFFIX}",
    f"test_ll_{RESULTS_SUFFIX}",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_ROOT = REPO_ROOT / "data" / "uci_splits"
CHECKSUMS_PATH = REPO_ROOT / "data" / "uci_splits.checksums.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clone_pinned_commit(tmp_dir: Path) -> Path:
    repo_dir = tmp_dir / "DropoutUncertaintyExps"
    subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", COMMIT], check=True)
    return repo_dir


def _split_index_files(n_splits: int):
    for i in range(n_splits):
        yield f"index_train_{i}.txt"
        yield f"index_test_{i}.txt"


def fetch(update_checksums: bool) -> None:
    if not update_checksums and not CHECKSUMS_PATH.exists():
        print(
            f"ERROR: {CHECKSUMS_PATH} not found. If this is a fresh checkout, "
            "it should be committed in the repo. To bootstrap it for the "
            "first time, run with --update-checksums.",
            file=sys.stderr,
        )
        sys.exit(1)

    known_checksums = (
        json.loads(CHECKSUMS_PATH.read_text()) if CHECKSUMS_PATH.exists() else {}
    )
    computed_checksums = {}

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = _clone_pinned_commit(Path(tmp))

        for our_name, upstream_name in DATASETS.items():
            upstream_dir = repo_dir / "UCI_Datasets" / upstream_name
            data_src_dir = upstream_dir / "data"
            results_src_dir = upstream_dir / "results"

            data_dest_dir = DEST_ROOT / our_name
            results_dest_dir = data_dest_dir / "results"
            data_dest_dir.mkdir(parents=True, exist_ok=True)
            results_dest_dir.mkdir(parents=True, exist_ok=True)

            n_splits = int((data_src_dir / "n_splits.txt").read_text().split()[0])
            data_filenames = FIXED_FILES + list(_split_index_files(n_splits))

            file_pairs = (
                [(data_src_dir / f, data_dest_dir / f, f"{our_name}/{f}") for f in data_filenames]
                + [(results_src_dir / f, results_dest_dir / f, f"{our_name}/results/{f}") for f in RESULTS_FILES]
            )

            for src_path, dest_path, key in file_pairs:
                shutil.copy(src_path, dest_path)

                digest = _sha256(dest_path)
                computed_checksums[key] = digest

                if not update_checksums:
                    expected = known_checksums.get(key)
                    if expected is None:
                        print(f"ERROR: no committed checksum for '{key}'", file=sys.stderr)
                        sys.exit(1)
                    if expected != digest:
                        print(
                            f"ERROR: checksum mismatch for '{key}' "
                            f"(expected {expected}, got {digest}). "
                            "Upstream file changed or download is corrupted — "
                            "do not proceed without investigating.",
                            file=sys.stderr,
                        )
                        sys.exit(1)

    if update_checksums:
        CHECKSUMS_PATH.write_text(json.dumps(computed_checksums, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {len(computed_checksums)} checksums to {CHECKSUMS_PATH}")
    else:
        print(f"OK: fetched and verified {len(computed_checksums)} files into {DEST_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--update-checksums", action="store_true",
        help="regenerate data/uci_splits.checksums.json instead of verifying against it (maintainer only)",
    )
    args = parser.parse_args()
    fetch(update_checksums=args.update_checksums)
