#!/usr/bin/env python
"""Train the MDL classifier across the configured training subsets and
append one row per (subset) run to the shared results CSV.

Trains fresh on each subset (never incrementally) via
`emotion_ait.experiments.run_experiment`, which evaluates against the
same fixed validation/test splits every time and logs to
`results/metrics/results.csv`.

Usage
-----
    python scripts/train_mdl.py                         # all 9 configured subsets, increasing size
    python scripts/train_mdl.py --subsets train_abs100   # a single subset
    python scripts/train_mdl.py --n 3 --k 1.0            # override hyperparameters
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_ait.config import load_config  # noqa: E402
from emotion_ait.experiments import run_experiment  # noqa: E402
from emotion_ait.mdl_classifier import MDLClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_mdl")

# The 9 configured subsets, in the increasing-size order agreed for the
# learning-curve sweep (see README section 10 / data/subsets/manifest.json
# for the realized sizes: 100, 157, 313, 249, 625, 500, 1250, 2188, 3125).
DEFAULT_SUBSET_ORDER = [
    "train_abs100",
    "train_pct5",
    "train_pct10",
    "train_abs250",
    "train_pct20",
    "train_abs500",
    "train_pct40",
    "train_pct70",
    "train_pct100",
]


def load_processed_df(config) -> pd.DataFrame:
    """Load the already-prepared processed splits from disk (written by
    `scripts/prepare_data.py`); never re-filters or re-splits the data."""
    frames = []
    for split in config.dataset.splits:
        path = config.paths.processed_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python scripts/prepare_data.py` first.")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        metavar="SUBSET_NAME",
        help=f"Subset names to train on, in order (default: all 9, increasing size: {DEFAULT_SUBSET_ORDER}).",
    )
    parser.add_argument("--n", type=int, default=3, help="n-gram order (default: 3, i.e. trigram).")
    parser.add_argument("--smoothing", type=str, default="laplace", help="Smoothing scheme (default: laplace).")
    parser.add_argument("--k", type=float, default=1.0, help="Additive smoothing constant (default: 1.0).")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_df = load_processed_df(config)
    subset_names = args.subsets or DEFAULT_SUBSET_ORDER

    for subset_name in subset_names:
        logger.info(
            "Training MDL (n=%d, smoothing=%s, k=%s) on subset '%s'", args.n, args.smoothing, args.k, subset_name
        )
        model = MDLClassifier(n=args.n, smoothing=args.smoothing, k=args.k)
        record = run_experiment(
            model=model,
            model_name="mdl",
            subset_name=subset_name,
            processed_df=processed_df,
            config=config,
            config_summary={"n": args.n, "smoothing": args.smoothing, "k": args.k},
        )
        logger.info(
            "Subset '%s': train_size=%d test_accuracy=%.4f test_macro_f1=%.4f",
            subset_name,
            record.train_subset_size,
            record.test_accuracy,
            record.test_macro_f1,
        )

    logger.info(
        "Done. Combined results: %s. Per-subset metrics CSVs: %s. Per-subset predictions: %s.",
        config.paths.results_dir / "metrics" / "results.csv",
        config.paths.results_dir / "metrics" / "csv",
        config.paths.results_dir / "predictions",
    )


if __name__ == "__main__":
    main()
