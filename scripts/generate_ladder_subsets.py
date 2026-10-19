#!/usr/bin/env python
"""Generate the multi-seed training-size "ladder" subsets used for the
MDL-vs-LSTM comparison (8 sizes x 5 seeds = 40 subsets, shared by both
models for 80 total model runs).

Reuses the already-prepared processed splits (run
`scripts/prepare_data.py` first if you haven't). Does not train or
evaluate any model -- this only generates and validates the nested,
stratified training subsets and writes their manifests to
`data/subsets/ladder/`.

Usage
-----
    python scripts/generate_ladder_subsets.py
    python scripts/generate_ladder_subsets.py --sizes 100 500 3125 --seeds 42 43
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
from emotion_ait.sampling import generate_ladder_subsets  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_ladder_subsets")


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
        "--sizes", type=int, nargs="+", default=None,
        help="Absolute training sizes (default: config.sampling.ladder_sizes).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Random seeds (default: config.sampling.ladder_seeds).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_df = load_processed_df(config)

    results = generate_ladder_subsets(processed_df, config, sizes=args.sizes, seeds=args.seeds)

    print("\nGenerated and validated ladder subsets:")
    for seed in sorted(results):
        for size in sorted(results[seed]):
            subset = results[seed][size]
            print(
                f"  seed={seed} size={size}: actual={subset['actual_size']} "
                f"classes={subset['class_counts']}"
            )
    print(f"\nWritten under {config.paths.subsets_dir / 'ladder'}")


if __name__ == "__main__":
    main()
