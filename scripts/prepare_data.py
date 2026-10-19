#!/usr/bin/env python
"""Run the full, reproducible data-preparation pipeline:

1. Download (or reuse a cached copy of) the raw GoEmotions dataset.
2. Filter it down to the 4-class single-label subset.
3. Save the processed splits and a per-split/per-class count summary.
4. Generate every configured (nested, stratified) training subset.

Usage
-----
    python scripts/prepare_data.py [--config configs/default.yaml] [--force-download]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_ait.config import load_config  # noqa: E402
from emotion_ait.data_loader import download_raw_dataset  # noqa: E402
from emotion_ait.preprocessing import class_counts_per_split, process_all_splits  # noqa: E402
from emotion_ait.sampling import generate_all_subsets  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download raw data even if a local cache exists."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    logger.info("Step 1/4: downloading / loading raw GoEmotions data")
    raw_frames = download_raw_dataset(config, force=args.force_download)

    logger.info("Step 2/4: filtering to the 4-class single-label subset")
    processed_df = process_all_splits(raw_frames, config)

    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    for split in config.dataset.splits:
        split_df = processed_df[processed_df["original_split"] == split]
        out_path = config.paths.processed_dir / f"{split}.csv"
        split_df.to_csv(out_path, index=False)
        logger.info("Wrote %d rows for split '%s' to %s", len(split_df), split, out_path)

    counts = class_counts_per_split(processed_df)
    stats_path = config.paths.processed_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(json.loads(counts.to_json(orient="index")), fh, indent=2)
    logger.info("Step 3/4: per-split / per-class counts written to %s", stats_path)
    print("\nRetained example counts (split x class):")
    print(counts.to_string())

    logger.info("Step 4/4: generating deterministic, nested, stratified training subsets")
    subsets = generate_all_subsets(processed_df, config)
    print("\nGenerated subsets:")
    for name, subset in subsets.items():
        print(
            f"  {name}: requested={subset['requested_size']} actual={subset['actual_size']} "
            f"classes={subset['class_counts']}"
        )
        for w in subset["warnings"]:
            print(f"    WARNING: {w}")

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
