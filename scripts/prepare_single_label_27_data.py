#!/usr/bin/env python
"""Build the 27-class single-label dataset (separate from every other track).

1. Load the raw GoEmotions splits (reuses the cached `data/raw/*.parquet`;
   downloads only if missing).
2. Keep comments whose complete label set is exactly one non-neutral emotion.
3. Check structural invariants and the `expected_counts` in
   configs/single_label_27.yaml. On any mismatch nothing is written and the
   discrepancies are printed.
4. Write data/processed/single_label_27/{train,validation,test}.csv and dataset_stats.json.

Usage
-----
    python scripts/prepare_single_label_27_data.py            # refuses to overwrite existing output
    python scripts/prepare_single_label_27_data.py --force    # regenerate (deterministic)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_ait.data_loader import download_raw_dataset  # noqa: E402
from emotion_ait.single_label_27.config import load_single_label_27_config  # noqa: E402
from emotion_ait.single_label_27.preprocessing import (  # noqa: E402
    compare_with_expected,
    filter_single_label_27,
    split_statistics,
    validate_processed_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_single_label_27_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="Path to the 27-class YAML config.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed 27-class files.")
    args = parser.parse_args()
    config = load_single_label_27_config(args.config)

    outputs = [config.processed_dir / f"{s}.csv" for s in config.splits] + [config.processed_dir / "dataset_stats.json"]
    if any(p.exists() for p in outputs) and not args.force:
        parser.error(f"{config.processed_dir} already has output; pass --force to regenerate it.")

    names = config.base.dataset.original_label_names
    raw_frames = download_raw_dataset(config.base)
    processed, stats, problems = {}, {}, []
    for split in config.splits:
        df = filter_single_label_27(raw_frames[split], split, names, config.class_names)
        processed[split] = df
        stats[split] = split_statistics(df, raw_frames[split], names, config.class_names)
        problems += validate_processed_split(df, split, config.class_names)
        problems += compare_with_expected(stats[split], config.expected_counts.get(split, {}), split)
        accounted = stats[split]["retained"] + sum(stats[split]["discarded"].values())
        if accounted != stats[split]["raw"]:
            problems.append(f"{split}: retained + discarded = {accounted} != raw {stats[split]['raw']}")
    ids = {split: set(df["example_id"]) for split, df in processed.items()}
    for i, a in enumerate(config.splits):
        for b in config.splits[i + 1:]:
            if ids[a] & ids[b]:
                problems.append(f"{len(ids[a] & ids[b])} example ids appear in both {a} and {b}")

    for split in config.splits:
        s = stats[split]
        logger.info("%s: raw=%d retained=%d discarded=%s classes=%d min=%s(%d) max=%s(%d) imbalance=%.1fx",
                    split, s["raw"], s["retained"], s["discarded"], s["classes_present"], s["min_class"],
                    s["min_count"], s["max_class"], s["max_count"], s["imbalance_ratio"])
    if problems:
        for problem in problems:
            logger.error("DISCREPANCY: %s", problem)
        logger.error("Nothing was written. Investigate the discrepancies above before changing data or config.")
        sys.exit(1)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    for split, df in processed.items():
        df.to_csv(config.processed_dir / f"{split}.csv", index=False)
    with open(config.processed_dir / "dataset_stats.json", "w", encoding="utf-8") as fh:
        json.dump({"dataset_id": config.dataset_id, "class_names": config.class_names, "splits": stats}, fh, indent=2)
    logger.info("All counts match configs/single_label_27.yaml. Wrote %s", config.processed_dir)


if __name__ == "__main__":
    main()
