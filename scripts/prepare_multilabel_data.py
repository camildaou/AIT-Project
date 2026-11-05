#!/usr/bin/env python
"""Build the Option A multi-label dataset (separate from the single-label track).

1. Load the raw GoEmotions splits (reuses the cached `data/raw/*.parquet`
   of the single-label pipeline; downloads them only if missing).
2. Project labels onto [joy, sadness, anger, fear] and keep rows with >= 1 target.
3. Check structural invariants and the `expected_counts` in configs/multilabel.yaml.
   On any mismatch nothing is written and the discrepancies are printed.
4. Write data/processed/multilabel/{train,validation,test}.csv and dataset_stats.json.

Usage
-----
    python scripts/prepare_multilabel_data.py            # refuses to overwrite existing output
    python scripts/prepare_multilabel_data.py --force    # regenerate (deterministic)
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
from emotion_ait.multilabel.config import load_multilabel_config  # noqa: E402
from emotion_ait.multilabel.preprocessing import (  # noqa: E402
    compare_with_expected,
    filter_multilabel_option_a,
    split_statistics,
    validate_processed_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_multilabel_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="Path to the multi-label YAML config.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed multi-label files.")
    args = parser.parse_args()
    config = load_multilabel_config(args.config)

    outputs = [config.processed_dir / f"{s}.csv" for s in config.splits] + [config.processed_dir / "dataset_stats.json"]
    if any(p.exists() for p in outputs) and not args.force:
        parser.error(f"{config.processed_dir} already has output; pass --force to regenerate it.")

    raw_frames = download_raw_dataset(config.base)
    processed, stats, problems = {}, {}, []
    for split in config.splits:
        df = filter_multilabel_option_a(raw_frames[split], split, config.base.dataset.original_label_names, config.target_labels)
        processed[split] = df
        stats[split] = split_statistics(df, len(raw_frames[split]), config.target_labels)
        problems += validate_processed_split(df, split, config.target_labels)
        problems += compare_with_expected(stats[split], config.expected_counts.get(split, {}), split)
    ids = {split: set(df["example_id"]) for split, df in processed.items()}
    for i, a in enumerate(config.splits):
        for b in config.splits[i + 1:]:
            if ids[a] & ids[b]:
                problems.append(f"{len(ids[a] & ids[b])} example ids appear in both {a} and {b}")

    for split in config.splits:
        s = stats[split]
        logger.info("%s: raw=%d retained=%d discarded=%d multi-label=%d positives=%s",
                    split, s["raw"], s["retained"], s["discarded"], s["multi_label"], s["positives"])
    if problems:
        for problem in problems:
            logger.error("DISCREPANCY: %s", problem)
        logger.error("Nothing was written. Investigate the discrepancies above before changing data or config.")
        sys.exit(1)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    for split, df in processed.items():
        df.to_csv(config.processed_dir / f"{split}.csv", index=False)
    with open(config.processed_dir / "dataset_stats.json", "w", encoding="utf-8") as fh:
        json.dump({"dataset_id": config.dataset_id, "policy": config.policy,
                   "target_labels": config.target_labels, "splits": stats}, fh, indent=2)
    logger.info("All counts match configs/multilabel.yaml. Wrote %s", config.processed_dir)


if __name__ == "__main__":
    main()
