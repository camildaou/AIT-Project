#!/usr/bin/env python
"""Generate the nested, label-combination-stratified multi-label training ladder
(8 sizes x 5 seeds) from data/processed/multilabel/train.csv.

Writes data/subsets/multilabel_ladder/seed_<seed>/n<size>.json (ordered ids +
targets + counts), seed_<seed>/manifest.json and manifest.json (summaries).
Every ladder invariant is validated before anything is written.

Usage
-----
    python scripts/generate_multilabel_ladder.py            # refuses to overwrite existing manifests
    python scripts/generate_multilabel_ladder.py --force    # regenerate (deterministic)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_ait.multilabel.config import load_multilabel_config  # noqa: E402
from emotion_ait.multilabel.data import load_multilabel_dataframe  # noqa: E402
from emotion_ait.multilabel.sampling import (  # noqa: E402
    build_ladder_manifests,
    validate_ladder_manifests,
    write_ladder_manifests,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_multilabel_ladder")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="Path to the multi-label YAML config.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing multi-label ladder manifests.")
    args = parser.parse_args()
    config = load_multilabel_config(args.config)

    if (config.subsets_dir / "manifest.json").exists() and not args.force:
        parser.error(f"{config.subsets_dir} already has manifests; pass --force to regenerate them.")

    train = load_multilabel_dataframe("train", config)
    manifests = build_ladder_manifests(train, config)
    validate_ladder_manifests(manifests, train, config)
    write_ladder_manifests(manifests, config)
    for seed, per_size in manifests.items():
        for size, m in sorted(per_size.items()):
            logger.info("seed=%d n=%d positives=%s multi-label=%d", seed, size, m["positive_counts"],
                        sum(v for k, v in m["combination_counts"].items() if "+" in k))
    logger.info("Wrote %d manifests to %s", sum(len(p) for p in manifests.values()), config.subsets_dir)


if __name__ == "__main__":
    main()
