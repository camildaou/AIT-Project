#!/usr/bin/env python
"""Verify the integrity of the prepared dataset and generated subsets.

Run after ``scripts/prepare_data.py``. Exits with a non-zero status if
any check fails. This script re-implements (in a user-facing, printable
form) the same invariants that ``tests/`` enforce automatically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from emotion_ait.config import load_config  # noqa: E402
from emotion_ait.sampling import default_subset_requests, load_subset_ids  # noqa: E402

FAILURES: list = []


def check(condition: bool, message: str) -> None:
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        FAILURES.append(message)


def main() -> None:
    config = load_config()

    processed_frames = {}
    for split in config.dataset.splits:
        path = config.paths.processed_dir / f"{split}.csv"
        if not path.exists():
            print(f"Processed file missing: {path}. Run scripts/prepare_data.py first.")
            sys.exit(2)
        processed_frames[split] = pd.read_csv(path)
    processed_df = pd.concat(processed_frames.values(), ignore_index=True)

    target_emotions = set(config.target_task.target_emotions)
    check(
        set(processed_df["label_name"].unique()) <= target_emotions,
        f"Only the 4 target classes are present: {sorted(processed_df['label_name'].unique())}",
    )

    mapping = config.target_task.label_mapping
    consistent = all(
        (processed_df.loc[processed_df["label_name"] == name, "label"] == idx).all()
        for name, idx in mapping.items()
        if (processed_df["label_name"] == name).any()
    )
    check(consistent, "label -> label_name mapping is consistent with configs/default.yaml")

    check(
        set(processed_df["original_split"].unique()) <= set(config.dataset.splits),
        "original_split values are a subset of the configured splits",
    )

    id_to_splits = processed_df.groupby("example_id")["original_split"].nunique()
    check((id_to_splits == 1).all(), "no example_id appears in more than one split")

    check(processed_df["example_id"].is_unique, "all example_id values are globally unique")

    # --- Subset checks -----------------------------------------------
    requests = default_subset_requests(config)
    train_ids = set(processed_df.loc[processed_df["original_split"] == "train", "example_id"])

    subset_ids = {}
    for req in requests:
        ids = load_subset_ids(req.name, config)
        subset_ids[req.name] = ids
        check(set(ids) <= train_ids, f"subset '{req.name}' contains only training-split example ids")

    pct_sequence = [f"train_pct{int(p) if float(p).is_integer() else p}" for p in sorted(config.sampling.percentages)]
    nested_ok = True
    for smaller, larger in zip(pct_sequence, pct_sequence[1:]):
        if not (set(subset_ids[smaller]) <= set(subset_ids[larger])):
            nested_ok = False
    check(nested_ok, f"percentage subsets are nested in increasing order: {pct_sequence}")

    abs_sequence = [f"train_abs{n}" for n in sorted(config.sampling.absolute_sizes)]
    nested_abs_ok = True
    for smaller, larger in zip(abs_sequence, abs_sequence[1:]):
        if not (set(subset_ids[smaller]) <= set(subset_ids[larger])):
            nested_abs_ok = False
    check(nested_abs_ok, f"absolute-size subsets are nested in increasing order: {abs_sequence}")

    manifest_path = config.paths.subsets_dir / "manifest.json"
    check(manifest_path.exists(), f"subset manifest exists at {manifest_path}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED.")
        sys.exit(1)
    print("All integrity checks passed.")


if __name__ == "__main__":
    main()
