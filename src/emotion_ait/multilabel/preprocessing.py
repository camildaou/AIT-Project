"""Option A multi-label preprocessing: raw GoEmotions -> four binary targets.

For each raw comment with original label set ``L``:

* project ``L`` onto the targets ``[joy, sadness, anger, fear]`` (in that
  order) -> a 4-dim 0/1 vector; every other original label is ignored;
* keep the comment iff the vector has at least one 1 (any number of
  target emotions is allowed);
* discard it otherwise.

Official partitions are preserved (a row's ``original_split`` is the split
it came from) and rows keep their raw order. ``example_id`` uses the
single-label track's scheme (:func:`emotion_ait.preprocessing._make_example_id`),
so the same comment has the same id in both tracks.

Output columns, in order::

    example_id, text, joy, sadness, anger, fear,
    label_combination, original_labels, original_split

``label_combination`` joins the positive targets with ``+`` in target
order (e.g. ``sadness+fear``); ``original_labels`` joins *all* original
label names with ``|`` in their original order, for auditing.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import pandas as pd

from ..preprocessing import _make_example_id

logger = logging.getLogger(__name__)

COMBINATION_SEPARATOR = "+"
ORIGINAL_LABEL_SEPARATOR = "|"


def multilabel_columns(target_labels: Sequence[str]) -> List[str]:
    return ["example_id", "text", *target_labels, "label_combination", "original_labels", "original_split"]


def project_labels(label_ids: Sequence[int], original_label_names: Sequence[str], target_labels: Sequence[str]) -> List[int]:
    """Project original GoEmotions label indices onto the 0/1 target vector."""
    names = set()
    for idx in label_ids:
        idx = int(idx)
        if idx < 0 or idx >= len(original_label_names):
            raise ValueError(f"Label index {idx} out of range for {len(original_label_names)} original labels.")
        names.add(original_label_names[idx])
    return [int(label in names) for label in target_labels]


def combination_name(vector: Sequence[int], target_labels: Sequence[str]) -> str:
    """``"sadness+fear"`` for ``[0, 1, 0, 1]``; ``""`` for an all-zero vector."""
    return COMBINATION_SEPARATOR.join(label for label, bit in zip(target_labels, vector) if bit)


def filter_multilabel_option_a(
    raw_df: pd.DataFrame, split: str, original_label_names: Sequence[str], target_labels: Sequence[str]
) -> pd.DataFrame:
    """Apply Option A to one raw split (see module docstring)."""
    if "labels" not in raw_df.columns or "text" not in raw_df.columns:
        raise KeyError("Expected 'text' and 'labels' columns in the raw dataframe.")
    has_id = "id" in raw_df.columns
    rows = []
    for row_index, row in enumerate(raw_df.itertuples(index=False)):
        label_ids = [int(i) for i in getattr(row, "labels")]
        vector = project_labels(label_ids, original_label_names, target_labels)
        if not any(vector):
            continue
        rows.append({
            "example_id": _make_example_id(getattr(row, "id") if has_id else None, split, row_index),
            "text": getattr(row, "text"),
            **dict(zip(target_labels, vector)),
            "label_combination": combination_name(vector, target_labels),
            "original_labels": ORIGINAL_LABEL_SEPARATOR.join(original_label_names[i] for i in label_ids),
            "original_split": split,
        })
    out = pd.DataFrame(rows, columns=multilabel_columns(target_labels))
    logger.info("Split '%s': retained %d / %d rows under Option A", split, len(out), len(raw_df))
    return out


def split_statistics(processed: pd.DataFrame, raw_count: int, target_labels: Sequence[str]) -> Dict[str, object]:
    """Counts used both for the stats file and for checking against ``expected_counts``."""
    cardinality = processed[list(target_labels)].sum(axis=1)
    return {
        "raw": int(raw_count),
        "retained": int(len(processed)),
        "discarded": int(raw_count - len(processed)),
        "multi_label": int((cardinality > 1).sum()),
        "positives": {label: int(processed[label].sum()) for label in target_labels},
        "cardinality": {str(k): int((cardinality == k).sum()) for k in range(1, len(target_labels) + 1)},
        "combinations": {k: int(v) for k, v in processed["label_combination"].value_counts().sort_index().items()},
    }


def validate_processed_split(processed: pd.DataFrame, split: str, target_labels: Sequence[str]) -> List[str]:
    """Structural invariants of one processed split; returns problems (empty = ok)."""
    problems = []
    if list(processed.columns) != multilabel_columns(target_labels):
        problems.append(f"{split}: columns {list(processed.columns)} != {multilabel_columns(target_labels)}")
        return problems
    targets = processed[list(target_labels)]
    if not targets.isin([0, 1]).all().all():
        problems.append(f"{split}: target columns contain values other than 0/1")
    if (targets.sum(axis=1) == 0).any():
        problems.append(f"{split}: {(targets.sum(axis=1) == 0).sum()} rows with an empty target vector")
    if processed["example_id"].duplicated().any():
        problems.append(f"{split}: {processed['example_id'].duplicated().sum()} duplicate example ids")
    if (processed["original_split"] != split).any():
        problems.append(f"{split}: rows from another partition")
    recomputed = [combination_name(v, target_labels) for v in targets.itertuples(index=False)]
    if recomputed != processed["label_combination"].tolist():
        problems.append(f"{split}: label_combination does not match the target columns")
    return problems


def compare_with_expected(stats: Dict[str, object], expected: Dict[str, object], split: str) -> List[str]:
    """Discrepancies between computed and expected counts (empty = match)."""
    problems = []
    for key in ("raw", "retained", "multi_label"):
        if key in expected and int(expected[key]) != stats[key]:
            problems.append(f"{split}: {key} = {stats[key]}, expected {expected[key]}")
    for label, count in (expected.get("positives") or {}).items():
        if stats["positives"].get(label) != int(count):
            problems.append(f"{split}: positives[{label}] = {stats['positives'].get(label)}, expected {count}")
    return problems
