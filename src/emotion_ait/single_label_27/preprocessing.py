"""27-class single-label preprocessing: raw GoEmotions -> one emotion per example.

A raw comment is kept iff its *complete* original label set has exactly one
label and that label is not ``neutral``. Everything else is discarded and
counted by reason:

* ``neutral_only`` -- the only label is ``neutral``;
* ``neutral_plus_emotion`` -- ``neutral`` plus at least one emotion;
* ``multi_emotion`` -- two or more emotions (no ``neutral``).

Multi-label comments are never projected onto one class. Official partitions
are preserved and rows keep their raw order. ``example_id`` uses the project
scheme (:func:`emotion_ait.preprocessing._make_example_id`), so a comment has
the same id as in the 4-class and multi-label tracks.

Output columns, in order::

    example_id, text, label, label_name, original_labels, original_split

``label`` is the class index (0..26, see ``SingleLabel27Config.class_names``);
``original_labels`` joins the original label names with ``|`` for auditing
(for retained rows it always equals ``label_name``).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import pandas as pd

from ..preprocessing import _make_example_id
from .config import NEUTRAL

logger = logging.getLogger(__name__)

PROCESSED_COLUMNS = ["example_id", "text", "label", "label_name", "original_labels", "original_split"]
ORIGINAL_LABEL_SEPARATOR = "|"
DISCARD_REASONS = ("neutral_only", "neutral_plus_emotion", "multi_emotion")


def discard_reason(label_names: Sequence[str]) -> str:
    """``""`` if the label set is kept, else one of :data:`DISCARD_REASONS`."""
    names = set(label_names)
    if len(label_names) == 1 and NEUTRAL not in names:
        return ""
    if NEUTRAL in names:
        return "neutral_only" if names == {NEUTRAL} else "neutral_plus_emotion"
    if len(names) == 0:
        raise ValueError("Encountered a comment with no labels.")
    return "multi_emotion"


def filter_single_label_27(
    raw_df: pd.DataFrame, split: str, original_label_names: Sequence[str], class_names: Sequence[str]
) -> pd.DataFrame:
    """Apply the 27-class single-label policy to one raw split."""
    if "labels" not in raw_df.columns or "text" not in raw_df.columns:
        raise KeyError("Expected 'text' and 'labels' columns in the raw dataframe.")
    class_index = {name: i for i, name in enumerate(class_names)}
    has_id = "id" in raw_df.columns
    rows = []
    for row_index, row in enumerate(raw_df.itertuples(index=False)):
        label_ids = [int(i) for i in getattr(row, "labels")]
        if any(i < 0 or i >= len(original_label_names) for i in label_ids):
            raise ValueError(f"Label index out of range in {label_ids}.")
        names = [original_label_names[i] for i in label_ids]
        if discard_reason(names):
            continue
        rows.append({
            "example_id": _make_example_id(getattr(row, "id") if has_id else None, split, row_index),
            "text": getattr(row, "text"),
            "label": class_index[names[0]],
            "label_name": names[0],
            "original_labels": ORIGINAL_LABEL_SEPARATOR.join(names),
            "original_split": split,
        })
    out = pd.DataFrame(rows, columns=PROCESSED_COLUMNS)
    logger.info("Split '%s': retained %d / %d rows (exactly one non-neutral label)", split, len(out), len(raw_df))
    return out


def split_statistics(
    processed: pd.DataFrame, raw_df: pd.DataFrame, original_label_names: Sequence[str], class_names: Sequence[str]
) -> Dict[str, object]:
    reasons = [discard_reason([original_label_names[int(i)] for i in labels]) for labels in raw_df["labels"]]
    counts = processed["label_name"].value_counts()
    classes = {name: int(counts.get(name, 0)) for name in class_names}
    present = {k: v for k, v in classes.items() if v > 0}
    return {
        "raw": int(len(raw_df)),
        "retained": int(len(processed)),
        "discarded": {reason: reasons.count(reason) for reason in DISCARD_REASONS},
        "classes": classes,
        "classes_present": len(present),
        "min_class": min(present, key=present.get) if present else None,
        "min_count": min(present.values()) if present else 0,
        "max_class": max(present, key=present.get) if present else None,
        "max_count": max(present.values()) if present else 0,
        "imbalance_ratio": (max(present.values()) / min(present.values())) if present else None,
    }


def validate_processed_split(processed: pd.DataFrame, split: str, class_names: Sequence[str]) -> List[str]:
    """Structural invariants of one processed split; returns problems (empty = ok)."""
    problems = []
    if list(processed.columns) != PROCESSED_COLUMNS:
        return [f"{split}: columns {list(processed.columns)} != {PROCESSED_COLUMNS}"]
    if processed["example_id"].duplicated().any():
        problems.append(f"{split}: {processed['example_id'].duplicated().sum()} duplicate example ids")
    if (processed["original_split"] != split).any():
        problems.append(f"{split}: rows from another partition")
    if not processed["label"].between(0, len(class_names) - 1).all():
        problems.append(f"{split}: label index outside 0..{len(class_names) - 1}")
    elif [class_names[i] for i in processed["label"]] != processed["label_name"].tolist():
        problems.append(f"{split}: label and label_name disagree")
    if (processed["label_name"] == NEUTRAL).any():
        problems.append(f"{split}: neutral rows retained")
    originals = processed["original_labels"].str.split(ORIGINAL_LABEL_SEPARATOR, regex=False)
    if (originals.str.len() != 1).any():
        problems.append(f"{split}: {(originals.str.len() != 1).sum()} rows with more than one original label")
    if (processed["original_labels"] != processed["label_name"]).any():
        problems.append(f"{split}: original_labels differ from label_name")
    return problems


def compare_with_expected(stats: Dict[str, object], expected: Dict[str, object], split: str) -> List[str]:
    """Discrepancies between computed and expected counts (empty = match)."""
    problems = []
    for key in ("raw", "retained"):
        if key in expected and int(expected[key]) != stats[key]:
            problems.append(f"{split}: {key} = {stats[key]}, expected {expected[key]}")
    for group in ("discarded", "classes"):
        for name, count in (expected.get(group) or {}).items():
            if stats[group].get(name) != int(count):
                problems.append(f"{split}: {group}[{name}] = {stats[group].get(name)}, expected {count}")
    return problems
