"""Filtering of raw GoEmotions rows down to a 4-class single-label subset.

Pipeline contract
------------------
Input: a raw GoEmotions dataframe (one row per original comment) with at
least ``text`` (str), ``labels`` (list[int] indices into the 28-name
original label list) and ``original_split`` columns, plus an optional
stable ``id`` column.

Output: a dataframe with exactly the columns ``example_id``, ``text``,
``label`` (int, 0-3), ``label_name`` (str), ``original_split`` (str),
containing only comments whose *complete* original label set was a
single label, and where that label is one of the four target emotions
(anger, fear, joy, sadness).
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

PROCESSED_COLUMNS = ["example_id", "text", "label", "label_name", "original_split"]


def _make_example_id(raw_id: object, split: str, row_index: int) -> str:
    """Build a stable example id.

    Prefers the dataset's own stable id (GoEmotions ``id`` column, a
    Reddit comment id) when present and non-empty. Otherwise falls back
    to a deterministic id derived from the original split name and the
    original row index within that split -- *never* the comment text,
    since text is not guaranteed unique and should not be treated as an
    identifier.
    """
    if raw_id is not None and str(raw_id).strip() != "" and str(raw_id).lower() != "nan":
        return str(raw_id)
    return f"{split}-{row_index}"


def filter_single_label_target(
    raw_df: pd.DataFrame,
    split: str,
    config: Config,
) -> pd.DataFrame:
    """Filter one raw split down to single-label, target-emotion rows.

    Filtering rules (applied in order):

    1. Keep a row only if its complete original ``labels`` list has
       length exactly 1 (single-label comments only; comments annotated
       with a target emotion *plus* any other label, including
       ``neutral``, are excluded).
    2. Keep the row only if that single label is one of the four target
       emotions (anger, fear, joy, sadness).
    3. Map the retained label name to the fixed integer id defined in
       ``configs/default.yaml`` (``target_task.label_mapping``).
    """
    if "labels" not in raw_df.columns:
        raise KeyError("Expected a 'labels' column in the raw dataframe.")

    label_names = config.dataset.original_label_names
    label_mapping = config.target_task.label_mapping
    target_names = set(label_mapping.keys())

    def _resolve_label_name(label_ids: List[int]) -> str:
        if len(label_ids) != 1:
            return ""
        idx = label_ids[0]
        if idx < 0 or idx >= len(label_names):
            raise ValueError(f"Label index {idx} out of range for 28 original GoEmotions labels.")
        return label_names[idx]

    rows = []
    for row_index, row in enumerate(raw_df.itertuples(index=False)):
        label_ids = list(getattr(row, "labels"))
        if len(label_ids) != 1:
            continue  # rule 1: exclude multi-label (and zero-label) comments
        label_name = _resolve_label_name(label_ids)
        if label_name not in target_names:
            continue  # rules 2/4: exclude comments outside the 4 target classes
        example_id = _make_example_id(getattr(row, "id", None), split, row_index)
        rows.append(
            {
                "example_id": example_id,
                "text": getattr(row, "text"),
                "label": label_mapping[label_name],
                "label_name": label_name,
                "original_split": split,
            }
        )

    result = pd.DataFrame(rows, columns=PROCESSED_COLUMNS)
    logger.info(
        "Split '%s': retained %d / %d rows after single-label + target-class filtering",
        split,
        len(result),
        len(raw_df),
    )
    return result


def process_all_splits(raw_frames: Dict[str, pd.DataFrame], config: Config) -> pd.DataFrame:
    """Apply :func:`filter_single_label_target` to every configured split
    and concatenate the results, preserving the original split labels.
    """
    processed = []
    for split in config.dataset.splits:
        if split not in raw_frames:
            raise KeyError(f"Missing raw split '{split}'.")
        processed.append(filter_single_label_target(raw_frames[split], split, config))
    combined = pd.concat(processed, ignore_index=True)

    duplicate_ids = combined["example_id"][combined["example_id"].duplicated()].unique()
    if len(duplicate_ids) > 0:
        raise ValueError(
            f"Found {len(duplicate_ids)} duplicate example_id values after processing "
            f"all splits (e.g. {list(duplicate_ids[:5])}). Example ids must be unique "
            "across the whole dataset."
        )
    return combined


def class_counts_per_split(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Return a (split x label_name) pivot table of retained example counts.

    Used both by ``scripts/prepare_data.py`` (to document real counts, as
    required) and by ``scripts/verify_data.py`` (as an integrity check).
    """
    counts = (
        processed_df.groupby(["original_split", "label_name"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    counts["total"] = counts.sum(axis=1)
    return counts
