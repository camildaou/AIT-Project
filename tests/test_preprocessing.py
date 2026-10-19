"""Tests for emotion_ait.preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from emotion_ait.config import load_config  # noqa: E402
from emotion_ait.preprocessing import (  # noqa: E402
    class_counts_per_split,
    filter_single_label_target,
    process_all_splits,
)


@pytest.fixture
def config():
    return load_config()


def _idx(config, name: str) -> int:
    return config.dataset.original_label_names.index(name)


def _raw_df(config, rows):
    """rows: list of (id_or_None, text, label_names: list[str])"""
    data = []
    for raw_id, text, names in rows:
        data.append({"id": raw_id, "text": text, "labels": [_idx(config, n) for n in names]})
    return pd.DataFrame(data, columns=["id", "text", "labels"])


def test_retains_only_single_label_target_classes(config):
    raw_df = _raw_df(
        config,
        [
            ("a1", "I am so happy today", ["joy"]),
            ("a2", "I am furious and sad", ["anger", "sadness"]),  # multi-label -> excluded
            ("a3", "neutral comment", ["neutral"]),  # not a target class -> excluded
            ("a4", "I am scared", ["fear"]),
            ("a5", "no labels at all", []),  # zero labels -> excluded
            ("a6", "happy but also optimistic", ["joy", "optimism"]),  # multi-label incl. target -> excluded
        ],
    )
    result = filter_single_label_target(raw_df, "train", config)

    assert set(result["example_id"]) == {"a1", "a4"}
    assert set(result["label_name"]) == {"joy", "fear"}
    assert list(result.columns) == ["example_id", "text", "label", "label_name", "original_split"]
    assert (result["original_split"] == "train").all()


def test_label_id_mapping_is_consistent_with_config(config):
    raw_df = _raw_df(config, [("a1", "t1", ["anger"]), ("a2", "t2", ["fear"]), ("a3", "t3", ["joy"]), ("a4", "t4", ["sadness"])])
    result = filter_single_label_target(raw_df, "train", config)
    for _, row in result.iterrows():
        assert row["label"] == config.target_task.label_mapping[row["label_name"]]


def test_example_id_prefers_stable_source_id(config):
    raw_df = _raw_df(config, [("stable-123", "t1", ["joy"])])
    result = filter_single_label_target(raw_df, "train", config)
    assert result.iloc[0]["example_id"] == "stable-123"


def test_example_id_falls_back_to_split_and_row_index_when_missing(config):
    raw_df = _raw_df(config, [(None, "t1", ["joy"]), (None, "t2", ["fear"])])
    result = filter_single_label_target(raw_df, "validation", config)
    assert set(result["example_id"]) == {"validation-0", "validation-1"}


def test_process_all_splits_preserves_original_split_and_detects_no_cross_split_duplicates(config):
    raw_frames = {
        "train": _raw_df(config, [("id1", "t1", ["joy"])]),
        "validation": _raw_df(config, [("id2", "t2", ["fear"])]),
        "test": _raw_df(config, [("id3", "t3", ["anger"])]),
    }
    combined = process_all_splits(raw_frames, config)
    assert set(combined["original_split"]) == {"train", "validation", "test"}
    assert combined["example_id"].is_unique


def test_process_all_splits_raises_on_duplicate_example_ids_across_splits(config):
    raw_frames = {
        "train": _raw_df(config, [("dup", "t1", ["joy"])]),
        "validation": _raw_df(config, [("dup", "t2", ["fear"])]),
        "test": _raw_df(config, []),
    }
    with pytest.raises(ValueError):
        process_all_splits(raw_frames, config)


def test_class_counts_per_split(config):
    raw_frames = {
        "train": _raw_df(config, [("1", "t", ["joy"]), ("2", "t", ["joy"]), ("3", "t", ["fear"])]),
        "validation": _raw_df(config, [("4", "t", ["anger"])]),
        "test": _raw_df(config, []),
    }
    combined = process_all_splits(raw_frames, config)
    counts = class_counts_per_split(combined)
    assert counts.loc["train", "joy"] == 2
    assert counts.loc["train", "fear"] == 1
    assert counts.loc["validation", "anger"] == 1
    assert counts.loc["train", "total"] == 3
