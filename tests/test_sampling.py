"""Tests for emotion_ait.sampling."""

from __future__ import annotations

import dataclasses
import warnings

import pandas as pd
import pytest

from emotion_ait.config import load_config  # noqa: E402
from emotion_ait.sampling import (  # noqa: E402
    SubsetRequest,
    build_subset,
    default_subset_requests,
    generate_all_subsets,
    load_subset_dataframe,
    load_subset_ids,
)


@pytest.fixture
def config():
    return load_config()


def _synthetic_train_df(config, counts: dict) -> pd.DataFrame:
    """Build a synthetic *train*-only processed dataframe with the given
    per-class example counts, e.g. {"joy": 100, "fear": 5, ...}."""
    rows = []
    for label_name, count in counts.items():
        label_id = config.target_task.label_mapping[label_name]
        for i in range(count):
            rows.append(
                {
                    "example_id": f"{label_name}-{i:04d}",
                    "text": f"synthetic {label_name} text {i}",
                    "label": label_id,
                    "label_name": label_name,
                    "original_split": "train",
                }
            )
    return pd.DataFrame(rows)


def test_build_subset_rejects_non_train_rows(config):
    df = _synthetic_train_df(config, {"joy": 10, "fear": 10, "anger": 10, "sadness": 10})
    df.loc[0, "original_split"] = "validation"
    with pytest.raises(ValueError):
        build_subset(df, SubsetRequest(name="x", percentage=50.0), config.target_task.target_emotions, seed=42)


def test_build_subset_is_deterministic_for_a_fixed_seed(config):
    df = _synthetic_train_df(config, {"joy": 100, "fear": 40, "anger": 60, "sadness": 20})
    req = SubsetRequest(name="train_pct20", percentage=20.0)
    s1 = build_subset(df, req, config.target_task.target_emotions, seed=42)
    s2 = build_subset(df, req, config.target_task.target_emotions, seed=42)
    assert s1["example_ids"] == s2["example_ids"]


def test_build_subset_never_includes_more_than_the_available_class_examples(config):
    df = _synthetic_train_df(config, {"joy": 100, "fear": 40, "anger": 60, "sadness": 20})
    req = SubsetRequest(name="train_pct100", percentage=100.0)
    subset = build_subset(df, req, config.target_task.target_emotions, seed=42)
    assert subset["actual_size"] == 220
    assert subset["class_counts"] == {"anger": 60, "fear": 40, "joy": 100, "sadness": 20}


def test_percentage_subsets_are_nested(config):
    df = _synthetic_train_df(config, {"joy": 200, "fear": 80, "anger": 120, "sadness": 40})
    label_names = config.target_task.target_emotions
    percentages = [5, 10, 20, 40, 70, 100]
    subsets = [
        build_subset(df, SubsetRequest(name=f"train_pct{p}", percentage=float(p)), label_names, seed=42)
        for p in percentages
    ]
    for smaller, larger in zip(subsets, subsets[1:]):
        assert set(smaller["example_ids"]) <= set(larger["example_ids"]), (
            f"{smaller['name']} is not nested within {larger['name']}"
        )
        assert smaller["actual_size"] <= larger["actual_size"]


def test_absolute_size_subsets_are_nested(config):
    df = _synthetic_train_df(config, {"joy": 300, "fear": 150, "anger": 200, "sadness": 100})
    label_names = config.target_task.target_emotions
    sizes = [100, 250, 500]
    subsets = [
        build_subset(df, SubsetRequest(name=f"train_abs{n}", absolute_size=n), label_names, seed=42)
        for n in sizes
    ]
    for smaller, larger in zip(subsets, subsets[1:]):
        assert set(smaller["example_ids"]) <= set(larger["example_ids"])


def test_warns_when_a_rare_class_rounds_to_zero(config):
    # 1 fear example: a 5% subset rounds its quota down to 0.
    df = _synthetic_train_df(config, {"joy": 200, "fear": 1, "anger": 100, "sadness": 100})
    with pytest.warns(UserWarning, match="fear"):
        build_subset(df, SubsetRequest(name="train_pct5", percentage=5.0), config.target_task.target_emotions, seed=42)


def test_no_warning_when_stratification_is_feasible(config):
    df = _synthetic_train_df(config, {"joy": 200, "fear": 80, "anger": 100, "sadness": 100})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_subset(df, SubsetRequest(name="train_pct50", percentage=50.0), config.target_task.target_emotions, seed=42)


def test_generate_all_subsets_writes_files_and_manifest(tmp_path, config):
    df = _synthetic_train_df(config, {"joy": 300, "fear": 150, "anger": 200, "sadness": 100})
    processed_df = df.copy()
    tmp_paths = dataclasses.replace(config.paths, subsets_dir=tmp_path / "subsets")
    tmp_config = dataclasses.replace(config, paths=tmp_paths)

    results = generate_all_subsets(processed_df, tmp_config)

    expected_names = {r.name for r in default_subset_requests(tmp_config)}
    assert set(results.keys()) == expected_names
    assert (tmp_paths.subsets_dir / "manifest.json").exists()
    for name in expected_names:
        assert (tmp_paths.subsets_dir / f"{name}.json").exists()

    ids = load_subset_ids("train_pct10", tmp_config)
    assert ids == sorted(ids)

    subset_df = load_subset_dataframe("train_pct10", processed_df, tmp_config)
    assert set(subset_df["example_id"]) == set(ids)


def test_repeated_runs_with_same_seed_reproduce_the_same_subsets(tmp_path, config):
    df = _synthetic_train_df(config, {"joy": 300, "fear": 150, "anger": 200, "sadness": 100})

    paths_a = dataclasses.replace(config.paths, subsets_dir=tmp_path / "a")
    paths_b = dataclasses.replace(config.paths, subsets_dir=tmp_path / "b")
    config_a = dataclasses.replace(config, paths=paths_a)
    config_b = dataclasses.replace(config, paths=paths_b)

    generate_all_subsets(df, config_a)
    generate_all_subsets(df, config_b)

    for name in {r.name for r in default_subset_requests(config)}:
        assert load_subset_ids(name, config_a) == load_subset_ids(name, config_b)
