"""Tests for the 27-class single-label data pipeline (emotion_ait.single_label_27).

Synthetic tests always run; tests against the real processed data / manifests
are skipped when those files have not been generated locally.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from emotion_ait.single_label_27.config import load_single_label_27_config
from emotion_ait.single_label_27.data import (
    CLASS_NAMES,
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    get_single_label_27_example_ids,
    get_single_label_27_labels,
    load_single_label_27_dataframe,
    load_single_label_27_ladder_ids,
    load_single_label_27_ladder_manifest,
    load_single_label_27_ladder_subset_dataframe,
)
from emotion_ait.single_label_27.preprocessing import (
    PROCESSED_COLUMNS,
    compare_with_expected,
    discard_reason,
    filter_single_label_27,
    split_statistics,
    validate_processed_split,
)
from emotion_ait.single_label_27.sampling import (
    build_ladder_manifests,
    nested_order,
    validate_ladder_manifests,
    write_ladder_manifests,
)

CONFIG = load_single_label_27_config()
NAMES = CONFIG.base.dataset.original_label_names
IDX = {name: i for i, name in enumerate(NAMES)}
HAVE_REAL_DATA = (CONFIG.processed_dir / "train.csv").exists() and (CONFIG.subsets_dir / "manifest.json").exists()
real_data = pytest.mark.skipif(not HAVE_REAL_DATA, reason="27-class data/manifests not generated locally")


def _raw(rows):
    return pd.DataFrame([{"id": i, "text": f"text {i}", "labels": [IDX[n] for n in names]} for i, names in rows])


# --- label mapping -----------------------------------------------------------------

def test_class_mapping_is_the_27_emotions_in_original_order():
    assert len(CLASS_NAMES) == 27 and "neutral" not in CLASS_NAMES
    assert CLASS_NAMES == [n for n in NAMES if n != "neutral"] == CONFIG.class_names
    assert all(INDEX_TO_LABEL[LABEL_TO_INDEX[name]] == name for name in CLASS_NAMES)
    assert sorted(LABEL_TO_INDEX.values()) == list(range(27))
    assert (LABEL_TO_INDEX["admiration"], LABEL_TO_INDEX["grief"], LABEL_TO_INDEX["surprise"]) == (0, 16, 26)


# --- preprocessing -----------------------------------------------------------------

@pytest.mark.parametrize("names, reason", [
    (["joy"], ""), (["gratitude"], ""), (["grief"], ""),
    (["joy", "admiration"], "multi_emotion"), (["sadness", "disappointment"], "multi_emotion"),
    (["neutral"], "neutral_only"), (["neutral", "joy"], "neutral_plus_emotion"),
])
def test_policy_examples_from_the_spec(names, reason):
    assert discard_reason(names) == reason


def test_filter_keeps_exactly_one_non_neutral_label():
    raw = _raw([("a", ["joy"]), ("b", ["joy", "admiration"]), ("c", ["neutral"]), ("d", ["neutral", "joy"]),
                ("e", ["grief"]), ("f", ["sadness", "disappointment"]), ("g", ["gratitude"])])
    out = filter_single_label_27(raw, "train", NAMES, CLASS_NAMES)
    assert list(out.columns) == PROCESSED_COLUMNS
    assert out["example_id"].tolist() == ["a", "e", "g"]
    assert out["label_name"].tolist() == ["joy", "grief", "gratitude"]
    assert out["label"].tolist() == [LABEL_TO_INDEX[n] for n in ("joy", "grief", "gratitude")]
    assert out["original_labels"].tolist() == out["label_name"].tolist()
    assert validate_processed_split(out, "train", CLASS_NAMES) == []
    stats = split_statistics(out, raw, NAMES, CLASS_NAMES)
    assert stats["discarded"] == {"neutral_only": 1, "neutral_plus_emotion": 1, "multi_emotion": 2}
    assert stats["retained"] + sum(stats["discarded"].values()) == len(raw)


def test_validation_flags_duplicates_multi_label_and_neutral_rows():
    out = filter_single_label_27(_raw([("a", ["joy"]), ("b", ["fear"])]), "train", NAMES, CLASS_NAMES)
    broken = pd.concat([out, out.iloc[[0]]], ignore_index=True)
    broken.loc[1, "original_labels"] = "fear|nervousness"
    problems = " ".join(validate_processed_split(broken, "train", CLASS_NAMES))
    assert "duplicate" in problems and "more than one original label" in problems
    assert any("another partition" in p for p in validate_processed_split(out, "test", CLASS_NAMES))


def test_expected_count_mismatch_is_reported():
    stats = {"raw": 10, "retained": 6, "discarded": {"neutral_only": 2}, "classes": {"joy": 4, "grief": 2}}
    problems = compare_with_expected(stats, {"raw": 10, "retained": 7, "classes": {"grief": 1}}, "train")
    assert problems == ["train: retained = 6, expected 7", "train: classes[grief] = 2, expected 1"]


# --- sampling (synthetic) ------------------------------------------------------------

def _synthetic_train(counts):
    rows = [{"example_id": f"{name}-{i:03d}", "text": "t", "label": LABEL_TO_INDEX[name], "label_name": name,
             "original_labels": name, "original_split": "train"}
            for name, count in counts.items() for i in range(count)]
    return pd.DataFrame(rows, columns=PROCESSED_COLUMNS)


SYNTHETIC = {"admiration": 60, "joy": 30, "anger": 20, "grief": 3}


def _synthetic_config(tmp_path, sizes=(30, 60, 113)):
    return dataclasses.replace(CONFIG, ladder_sizes=list(sizes), ladder_seeds=[1, 2], subsets_dir=tmp_path / "ladder")


def test_ladder_is_exact_nested_complete_stratified_and_deterministic(tmp_path):
    train = _synthetic_train(SYNTHETIC)
    config = _synthetic_config(tmp_path)
    manifests = build_ladder_manifests(train, config)
    validate_ladder_manifests(manifests, train, config)
    for per_size in manifests.values():
        assert [per_size[n]["actual_size"] for n in (30, 60, 113)] == [30, 60, 113]
        assert set(per_size[30]["example_ids"]) <= set(per_size[60]["example_ids"]) <= set(per_size[113]["example_ids"])
        assert sorted(per_size[113]["example_ids"]) == sorted(train["example_id"])
        for n in (30, 60):
            for name, count in SYNTHETIC.items():
                assert abs(per_size[n]["class_counts"][name] - n * count / 113) < 1
    shuffled = train.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert nested_order(shuffled, 1, CLASS_NAMES) == nested_order(train, 1, CLASS_NAMES)
    assert nested_order(train, 1, CLASS_NAMES) != nested_order(train, 2, CLASS_NAMES)
    assert build_ladder_manifests(train, config) == manifests


def test_ladder_rejects_missing_classes_wrong_full_size_and_leaks(tmp_path):
    train = _synthetic_train(SYNTHETIC)
    with pytest.raises(ValueError, match="classes missing"):
        config = _synthetic_config(tmp_path, sizes=(5, 113))
        validate_ladder_manifests(build_ladder_manifests(train, config), train, config)
    with pytest.raises(ValueError, match="full training partition"):
        build_ladder_manifests(train, _synthetic_config(tmp_path, sizes=(30, 100)))
    config = _synthetic_config(tmp_path)
    manifests = build_ladder_manifests(train, config)
    manifests[1][30]["example_ids"][0] = "test-id"
    with pytest.raises(ValueError, match="outside the training partition"):
        validate_ladder_manifests(manifests, train, config)


def test_written_manifests_load_through_shared_api(tmp_path):
    train = _synthetic_train(SYNTHETIC)
    config = dataclasses.replace(_synthetic_config(tmp_path), processed_dir=tmp_path / "processed")
    config.processed_dir.mkdir()
    train.to_csv(config.processed_dir / "train.csv", index=False)
    write_ladder_manifests(build_ladder_manifests(train, config), config)
    subset = load_single_label_27_ladder_subset_dataframe(2, 60, config)
    assert get_single_label_27_example_ids(subset) == load_single_label_27_ladder_ids(2, 60, config)
    assert [INDEX_TO_LABEL[i] for i in get_single_label_27_labels(subset)] == load_single_label_27_ladder_manifest(2, 60, config)["labels"]


# --- real data (skipped if not generated) -----------------------------------------------

@real_data
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_real_counts_and_invariants(split):
    df = load_single_label_27_dataframe(split)
    assert validate_processed_split(df, split, CLASS_NAMES) == []
    assert not df["example_id"].duplicated().any()
    assert (df["original_split"] == split).all()
    assert (df["label_name"] != "neutral").all()
    assert (~df["original_labels"].str.contains("|", regex=False)).all()  # exactly one original label
    counts = Counter(df["label_name"])
    expected = CONFIG.expected_counts[split]
    assert len(df) == expected["retained"]
    assert {name: counts.get(name, 0) for name in CLASS_NAMES} == expected["classes"]


@real_data
def test_real_splits_are_disjoint_and_match_raw_source():
    ids = {s: set(load_single_label_27_dataframe(s)["example_id"]) for s in CONFIG.splits}
    assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
    raw = pd.read_parquet(CONFIG.base.paths.raw_dir / "train.parquet")
    source = {str(r.id): [NAMES[int(i)] for i in r.labels] for r in raw.itertuples(index=False)}
    train = load_single_label_27_dataframe("train")
    assert all(source[i] == [name] for i, name in zip(train["example_id"], train["label_name"]))


@real_data
def test_real_ladder_is_valid_and_has_all_classes_everywhere():
    train = load_single_label_27_dataframe("train")
    manifests = {seed: {size: load_single_label_27_ladder_manifest(seed, size) for size in CONFIG.ladder_sizes}
                 for seed in CONFIG.ladder_seeds}
    validate_ladder_manifests(manifests, train, CONFIG)
    held_out = set(load_single_label_27_dataframe("validation")["example_id"]) | set(load_single_label_27_dataframe("test")["example_id"])
    for per_size in manifests.values():
        for size, m in per_size.items():
            assert m["classes_present"] == 27 and not set(m["example_ids"]) & held_out
    assert build_ladder_manifests(train, CONFIG) == manifests  # deterministic regeneration


@real_data
def test_shared_loader_gives_identical_ids_to_both_models():
    mdl_view = load_single_label_27_ladder_subset_dataframe(44, 2000)
    lstm_view = load_single_label_27_ladder_subset_dataframe(44, 2000)
    pd.testing.assert_frame_equal(mdl_view, lstm_view)
    assert get_single_label_27_example_ids(mdl_view) == load_single_label_27_ladder_ids(44, 2000)
    labels = get_single_label_27_labels(mdl_view)
    assert labels.dtype == np.int64 and labels.shape == (2000,) and set(labels) <= set(range(27))


@real_data
def test_real_data_is_consistent_with_the_4_class_track():
    # The 4-class single-label examples are exactly this track's anger/fear/joy/sadness rows.
    for split in CONFIG.splits:
        four = pd.read_csv(CONFIG.base.paths.processed_dir / f"{split}.csv", dtype={"example_id": str})
        df = load_single_label_27_dataframe(split)
        subset = df[df["label_name"].isin(["anger", "fear", "joy", "sadness"])]
        assert set(subset["example_id"]) == set(four["example_id"])
        merged = four.merge(subset, on="example_id", suffixes=("_4", "_27"))
        assert (merged["label_name_4"] == merged["label_name_27"]).all() and (merged["text_4"] == merged["text_27"]).all()
