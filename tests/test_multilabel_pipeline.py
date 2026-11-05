"""Tests for the multi-label data pipeline (emotion_ait.multilabel).

Synthetic tests always run; tests against the real processed data / manifests
are skipped when those files have not been generated locally.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from emotion_ait.multilabel import data as ml_data
from emotion_ait.multilabel.config import load_multilabel_config
from emotion_ait.multilabel.data import (
    TARGET_LABELS,
    get_multilabel_example_ids,
    get_multilabel_targets,
    load_multilabel_dataframe,
    load_multilabel_ladder_ids,
    load_multilabel_ladder_manifest,
    load_multilabel_ladder_subset_dataframe,
)
from emotion_ait.multilabel.preprocessing import (
    compare_with_expected,
    filter_multilabel_option_a,
    multilabel_columns,
    project_labels,
    split_statistics,
    validate_processed_split,
)
from emotion_ait.multilabel.sampling import (
    build_ladder_manifests,
    canonical_stratum_order,
    nested_order,
    validate_ladder_manifests,
    webster_sequence,
    write_ladder_manifests,
)

CONFIG = load_multilabel_config()
NAMES = CONFIG.base.dataset.original_label_names
IDX = {name: i for i, name in enumerate(NAMES)}
HAVE_REAL_DATA = (CONFIG.processed_dir / "train.csv").exists() and (CONFIG.subsets_dir / "manifest.json").exists()
real_data = pytest.mark.skipif(not HAVE_REAL_DATA, reason="multi-label data/manifests not generated locally")


def _raw(rows):
    return pd.DataFrame([{"id": i, "text": f"text {i}", "labels": [IDX[n] for n in names]} for i, names in rows])


# --- preprocessing -----------------------------------------------------------------

@pytest.mark.parametrize("names, vector", [
    (["joy"], [1, 0, 0, 0]),
    (["sadness", "fear"], [0, 1, 0, 1]),
    (["joy", "anger", "surprise"], [1, 0, 1, 0]),
    (["fear", "joy", "anger", "sadness"], [1, 1, 1, 1]),
    (["surprise", "love"], [0, 0, 0, 0]),
    ([], [0, 0, 0, 0]),
])
def test_projection_onto_target_order(names, vector):
    assert project_labels([IDX[n] for n in names], NAMES, TARGET_LABELS) == vector


def test_option_a_keeps_any_target_and_discards_no_target():
    raw = _raw([("a", ["joy"]), ("b", ["surprise", "love"]), ("c", ["sadness", "fear"]),
                ("d", ["joy", "anger", "surprise"]), ("e", ["neutral"])])
    out = filter_multilabel_option_a(raw, "train", NAMES, TARGET_LABELS)
    assert list(out.columns) == multilabel_columns(TARGET_LABELS)
    assert out["example_id"].tolist() == ["a", "c", "d"]
    assert out[TARGET_LABELS].values.tolist() == [[1, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
    assert out["label_combination"].tolist() == ["joy", "sadness+fear", "joy+anger"]
    assert out["original_labels"].tolist() == ["joy", "sadness|fear", "joy|anger|surprise"]
    assert (out["original_split"] == "train").all()
    assert validate_processed_split(out, "train", TARGET_LABELS) == []
    stats = split_statistics(out, len(raw), TARGET_LABELS)
    assert (stats["retained"], stats["discarded"], stats["multi_label"]) == (3, 2, 2)


def test_missing_raw_id_falls_back_to_existing_id_scheme():
    raw = _raw([("", ["joy"])])
    assert filter_multilabel_option_a(raw, "test", NAMES, TARGET_LABELS)["example_id"].tolist() == ["test-0"]


def test_validation_flags_duplicates_and_empty_vectors():
    out = filter_multilabel_option_a(_raw([("a", ["joy"]), ("b", ["fear"])]), "train", NAMES, TARGET_LABELS)
    broken = pd.concat([out, out.iloc[[0]]], ignore_index=True)
    broken.loc[1, TARGET_LABELS] = 0
    problems = " ".join(validate_processed_split(broken, "train", TARGET_LABELS))
    assert "duplicate" in problems and "empty target vector" in problems


def test_expected_count_mismatch_is_reported():
    stats = {"raw": 10, "retained": 4, "multi_label": 1, "positives": {"joy": 2, "sadness": 1, "anger": 1, "fear": 1}}
    problems = compare_with_expected(stats, {"raw": 10, "retained": 5, "positives": {"joy": 2, "fear": 0}}, "train")
    assert problems == ["train: retained = 4, expected 5", "train: positives[fear] = 1, expected 0"]


# --- sampling (synthetic) ------------------------------------------------------------

def _synthetic_train(counts):
    rows = []
    for combination, count in counts.items():
        for i in range(count):
            vector = [int(label in combination.split("+")) for label in TARGET_LABELS]
            rows.append({"example_id": f"{combination}-{i:03d}", "text": "t", **dict(zip(TARGET_LABELS, vector)),
                         "label_combination": combination, "original_labels": combination.replace("+", "|"),
                         "original_split": "train"})
    return pd.DataFrame(rows, columns=multilabel_columns(TARGET_LABELS))


SYNTHETIC = {"joy": 40, "sadness": 30, "anger": 20, "fear": 8, "sadness+fear": 2}


def _synthetic_config(tmp_path, sizes=(10, 25, 50, 100)):
    return dataclasses.replace(CONFIG, ladder_sizes=list(sizes), ladder_seeds=[1, 2], subsets_dir=tmp_path / "ladder")


def test_canonical_stratum_order():
    combos = ["sadness+fear", "fear", "joy", "joy+sadness", "anger", "sadness"]
    assert canonical_stratum_order(combos, TARGET_LABELS) == ["joy", "sadness", "anger", "fear", "joy+sadness", "sadness+fear"]


def test_webster_allocation_is_proportional_monotone_and_exact():
    sizes = {"a": 50, "b": 30, "c": 15, "d": 5}
    sequence = webster_sequence(sizes, list(sizes))
    assert Counter(sequence) == Counter(sizes)
    previous = Counter()
    for n in range(1, 101):
        counts = Counter(sequence[:n])
        assert sum(counts.values()) == n
        assert all(counts[s] >= previous[s] for s in sizes)  # house-monotone
        assert all(abs(counts[s] - n * sizes[s] / 100) < 1 for s in sizes)
        previous = counts
    assert Counter(sequence[:20]) == {"a": 10, "b": 6, "c": 3, "d": 1}


def test_ladder_is_exact_nested_complete_and_deterministic(tmp_path):
    train = _synthetic_train(SYNTHETIC)
    config = _synthetic_config(tmp_path)
    manifests = build_ladder_manifests(train, config)
    validate_ladder_manifests(manifests, train, config)
    for seed, per_size in manifests.items():
        sizes = sorted(per_size)
        for small, large in zip(sizes, sizes[1:]):
            assert set(per_size[small]["example_ids"]) <= set(per_size[large]["example_ids"])
        assert all(per_size[n]["actual_size"] == n for n in sizes)
        assert sorted(per_size[100]["example_ids"]) == sorted(train["example_id"])
        assert all(min(per_size[n]["positive_counts"].values()) > 0 for n in sizes)
    shuffled = train.sample(frac=1.0, random_state=0).reset_index(drop=True)
    shuffled_manifests = build_ladder_manifests(shuffled, config)
    for seed in manifests:
        for size in manifests[seed]:
            for key in ("example_ids", "targets", "positive_counts", "combination_counts"):
                assert shuffled_manifests[seed][size][key] == manifests[seed][size][key]
    assert build_ladder_manifests(train, config) == manifests
    assert nested_order(shuffled, 1, TARGET_LABELS) == nested_order(train, 1, TARGET_LABELS)
    assert nested_order(train, 1, TARGET_LABELS) != nested_order(train, 2, TARGET_LABELS)
    assert manifests[1][50]["combination_counts"] == manifests[2][50]["combination_counts"]


def test_ladder_rejects_wrong_full_size_and_leaked_ids(tmp_path):
    train = _synthetic_train(SYNTHETIC)
    with pytest.raises(ValueError):
        build_ladder_manifests(train, _synthetic_config(tmp_path, sizes=(10, 99)))
    config = _synthetic_config(tmp_path)
    manifests = build_ladder_manifests(train, config)
    manifests[1][10]["example_ids"][0] = "validation-id"
    with pytest.raises(ValueError, match="outside the training partition"):
        validate_ladder_manifests(manifests, train, config)


def test_written_manifests_load_through_shared_api(tmp_path, monkeypatch):
    train = _synthetic_train(SYNTHETIC)
    config = dataclasses.replace(_synthetic_config(tmp_path), processed_dir=tmp_path / "processed")
    config.processed_dir.mkdir()
    train.to_csv(config.processed_dir / "train.csv", index=False)
    write_ladder_manifests(build_ladder_manifests(train, config), config)
    manifest = json.loads((config.subsets_dir / "seed_1" / "n25.json").read_text())
    subset = load_multilabel_ladder_subset_dataframe(1, 25, config)
    assert get_multilabel_example_ids(subset) == manifest["example_ids"]
    assert get_multilabel_targets(subset).tolist() == manifest["targets"]


# --- real data (skipped if not generated) -----------------------------------------------

@real_data
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_real_counts_match_expected(split):
    df = load_multilabel_dataframe(split)
    expected = CONFIG.expected_counts[split]
    stats = split_statistics(df, expected["raw"], TARGET_LABELS)
    assert compare_with_expected(stats, expected, split) == []
    assert validate_processed_split(df, split, TARGET_LABELS) == []


@real_data
def test_real_partitions_are_disjoint():
    ids = {s: set(load_multilabel_dataframe(s)["example_id"]) for s in CONFIG.splits}
    assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])


@real_data
def test_real_ladder_manifests_are_valid_and_complete():
    train = load_multilabel_dataframe("train")
    manifests = {seed: {size: load_multilabel_ladder_manifest(seed, size) for size in CONFIG.ladder_sizes}
                 for seed in CONFIG.ladder_seeds}
    validate_ladder_manifests(manifests, train, CONFIG)
    held_out = set(load_multilabel_dataframe("validation")["example_id"]) | set(load_multilabel_dataframe("test")["example_id"])
    for seed in CONFIG.ladder_seeds:
        assert len(set(manifests[seed][4883]["example_ids"])) == 4883
        for size in CONFIG.ladder_sizes:
            assert not set(manifests[seed][size]["example_ids"]) & held_out
    # Regenerating gives byte-identical manifests (determinism).
    assert build_ladder_manifests(train, CONFIG) == manifests


@real_data
def test_real_stratification_tracks_full_training_distribution():
    train = load_multilabel_dataframe("train")
    share = train["label_combination"].value_counts()
    for seed in CONFIG.ladder_seeds:
        for size in CONFIG.ladder_sizes:
            m = load_multilabel_ladder_manifest(seed, size)
            for combination, count in m["combination_counts"].items():
                assert abs(count - size * share[combination] / len(train)) < 1


@real_data
def test_shared_loader_is_model_agnostic_and_deterministic():
    # "MDL" and "LSTM" call the same loader; they must get identical inputs.
    mdl_view = load_multilabel_ladder_subset_dataframe(43, 500)
    lstm_view = load_multilabel_ladder_subset_dataframe(43, 500)
    pd.testing.assert_frame_equal(mdl_view, lstm_view)
    assert get_multilabel_example_ids(mdl_view) == load_multilabel_ladder_ids(43, 500)
    targets = get_multilabel_targets(mdl_view)
    assert targets.shape == (500, 4) and targets.dtype == np.int64 and set(np.unique(targets)) <= {0, 1}
    assert list(mdl_view.columns) == multilabel_columns(TARGET_LABELS)
    assert ml_data.TARGET_LABELS == ["joy", "sadness", "anger", "fear"] == CONFIG.target_labels
    assert all(isinstance(t, str) for t in mdl_view["text"])
    test_a, test_b = load_multilabel_dataframe("test"), load_multilabel_dataframe("test")
    pd.testing.assert_frame_equal(test_a, test_b)


@real_data
def test_real_ids_are_consistent_with_single_label_track():
    for split in CONFIG.splits:
        single = pd.read_csv(CONFIG.base.paths.processed_dir / f"{split}.csv", dtype={"example_id": str})
        multi = load_multilabel_dataframe(split).set_index("example_id")
        assert set(single["example_id"]) <= set(multi.index)
        rows = multi.loc[single["example_id"]]
        assert (rows["text"].values == single["text"].values).all()
        assert [rows.iloc[i][name] for i, name in enumerate(single["label_name"])] == [1] * len(single)
