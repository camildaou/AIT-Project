"""Tests for the shared 27-class experiment runner and the run_experiment class mapping."""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

from emotion_ait.baselines import MajorityClassBaseline
from emotion_ait.experiments import run_experiment
from emotion_ait.mdl_classifier import MDLClassifier
from emotion_ait.single_label_27.config import load_single_label_27_config
from emotion_ait.single_label_27.data import CLASS_NAMES, INDEX_TO_LABEL, LABEL_TO_INDEX
from emotion_ait.single_label_27.experiments import results_path, run_name, run_single_label_27_ladder
from emotion_ait.single_label_27.preprocessing import PROCESSED_COLUMNS
from emotion_ait.single_label_27.sampling import build_ladder_manifests, write_ladder_manifests

CONFIG = load_single_label_27_config()
WORDS = {"admiration": "great work", "joy": "so happy", "anger": "so angry", "grief": "deeply mourning"}


def _rows(split, per_class):
    return [{"example_id": f"{split}-{name}-{i}", "text": f"{WORDS[name]} {i}", "label": LABEL_TO_INDEX[name],
             "label_name": name, "original_labels": name, "original_split": split}
            for name, count in per_class.items() for i in range(count)]


@pytest.fixture
def tiny_track(tmp_path):
    """A complete miniature 27-class track (data + manifests) under tmp_path."""
    base = dataclasses.replace(CONFIG.base, paths=dataclasses.replace(CONFIG.base.paths, results_dir=tmp_path / "results"))
    config = dataclasses.replace(CONFIG, processed_dir=tmp_path / "processed", subsets_dir=tmp_path / "ladder",
                                 ladder_sizes=[8, 16], ladder_seeds=[1, 2], base=base)
    config.processed_dir.mkdir()
    splits = {"train": {"admiration": 6, "joy": 5, "anger": 4, "grief": 1},
              "validation": {"admiration": 2, "joy": 2, "anger": 1, "grief": 1},
              "test": {"admiration": 2, "joy": 1, "anger": 2, "grief": 1}}
    for split, per_class in splits.items():
        pd.DataFrame(_rows(split, per_class), columns=PROCESSED_COLUMNS).to_csv(config.processed_dir / f"{split}.csv", index=False)
    train = pd.read_csv(config.processed_dir / "train.csv", dtype={"example_id": str})
    config = dataclasses.replace(config, require_all_classes=False)
    write_ladder_manifests(build_ladder_manifests(train, config), config)
    return config


def _mdl(seed):
    model = MDLClassifier(n=2)
    return model, {"n": 2, "k": 1.0, "smoothing": "laplace"}


def test_run_name_prefix_keeps_tracks_apart():
    assert run_name("mdl_n2_k1") == "sl27_mdl_n2_k1" == run_name("sl27_mdl_n2_k1")


def test_runner_scores_all_27_classes_with_shared_evaluation(tiny_track):
    records = run_single_label_27_ladder(_mdl, "mdl_n2_k1", tiny_track)
    assert len(records) == 4
    for record in records:
        assert record.model_name == "sl27_mdl_n2_k1"
        assert list(record.test_per_class) == CLASS_NAMES  # all 27 classes reported, in class order
        assert record.config_summary["track"] == "single_label_27" and "subset_ids_sha256" in record.config_summary
    results = pd.read_csv(results_path(tiny_track, "mdl_n2_k1"))
    assert len(results) == 4 and set(results["train_subset_size"]) == {8, 16}
    predictions = tiny_track.base.paths.results_dir / "predictions" / "sl27_mdl_n2_k1_ladder_n16_2.csv"
    pred = pd.read_csv(predictions)
    assert len(pred) == 6 and set(pred["true_label_name"]) <= set(CLASS_NAMES)
    assert set(pred["predicted_label_name"]) <= set(CLASS_NAMES)


def test_runner_refuses_to_overwrite_and_rejects_unknown_sizes(tiny_track):
    run_single_label_27_ladder(_mdl, "mdl_n2_k1", tiny_track, sizes=[8], seeds=[1])
    with pytest.raises(FileExistsError):
        run_single_label_27_ladder(_mdl, "mdl_n2_k1", tiny_track, sizes=[8], seeds=[1])
    run_single_label_27_ladder(_mdl, "mdl_n2_k1", tiny_track, sizes=[8], seeds=[1], results_tag="debug")
    with pytest.raises(ValueError):
        run_single_label_27_ladder(_mdl, "x", tiny_track, sizes=[10])


def test_both_models_see_identical_training_ids(tiny_track):
    seen = {}

    def spy(tag):
        def build(seed):
            model = MajorityClassBaseline()
            original_fit = model.fit

            def fit(train_data, validation_data=None):
                seen.setdefault(tag, []).append(train_data["example_id"].tolist())
                return original_fit(train_data, validation_data)

            model.fit = fit
            return model, {}
        return build

    run_single_label_27_ladder(spy("mdl"), "spy_mdl", tiny_track)
    run_single_label_27_ladder(spy("lstm"), "spy_lstm", tiny_track)
    assert seen["mdl"] == seen["lstm"] and len(seen["mdl"]) == 4


def test_run_experiment_defaults_to_the_4_class_mapping(tiny_track):
    four = pd.DataFrame([
        {"example_id": f"{s}{i}", "text": t, "label": y, "label_name": INDEX_TO_LABEL[0], "original_split": s}
        for s in ("train", "validation", "test") for i, (t, y) in enumerate([("so angry", 0), ("so happy", 2)])
    ])
    record = run_experiment(MajorityClassBaseline(), "majority", "toy", four, tiny_track.base,
                            train_df=four[four["original_split"] == "train"], save_predictions=False, record_results=False)
    assert list(record.test_per_class) == ["anger", "fear", "joy", "sadness"]


def test_runner_with_mdl27_writes_diagnostics_and_separates_prior_variant(tiny_track):
    from emotion_ait.single_label_27.mdl import MDL27Classifier

    def build(use_prior):
        return lambda seed: (MDL27Classifier(use_prior=use_prior), {"use_prior": use_prior})

    with pytest.warns(UserWarning, match="no training examples"):  # the toy track has only 4 of 27 classes
        plain = run_single_label_27_ladder(build(False), "mdl_n2_k1", tiny_track, sizes=[16], seeds=[1], diagnostics=True)
        prior = run_single_label_27_ladder(build(True), "mdl_prior_n2_k1", tiny_track, sizes=[16], seeds=[1], diagnostics=True)
    assert results_path(tiny_track, "mdl_n2_k1") != results_path(tiny_track, "mdl_prior_n2_k1")
    assert plain[0].model_name == "sl27_mdl_n2_k1" and prior[0].model_name == "sl27_mdl_prior_n2_k1"
    report = json.loads((tiny_track.base.paths.results_dir / "diagnostics" / "sl27_mdl_n2_k1_ladder_n16_1.json").read_text())
    assert set(report) >= {"per_class", "validation", "test", "shared_vocabulary_size", "missing_classes"}
    assert sum(report["test"]["predictions_per_class"].values()) == 6
