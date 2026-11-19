"""Tests for the dataset-level two-part MDL accounting (emotion_ait.two_part_mdl)."""

import math

import pandas as pd
import pytest

from emotion_ait.mdl_classifier import MDLClassifier
from emotion_ait.mdl_kneser_ney import MDLKneserNey4Classifier
from emotion_ait.two_part_mdl import (
    count_entries,
    data_description_length,
    model_description_length,
    two_part_summary,
)

TRAIN = pd.DataFrame(
    {
        "text": ["i am so angry", "angry angry man", "scared of the dark", "so happy today",
                 "happy happy joy", "sad and lonely", "i am sad"],
        "label": [0, 0, 1, 2, 2, 3, 3],
    }
)


def _builders():
    return [
        lambda: MDLClassifier(n=2, k=1.0),
        lambda: MDLClassifier(n=3, k=1.0),
        lambda: MDLKneserNey4Classifier(n=2, kn_backoff="class"),
    ]


@pytest.mark.parametrize("build", _builders())
def test_sufficient_entries_are_the_observed_ngram_types(build):
    model = build().fit(TRAIN)
    entries = count_entries(model)
    for c, tables in model._per_class_models.items():
        assert entries[c]["sufficient"] == len(tables["ngram_counts"])
        assert all(v > 0 for v in tables["ngram_counts"].values())  # zero counts never stored


def test_add1_and_kn_bigram_store_the_same_count_table():
    a = model_description_length(MDLClassifier(n=2, k=1.0).fit(TRAIN))
    k = model_description_length(MDLKneserNey4Classifier(n=2).fit(TRAIN))
    assert a["entries_sufficient"] == k["entries_sufficient"]
    # KN pays only for its extra discount scalars (one per order per class).
    assert k["L_model_bits"] - a["L_model_bits"] == 32 * (k["scalars"] - a["scalars"])
    assert k["entries_stored"] > a["entries_stored"]


def test_model_cost_is_fixed_width_and_paid_once():
    model = MDLClassifier(n=2, k=1.0).fit(TRAIN)
    l16 = model_description_length(model, bits_per_entry=16)["L_model_bits"]
    l32 = model_description_length(model, bits_per_entry=32)["L_model_bits"]
    assert l32 == 2 * l16
    one = two_part_summary(model, TRAIN.text[:1].tolist(), TRAIN.label[:1].tolist())
    all_ = two_part_summary(model, TRAIN.text.tolist(), TRAIN.label.tolist())
    assert one["L_model_bits"] == all_["L_model_bits"]  # does not grow with the number of comments


@pytest.mark.parametrize("build", _builders())
def test_data_cost_uses_the_true_label_model(build):
    model = build().fit(TRAIN)
    scores = model.description_lengths(TRAIN.text.tolist())
    expected = sum(s[y] for s, y in zip(scores, TRAIN.label))
    assert math.isclose(data_description_length(model, TRAIN.text, TRAIN.label), expected)
    s = two_part_summary(model, TRAIN.text.tolist(), TRAIN.label.tolist())
    assert math.isclose(s["L_total_bits"], s["L_model_bits"] + s["L_data_bits"])


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        count_entries(MDLClassifier(n=2))
