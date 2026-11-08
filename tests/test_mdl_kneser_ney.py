"""Tests for the 4-class Kneser-Ney MDL (emotion_ait.mdl_kneser_ney)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from emotion_ait.mdl_kneser_ney import (
    FOUR_CLASS_NAMES,
    KN_CONFIGS,
    MDLKneserNey4Classifier,
    kn_config_summary,
)
from emotion_ait.single_label_27.mdl import MDL27Classifier


def _df() -> pd.DataFrame:
    return pd.DataFrame([
        {"text": "so angry right now", "label": 0},
        {"text": "angry and furious", "label": 0},
        {"text": "i am so scared", "label": 1},
        {"text": "scared of the dark", "label": 1},
        {"text": "so happy today", "label": 2},
        {"text": "happy happy day", "label": 2},
        {"text": "i really miss her", "label": 3},
        {"text": "so sad and lonely", "label": 3},
    ])


def test_four_class_label_set_and_kn_defaults():
    model = MDLKneserNey4Classifier()
    assert model.class_names == ["anger", "fear", "joy", "sadness"] == FOUR_CLASS_NAMES
    assert (model.n, model.smoothing, model.kn_backoff, model.use_prior) == (2, "kneser_ney", "class", False)


def test_rejects_labels_outside_four_classes():
    bad = pd.DataFrame([{"text": "hello there", "label": 4}])
    with pytest.raises(ValueError):
        MDLKneserNey4Classifier().fit(bad)


@pytest.mark.parametrize("name,n,backoff", KN_CONFIGS)
def test_scores_are_four_wide_finite_and_predictions_valid(name, n, backoff):
    model = MDLKneserNey4Classifier(n=n, kn_backoff=backoff).fit(_df())
    texts = ["so angry", "unseen words entirely", "happy day"]
    scores = model.predict_scores(texts)
    assert scores.shape == (3, 4)
    assert np.isfinite(scores).all()
    assert model.predict(texts) == list(np.argmin(scores, axis=1))
    assert set(model.predict(texts)) <= {0, 1, 2, 3}


@pytest.mark.parametrize("n,backoff", [(2, "class"), (2, "shared"), (3, "class")])
def test_distributions_are_proper(n, backoff):
    model = MDLKneserNey4Classifier(n=n, kn_backoff=backoff).fit(_df())
    vocab = model.vocabulary + ["</s>", "<unk>"]  # vocabulary excludes the two special prediction tokens
    context = ("<s>",) * (n - 1)
    for c in range(4):
        total = sum(model._kn_probability(context, w, c) for w in vocab)
        assert math.isclose(total, 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("n,backoff", [(2, "class"), (2, "shared"), (3, "class")])
def test_identical_to_27_class_kn_on_same_labels(n, backoff):
    # Same implementation: on labels 0..3 the 4-class bits equal the 27-class model's bits for those classes.
    df = _df()
    four = MDLKneserNey4Classifier(n=n, kn_backoff=backoff).fit(df)
    with pytest.warns(UserWarning):  # 23 classes have no training data in the 27-class model
        big = MDL27Classifier(n=n, smoothing="kneser_ney", kn_backoff=backoff).fit(df)
    texts = ["so angry", "i miss the dark", "happy happy"]
    for small_row, big_row in zip(four.description_lengths(texts), big.description_lengths(texts)):
        assert set(small_row) == {0, 1, 2, 3}
        for c in range(4):
            assert math.isclose(small_row[c], big_row[c], rel_tol=1e-12)


def test_config_summary():
    assert kn_config_summary(3, "class") == {"kn_backoff": "class", "kn_discount": "estimated", "n": 3, "smoothing": "kneser_ney"}
