"""Tests for the 27-class bigram Naive Bayes classifier (emotion_ait.single_label_27.naive_bayes)."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emotion_ait.mdl_classifier import MDLClassifier
from emotion_ait.single_label_27.data import CLASS_NAMES, LABEL_TO_INDEX
from emotion_ait.single_label_27.mdl import MDL27Classifier
from emotion_ait.single_label_27.naive_bayes import BigramNaiveBayes27

REPO_ROOT = Path(__file__).resolve().parents[1]
HAVE_REAL_DATA = (REPO_ROOT / "data" / "subsets" / "single_label_27_ladder" / "manifest.json").exists()


def _all_classes_df(per_class: int = 2) -> pd.DataFrame:
    rows = [{"text": f"i feel {name} {name} today", "label": i} for i, name in enumerate(CLASS_NAMES) for _ in range(per_class)]
    # Imbalance so the MAP prior matters.
    rows += [{"text": "i feel great", "label": LABEL_TO_INDEX["admiration"]}] * 6
    return pd.DataFrame(rows)


SMALL = pd.DataFrame([{"text": "a b", "label": 0}, {"text": "a b", "label": 0}, {"text": "a c", "label": 1}])
TEXTS = ["i feel joy today", "i feel great", "zzz unseen", "", "grief grief", "!!!"]


# --- counts, probabilities, likelihoods --------------------------------------------------

def test_class_specific_bigram_counts():
    with pytest.warns(UserWarning):
        model = BigramNaiveBayes27().fit(SMALL)
    assert model._ngram_counts[0] == Counter({("<s>", "a"): 2, ("a", "b"): 2, ("b", "</s>"): 2})
    assert model._ngram_counts[1] == Counter({("<s>", "a"): 1, ("a", "c"): 1, ("c", "</s>"): 1})
    assert model._context_counts[0] == Counter({("<s>",): 2, ("a",): 2, ("b",): 2})
    assert all(not model._ngram_counts[c] for c in range(2, 27))


def test_add_one_probabilities_by_hand():
    with pytest.warns(UserWarning):
        model = BigramNaiveBayes27().fit(SMALL)
    v = model._prediction_vocab_size
    assert v == 5  # {a, b, c} + </s> + <unk>
    assert model.probability(("a",), "b", 0) == pytest.approx((2 + 1) / (2 + v))
    assert model.probability(("a",), "c", 0) == pytest.approx((0 + 1) / (2 + v))  # unseen bigram, seen context
    assert model.probability(("zzz",), "b", 0) == pytest.approx(1 / v)  # unseen context


@pytest.mark.parametrize("k", [1.0, 0.1])
def test_probabilities_are_normalised(k):
    model = BigramNaiveBayes27(k=k).fit(_all_classes_df())
    vocab = model.vocabulary + ["</s>", "<unk>"]
    for class_id in (0, 17, 26):
        for context in (("<s>",), ("feel",), ("never-seen",)):
            probabilities = [model.probability(context, w, class_id) for w in vocab]
            assert min(probabilities) > 0 and sum(probabilities) == pytest.approx(1.0)


def test_log_likelihood_by_hand():
    with pytest.warns(UserWarning):
        model = BigramNaiveBayes27().fit(SMALL)
    # "a b" under class 0: P(a|<s>) = 3/7, P(b|a) = 3/7, P(</s>|b) = 3/7.
    ll = model.class_log_likelihoods(["a b"])[0]
    assert ll[0] == pytest.approx(3 * math.log2(3 / 7))
    # class 1: P(a|<s>) = 2/6, P(b|a) = 1/6, P(</s>|b) unseen context = 1/5.
    assert ll[1] == pytest.approx(math.log2(2 / 6) + math.log2(1 / 6) + math.log2(1 / 5))


def test_class_priors_from_training_labels_only():
    train = _all_classes_df()
    validation = pd.DataFrame([{"text": "joy", "label": 17}] * 100)
    model = BigramNaiveBayes27(variant="map").fit(train, validation_data=validation)
    assert sum(model.class_priors.values()) == pytest.approx(1.0)
    assert model.class_priors[0] == pytest.approx(8 / 60) and model.class_priors[17] == pytest.approx(2 / 60)
    assert model.log_priors()[0] == pytest.approx(math.log2(8 / 60))


# --- decision rules --------------------------------------------------------------------

def test_ml_and_map_scores_and_predictions():
    train = _all_classes_df()
    ml = BigramNaiveBayes27(variant="ml").fit(train)
    map_ = BigramNaiveBayes27(variant="map").fit(train)
    np.testing.assert_allclose(ml.predict_scores(TEXTS), ml.class_log_likelihoods(TEXTS))
    np.testing.assert_allclose(map_.predict_scores(TEXTS), map_.class_log_likelihoods(TEXTS) + map_.log_priors())
    for model in (ml, map_):
        predictions = model.predict(TEXTS)
        assert predictions == [int(np.argmax(row)) for row in model.predict_scores(TEXTS)]
        assert all(0 <= p < 27 for p in predictions)
    # An uninformative text goes to the most frequent class under MAP only.
    assert map_.predict(["completely unrelated words"]) == [0]


def test_ties_go_to_lowest_class_index_and_predictions_are_deterministic():
    train = pd.DataFrame([{"text": "same words", "label": c} for c in (20, 5, 11)])
    with pytest.warns(UserWarning):
        model = BigramNaiveBayes27().fit(train)
    assert model.predict(["same words", "other"]) == [5, 5]
    a = BigramNaiveBayes27().fit(_all_classes_df())
    b = BigramNaiveBayes27().fit(_all_classes_df().sample(frac=1.0, random_state=1))
    assert a.predict(TEXTS) == b.predict(TEXTS) and a.predict(TEXTS) == a.predict(TEXTS)


def test_missing_classes_have_models_and_are_never_predicted():
    with pytest.warns(UserWarning, match="no training examples"):
        model = BigramNaiveBayes27(variant="map").fit(SMALL)
    assert len(model._ngram_counts) == 27 and len(model.missing_classes) == 25
    assert np.isfinite(model.class_log_likelihoods(TEXTS)).all()
    scores = model.predict_scores(TEXTS)
    assert np.isneginf(scores[:, model.missing_classes]).all()
    assert set(model.predict(TEXTS)) <= {0, 1}


def test_empty_and_oov_texts():
    model = BigramNaiveBayes27().fit(_all_classes_df())
    ll = model.class_log_likelihoods(["", "zzz", "qqq"])
    assert np.isfinite(ll).all()
    np.testing.assert_array_equal(ll[1], ll[2])  # distinct OOV words both map to <unk>
    v = model._prediction_vocab_size
    # Empty text = one </s> after <s>: P(</s> | <s>) = (0 + 1) / (count(<s>) + V) for every class.
    expected = [math.log2(1 / (model._context_counts[c][("<s>",)] + v)) for c in range(27)]
    np.testing.assert_allclose(ll[0], expected)


def test_vocabulary_from_training_data_only():
    validation = pd.DataFrame([{"text": "brand new words", "label": 3}])
    model = BigramNaiveBayes27().fit(_all_classes_df(), validation_data=validation)
    assert "brand" not in model.vocabulary and "<s>" not in model.vocabulary
    assert model._prediction_vocab_size == len(model.vocabulary) + 2


def test_invalid_configuration_and_labels():
    for kwargs in ({"variant": "bayes"}, {"k": 0}, {"n": 0}):
        with pytest.raises(ValueError):
            BigramNaiveBayes27(**kwargs)
    with pytest.raises(ValueError):
        BigramNaiveBayes27().fit(pd.DataFrame([{"text": "x", "label": 27}]))


# --- equivalence with conditional MDL ----------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("k", [1.0, 0.1])
def test_ml_equals_conditional_mdl_exactly(n, k):
    train = _all_classes_df()
    nb = BigramNaiveBayes27(n=n, k=k, variant="ml").fit(train)
    mdl = MDL27Classifier(n=n, k=k).fit(train)
    base = MDLClassifier(n=n, k=k).fit(train)
    # log2 P(x|e) is exactly -L(x|M_e), not just approximately.
    np.testing.assert_array_equal(nb.class_log_likelihoods(TEXTS), -mdl.predict_scores(TEXTS))
    assert nb.predict(TEXTS) == mdl.predict(TEXTS) == base.predict(TEXTS)


def test_map_equals_mdl_with_class_prior():
    train = _all_classes_df()
    nb = BigramNaiveBayes27(variant="map").fit(train)
    mdl = MDL27Classifier(use_prior=True).fit(train)
    np.testing.assert_allclose(nb.predict_scores(TEXTS), -mdl.predict_scores(TEXTS))
    assert nb.predict(TEXTS) == mdl.predict(TEXTS)


def test_save_load_round_trip(tmp_path):
    model = BigramNaiveBayes27(variant="map").fit(_all_classes_df())
    model.save(tmp_path / "nb.pkl")
    restored = BigramNaiveBayes27.load(tmp_path / "nb.pkl")
    np.testing.assert_array_equal(restored.predict_scores(TEXTS), model.predict_scores(TEXTS))


@pytest.mark.skipif(not HAVE_REAL_DATA, reason="27-class data/manifests not generated locally")
def test_real_subset_ml_matches_mdl_and_runner_gives_identical_ids():
    from emotion_ait.single_label_27.config import load_single_label_27_config
    from emotion_ait.single_label_27.data import load_single_label_27_dataframe
    from emotion_ait.single_label_27.experiments import run_single_label_27_ladder

    config = load_single_label_27_config()
    validation = load_single_label_27_dataframe("validation")["text"].tolist()[:300]
    seen = {}

    def spy(tag, factory):
        def build(seed):
            model = factory()
            original = model.fit

            def fit(train_data, validation_data=None):
                seen[tag] = train_data["example_id"].tolist()
                original(train_data, validation_data)
                seen[tag + "_pred"] = model.predict(validation)
                return model

            model.fit = fit
            return model, {}
        return build

    import tempfile, dataclasses
    with tempfile.TemporaryDirectory() as tmp:
        base = dataclasses.replace(config.base, paths=dataclasses.replace(config.base.paths, results_dir=Path(tmp)))
        cfg = dataclasses.replace(config, base=base)
        run_single_label_27_ladder(spy("nb", lambda: BigramNaiveBayes27()), "nb_test", cfg, sizes=[500], seeds=[42])
        run_single_label_27_ladder(spy("mdl", lambda: MDL27Classifier()), "mdl_test", cfg, sizes=[500], seeds=[42])
    assert seen["nb"] == seen["mdl"]
    assert seen["nb_pred"] == seen["mdl_pred"]
