"""Tests for the 27-class MDL classifier (emotion_ait.single_label_27.mdl)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emotion_ait.mdl_classifier import MDLClassifier, _laplace_probability
from emotion_ait.single_label_27.data import CLASS_NAMES, LABEL_TO_INDEX
from emotion_ait.single_label_27.mdl import MDL27Classifier

REPO_ROOT = Path(__file__).resolve().parents[1]


def _all_classes_df(per_class: int = 2) -> pd.DataFrame:
    # Each class has a distinctive word plus shared filler, so classes are separable.
    rows = [{"text": f"i feel {name} {name} today", "label": i} for i, name in enumerate(CLASS_NAMES) for _ in range(per_class)]
    return pd.DataFrame(rows)


def _few_classes_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"text": "so happy today", "label": LABEL_TO_INDEX["joy"]},
        {"text": "happy happy day", "label": LABEL_TO_INDEX["joy"]},
        {"text": "so angry now", "label": LABEL_TO_INDEX["anger"]},
        {"text": "deeply mourning", "label": LABEL_TO_INDEX["grief"]},
    ])


# --- models, mapping, vocabulary ---------------------------------------------------

def test_defaults_are_bigram_add_one_without_prior():
    model = MDL27Classifier()
    assert (model.n, model.k, model.smoothing, model.use_prior) == (2, 1.0, "laplace", False)


def test_creates_exactly_27_models_mapped_to_class_indices():
    model = MDL27Classifier().fit(_all_classes_df())
    assert model.class_names == CLASS_NAMES
    assert sorted(model._per_class_models) == list(range(27))
    assert model.missing_classes == []
    for i, name in enumerate(CLASS_NAMES):
        # Each class's model holds exactly the bigrams of its own training texts.
        assert ("i", "feel") in model._per_class_models[i]["ngram_counts"]
        assert (name, "today") in model._per_class_models[i]["ngram_counts"]
        other = CLASS_NAMES[(i + 1) % 27]
        assert (other, "today") not in model._per_class_models[i]["ngram_counts"]


def test_one_shared_vocabulary_from_training_data_only():
    train = _all_classes_df()
    validation = pd.DataFrame([{"text": "completely novel words", "label": 0}])
    model = MDL27Classifier().fit(train, validation_data=validation)
    expected = {"i", "feel", "today", *CLASS_NAMES}
    assert set(model.vocabulary) == expected
    assert model._prediction_vocab_size == len(expected) + 2  # + </s> + <unk>, never <s>
    assert "<s>" not in model.vocabulary and "novel" not in model.vocabulary
    reference = MDL27Classifier().fit(train)
    assert model.predict_scores(["a b", "joy"]).tolist() == reference.predict_scores(["a b", "joy"]).tolist()


def test_labels_outside_27_classes_are_rejected():
    with pytest.raises(ValueError):
        MDL27Classifier().fit(pd.DataFrame([{"text": "x", "label": 27}]))


# --- probabilities and description lengths ---------------------------------------------

def test_bigram_add_one_probabilities_and_description_length_by_hand():
    train = pd.DataFrame([{"text": "a b", "label": 0}, {"text": "a c", "label": 1}])
    model = MDL27Classifier().fit(train)
    v = model._prediction_vocab_size
    assert v == 5  # {a, b, c} + </s> + <unk>
    # class 0: counts (<s>,a)=1, (a,b)=1, (b,</s>)=1; context counts <s>=1, a=1, b=1.
    # "a b": P(a|<s>) = 2/6, P(b|a) = 2/6, P(</s>|b) = 2/6
    expected = -3 * math.log2(2 / 6)
    breakdown = model.score_breakdown(["a b"])[0]
    assert breakdown[0]["conditional_bits"] == pytest.approx(expected)
    # class 1 never saw (a, b): P(b|a) = 1/6 (unseen bigram, seen context); P(</s>|b): unseen context -> 1/V.
    assert breakdown[1]["conditional_bits"] == pytest.approx(-math.log2(2 / 6) - math.log2(1 / 6) - math.log2(1 / v))
    assert model.predict(["a b", "a c"]) == [0, 1]


@pytest.mark.parametrize("class_id", [0, 13, 26])
def test_add_one_distribution_is_normalised_and_positive(class_id):
    model = MDL27Classifier().fit(_all_classes_df())
    per_class = model._per_class_models[class_id]
    vocab = model.vocabulary + ["</s>", "<unk>"]
    for context in (("feel",), ("<s>",), ("never-seen",)):
        probabilities = [_laplace_probability(per_class["ngram_counts"].get(context + (w,), 0),
                                              per_class["context_counts"].get(context, 0), 1.0, len(vocab)) for w in vocab]
        assert min(probabilities) > 0 and sum(probabilities) == pytest.approx(1.0)


def test_scores_are_finite_for_oov_and_empty_texts():
    model = MDL27Classifier().fit(_all_classes_df())
    scores = model.predict_scores(["", "zzz qqq", "joy", "!!!"])
    assert scores.shape == (4, 27) and np.isfinite(scores).all()
    # Distinct OOV words both map to <unk>, hence identical scores.
    assert model.predict_scores(["zzz"]).tolist() == model.predict_scores(["qqq"]).tolist()
    assert all(0 <= p < 27 for p in model.predict(["", "zzz qqq"]))


def test_predictions_are_valid_indices_and_separate_classes():
    model = MDL27Classifier().fit(_all_classes_df())
    texts = [f"i feel {name} today" for name in CLASS_NAMES]
    assert model.predict(texts) == list(range(27))


def test_ties_go_to_lowest_class_index():
    train = pd.DataFrame([{"text": "same words", "label": c} for c in (20, 5, 11)])
    model = MDL27Classifier().fit(train)
    scores = model.predict_scores(["same words"])[0]
    assert scores[5] == scores[11] == scores[20]
    assert model.predict(["same words", "other"]) == [5, 5]


# --- missing classes ---------------------------------------------------------------

def test_missing_classes_get_models_are_reported_and_never_predicted():
    with pytest.warns(UserWarning, match="no training examples"):
        model = MDL27Classifier().fit(_few_classes_df())
    assert sorted(model._per_class_models) == list(range(27))
    assert len(model.missing_classes) == 24
    assert set(model.predictable_classes) == {LABEL_TO_INDEX[n] for n in ("joy", "anger", "grief")}
    breakdown = model.score_breakdown(["completely unseen words"])[0]
    assert all(math.isfinite(parts["conditional_bits"]) for parts in breakdown.values())
    scores = model.predict_scores(["completely unseen words", ""])
    assert np.isinf(scores[:, model.missing_classes]).all() and np.isfinite(scores[:, model.predictable_classes]).all()
    assert set(model.predict(["completely unseen words", "", "so happy"])) <= set(model.predictable_classes)


# --- class prior -------------------------------------------------------------------

def test_class_prior_option():
    train = _few_classes_df()
    with pytest.warns(UserWarning):
        plain = MDL27Classifier().fit(train)
    with pytest.warns(UserWarning):
        prior = MDL27Classifier(use_prior=True).fit(train)
    joy, anger = LABEL_TO_INDEX["joy"], LABEL_TO_INDEX["anger"]
    assert sum(prior.class_priors.values()) == pytest.approx(1.0)
    assert prior.class_priors[joy] == pytest.approx(0.5) and prior.class_priors[anger] == pytest.approx(0.25)
    row = prior.score_breakdown(["so angry"])[0]
    assert row[joy]["prior_bits"] == pytest.approx(1.0) and row[anger]["prior_bits"] == pytest.approx(2.0)
    assert row[joy]["total_bits"] == pytest.approx(row[joy]["conditional_bits"] + 1.0)
    # The no-prior variant is unaffected: prior_bits 0, total = conditional.
    plain_row = plain.score_breakdown(["so angry"])[0]
    assert plain_row[joy]["prior_bits"] == 0.0 and plain_row[joy]["total_bits"] == plain_row[joy]["conditional_bits"]
    assert plain_row[joy]["conditional_bits"] == row[joy]["conditional_bits"]


def test_validation_data_does_not_change_priors_or_counts():
    train = _all_classes_df()
    validation = pd.DataFrame([{"text": "joy joy joy", "label": 17}] * 50)
    a = MDL27Classifier(use_prior=True).fit(train, validation_data=validation)
    b = MDL27Classifier(use_prior=True).fit(train)
    assert a.class_priors == b.class_priors and a._per_class_models == b._per_class_models


# --- relation to the existing MDL ---------------------------------------------------------

def test_no_prior_variant_matches_baseline_mdl_when_all_classes_present():
    train = _all_classes_df()
    texts = ["i feel joy", "zzz", "", "grief grief"]
    baseline = MDLClassifier(n=2, k=1.0).fit(train)
    model = MDL27Classifier().fit(train)
    assert model.predict(texts) == baseline.predict(texts)
    assert [list(d.values()) for d in model.description_lengths(texts)] == [list(d.values()) for d in baseline.description_lengths(texts)]


def test_existing_four_class_mdl_is_unchanged():
    assert (MDLClassifier().n, MDLClassifier().k) == (3, 1.0)
    predictions = REPO_ROOT / "results" / "predictions" / "mdl_ladder_n100_42.csv"
    if not predictions.exists() or not (REPO_ROOT / "data" / "processed" / "test.csv").exists():
        pytest.skip("4-class data or committed predictions not available")
    from emotion_ait.config import load_config
    from emotion_ait.sampling import load_ladder_subset_dataframe

    config = load_config(None)
    processed = pd.concat([pd.read_csv(config.paths.processed_dir / f"{s}.csv") for s in config.dataset.splits], ignore_index=True)
    model = MDLClassifier().fit(load_ladder_subset_dataframe(42, 100, processed, config))
    test = processed[processed["original_split"] == "test"]
    assert model.predict(test["text"].tolist()) == pd.read_csv(predictions)["predicted_label"].tolist()


# --- diagnostics and persistence -------------------------------------------------------------

def test_diagnostics_report():
    train = _all_classes_df(per_class=3)
    model = MDL27Classifier().fit(train)
    report = model.diagnostics(train, {"validation": ["i feel joy", "unknown words here"], "test": ["grief"]})
    assert report["shared_vocabulary_size"] == len(model.vocabulary) and report["missing_classes"] == []
    joy = report["per_class"]["joy"]
    # "i feel joy joy today": <s>-i, i-feel, feel-joy, joy-joy, joy-today, today-</s>
    assert joy["train_examples"] == 3 and joy["unique_ngrams"] == 6
    assert 0 < joy["vocabulary_coverage"] <= 1 and joy["mean_train_bits_per_example"] > 0
    assert report["validation"]["tokens"] == 6 and report["validation"]["oov_tokens"] == 3
    assert sum(report["validation"]["predictions_per_class"].values()) == 2
    assert list(report["per_class"]) == CLASS_NAMES


def test_save_load_round_trip(tmp_path):
    model = MDL27Classifier(use_prior=True).fit(_all_classes_df())
    model.save(tmp_path / "m.pkl")
    restored = MDL27Classifier.load(tmp_path / "m.pkl")
    assert restored.predict_scores(["i feel joy", ""]).tolist() == model.predict_scores(["i feel joy", ""]).tolist()


# --- Kneser-Ney ------------------------------------------------------------------------

KN_TRAIN = pd.DataFrame([
    {"text": "a b", "label": 0}, {"text": "a b", "label": 0}, {"text": "a c", "label": 0},
    {"text": "c b", "label": 1}, {"text": "b", "label": 1},
])


def test_kneser_ney_options_are_validated():
    with pytest.raises(NotImplementedError):
        MDL27Classifier(n=3, smoothing="kneser_ney")
    with pytest.raises(NotImplementedError):
        MDL27Classifier(smoothing="witten_bell")
    with pytest.raises(ValueError):
        MDL27Classifier(smoothing="kneser_ney", kn_backoff="global")
    with pytest.raises(ValueError):
        MDL27Classifier(smoothing="kneser_ney", kn_discount=1.5)


@pytest.mark.parametrize("backoff", ["class", "shared"])
@pytest.mark.parametrize("discount", [None, 0.5, 1.0])
def test_kneser_ney_distributions_are_proper_and_positive(backoff, discount):
    with pytest.warns(UserWarning):  # only 2 of 27 classes present
        model = MDL27Classifier(smoothing="kneser_ney", kn_backoff=backoff, kn_discount=discount).fit(KN_TRAIN)
    vocab = model.vocabulary + ["</s>", "<unk>"]
    for class_id in (0, 1, 5):  # 5 has no training data
        for context in ("<s>", "a", "b", "c", "never-seen"):
            probabilities = [model._kn_probability((context,), w, class_id) for w in vocab]
            assert min(probabilities) > 0
            assert sum(probabilities) == pytest.approx(1.0, abs=1e-12)


def test_kneser_ney_probability_by_hand():
    # Class 0 bigrams: (<s>,a)=3, (a,b)=2, (a,c)=1, (b,</s>)=2, (c,</s>)=1; fixed discount D = 0.5.
    with pytest.warns(UserWarning):
        model = MDL27Classifier(smoothing="kneser_ney", kn_backoff="class", kn_discount=0.5).fit(KN_TRAIN)
    v = model._prediction_vocab_size  # {a, b, c} + </s> + <unk> = 5
    # Continuation counts (class 0): a:1 (<s>), b:1 (a), c:1 (a), </s>:2 (b, c); total 5 types, 4 words.
    floor = 0.5 * 4 / 5 / v
    p_low_b = (1 - 0.5) / 5 + floor
    # P(b | a) = (max(2 - .5, 0) + .5 * N1+(a .) = 2 * p_low(b)) / c(a) = 3
    assert model._kn_probability(("a",), "b", 0) == pytest.approx((1.5 + 0.5 * 2 * p_low_b) / 3)
    # Unseen word after a seen context: only the back-off mass.
    assert model._kn_probability(("a",), "<unk>", 0) == pytest.approx(0.5 * 2 * floor / 3)
    # Unseen context: pure lower order.
    assert model._kn_probability(("zzz",), "b", 0) == pytest.approx(p_low_b)
    expected = -sum(math.log2(model._kn_probability((v_,), w, 0)) for v_, w in (("<s>", "a"), ("a", "b"), ("b", "</s>")))
    assert model.score_breakdown(["a b"])[0][0]["conditional_bits"] == pytest.approx(expected)


def test_kneser_ney_shared_backoff_pools_training_types_only():
    validation = pd.DataFrame([{"text": "zzz yyy", "label": 1}])
    with pytest.warns(UserWarning):
        shared = MDL27Classifier(smoothing="kneser_ney", kn_backoff="shared").fit(KN_TRAIN, validation_data=validation)
    with pytest.warns(UserWarning):
        per_class = MDL27Classifier(smoothing="kneser_ney", kn_backoff="class").fit(KN_TRAIN)
    lower = shared._kn["shared_lower"]
    assert all(shared._kn["per_class"][c]["lower"] is lower for c in (0, 1))
    assert "zzz" not in lower["probabilities"] and "zzz" not in shared.vocabulary
    # "b" follows <s> only in class 1, so the pooled continuation count of b (a, c, <s>) exceeds class 0's (a).
    assert lower["probabilities"]["b"] > 0
    assert per_class._kn["per_class"][0]["lower"] is not per_class._kn["per_class"][1]["lower"]
    assert per_class._kn["shared_lower"] is None


def test_kneser_ney_estimated_discount_formula():
    model = MDL27Classifier(smoothing="kneser_ney")
    assert model._estimate_discount([1, 1, 1, 2, 5]) == pytest.approx(3 / (3 + 2 * 1))
    assert model._estimate_discount([1, 1]) == 1.0  # n2 = 0 -> clipped at 1
    assert model._estimate_discount([3, 4]) == 0.75  # undefined -> default
    assert MDL27Classifier(smoothing="kneser_ney", kn_discount=0.3)._estimate_discount([1, 2]) == 0.3


def test_kneser_ney_scores_finite_and_predictions_valid():
    train = _all_classes_df()
    for backoff in ("class", "shared"):
        model = MDL27Classifier(smoothing="kneser_ney", kn_backoff=backoff).fit(train)
        scores = model.predict_scores(["", "zzz qqq", "i feel joy today"])
        assert np.isfinite(scores).all()
        assert model.predict([f"i feel {name} today" for name in CLASS_NAMES]) == list(range(27))


def test_laplace_default_is_unchanged_by_kneser_ney_support():
    train = _all_classes_df()
    model = MDL27Classifier().fit(train)
    assert model.smoothing == "laplace" and model._kn is None
    baseline = MDLClassifier(n=2, k=1.0).fit(train)
    texts = ["i feel joy", "zzz", ""]
    assert [list(d.values()) for d in model.description_lengths(texts)] == [list(d.values()) for d in baseline.description_lengths(texts)]
