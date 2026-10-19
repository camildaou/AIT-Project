"""Tests for emotion_ait.mdl_classifier."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from emotion_ait.mdl_classifier import MDLClassifier, _laplace_probability, _pad, _tokenize  # noqa: E402


def _toy_train_df() -> pd.DataFrame:
    # Two trivially separable classes with disjoint vocabularies, so a
    # correctly working trigram model should classify them perfectly.
    happy_texts = [
        "i am so happy today",
        "this makes me very happy",
        "happy happy joyful day",
        "what a happy wonderful moment",
    ]
    sad_texts = [
        "i am so sad today",
        "this makes me very sad",
        "sad sad gloomy day",
        "what a sad terrible moment",
    ]
    rows = [{"text": t, "label": 2} for t in happy_texts] + [{"text": t, "label": 3} for t in sad_texts]
    return pd.DataFrame(rows)


def test_fit_rejects_empty_training_set():
    model = MDLClassifier()
    with pytest.raises(ValueError):
        model.fit(pd.DataFrame(columns=["text", "label"]))


def test_fit_rejects_missing_columns():
    model = MDLClassifier()
    with pytest.raises(KeyError):
        model.fit(pd.DataFrame({"text": ["hi"]}))


def test_predict_before_fit_raises():
    model = MDLClassifier()
    with pytest.raises(RuntimeError):
        model.predict(["hello"])


def test_unsupported_smoothing_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        MDLClassifier(smoothing="kneser-ney")


@pytest.mark.parametrize("bad_n", [0, -1, -3])
def test_non_positive_n_raises_value_error(bad_n):
    with pytest.raises(ValueError):
        MDLClassifier(n=bad_n)


@pytest.mark.parametrize("bad_n", [2.5, "3", None, True, False])
def test_non_integer_n_raises_type_error(bad_n):
    with pytest.raises(TypeError):
        MDLClassifier(n=bad_n)


@pytest.mark.parametrize("bad_k", [0, 0.0, -1.0, math.inf, -math.inf, math.nan])
def test_non_positive_or_non_finite_k_raises_value_error(bad_k):
    with pytest.raises(ValueError):
        MDLClassifier(k=bad_k)


@pytest.mark.parametrize("bad_k", ["1.0", None, [1.0]])
def test_non_numeric_k_raises_type_error(bad_k):
    with pytest.raises(TypeError):
        MDLClassifier(k=bad_k)


def test_default_order_is_trigram():
    model = MDLClassifier()
    assert model.n == 3


def test_fit_predict_separates_disjoint_vocabulary_classes():
    model = MDLClassifier(n=3)
    model.fit(_toy_train_df())

    # Held-out sentences that share local trigram context with the
    # training data (as any trigram model requires to generalize from a
    # handful of examples), but are not themselves in the training set.
    preds = model.predict(["what a happy wonderful day", "what a sad terrible day"])
    assert preds == [2, 3]


def test_predict_handles_out_of_vocabulary_text_without_raising():
    model = MDLClassifier(n=3)
    model.fit(_toy_train_df())

    preds = model.predict(["xyzzy qwerty completely unseen gibberish"])
    assert len(preds) == 1
    assert preds[0] in {2, 3}


def test_predict_handles_empty_string_without_raising():
    model = MDLClassifier(n=3)
    model.fit(_toy_train_df())

    preds = model.predict([""])
    assert len(preds) == 1
    assert preds[0] in {2, 3}


def test_save_load_round_trip_preserves_predictions(tmp_path):
    model = MDLClassifier(n=3)
    model.fit(_toy_train_df())

    texts = ["i feel happy and joyful", "i feel sad and gloomy", "a completely unseen sentence"]
    expected = model.predict(texts)

    save_path = tmp_path / "mdl_model.pkl"
    model.save(save_path)
    restored = MDLClassifier.load(save_path)

    assert restored.predict(texts) == expected


def test_tokenize_lowercases_and_extracts_word_tokens():
    assert _tokenize("Hello, World! It's great.") == ["hello", "world", "it's", "great"]


def test_tokenize_handles_empty_and_non_string_input():
    assert _tokenize("") == []
    assert _tokenize(None) == []


def test_pad_adds_n_minus_one_bos_and_one_eos():
    assert _pad(["a", "b"], n=3) == ["<s>", "<s>", "a", "b", "</s>"]
    assert _pad([], n=1) == ["</s>"]


def test_fit_handles_a_class_missing_from_the_training_subset():
    # Simulates a tiny subset where a rare class rounds down to a 0 quota
    # (see README section 4): the classifier must still fit and predict
    # using only the classes actually present.
    df = pd.DataFrame(
        [
            {"text": "i am so happy today", "label": 2},
            {"text": "happy happy joyful day", "label": 2},
        ]
    )
    model = MDLClassifier(n=3)
    model.fit(df)

    preds = model.predict(["completely unrelated text"])
    assert preds == [2]


def test_prediction_vocabulary_excludes_bos_marker():
    model = MDLClassifier(n=3)
    model.fit(_toy_train_df())

    # <s> must never be counted among predictable tokens: it is only
    # ever used as left-padding context.
    assert "<s>" not in model._word_vocab
    assert model._prediction_vocab_size == len(model._word_vocab) + 2  # + </s> + <unk>


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_laplace_probabilities_sum_to_one_over_the_prediction_vocabulary(n):
    model = MDLClassifier(n=n)
    model.fit(_toy_train_df())

    prediction_vocab = sorted(model._word_vocab) + ["</s>", "<unk>"]
    assert len(prediction_vocab) == model._prediction_vocab_size

    for class_id in (2, 3):
        per_class = model._per_class_models[class_id]
        # Use a context actually observed during training for this class
        # (the start-of-sequence context always exists for n > 1).
        context = tuple(["<s>"] * (n - 1)) if n > 1 else ()
        total_probability = sum(
            _laplace_probability(
                ngram_count=per_class["ngram_counts"].get(context + (word,), 0),
                context_count=per_class["context_counts"].get(context, 0),
                k=model.k,
                prediction_vocab_size=model._prediction_vocab_size,
            )
            for word in prediction_vocab
        )
        assert total_probability == pytest.approx(1.0, abs=1e-9)


def test_laplace_probabilities_are_strictly_positive_even_for_unseen_ngrams():
    model = MDLClassifier(n=3)
    model.fit(_toy_train_df())

    per_class = model._per_class_models[2]
    probability = _laplace_probability(
        ngram_count=per_class["ngram_counts"].get(("totally", "unseen", "context"), 0),
        context_count=per_class["context_counts"].get(("totally", "unseen"), 0),
        k=model.k,
        prediction_vocab_size=model._prediction_vocab_size,
    )
    assert probability > 0.0


@pytest.mark.parametrize("n", [1, 2, 3])
def test_unigram_bigram_and_trigram_orders_all_fit_and_predict(n):
    model = MDLClassifier(n=n)
    model.fit(_toy_train_df())
    preds = model.predict(["what a happy wonderful day", "what a sad terrible day"])
    assert len(preds) == 2
    assert all(p in {2, 3} for p in preds)


def test_description_lengths_matches_predict_argmin():
    model = MDLClassifier(n=3)
    model.fit(_toy_train_df())

    texts = ["what a happy wonderful day", "what a sad terrible day"]
    scores = model.description_lengths(texts)
    preds = model.predict(texts)

    assert [set(s.keys()) for s in scores] == [{2, 3}, {2, 3}]
    assert [min(s, key=s.get) for s in scores] == preds


def test_description_lengths_before_fit_raises():
    model = MDLClassifier()
    with pytest.raises(RuntimeError):
        model.description_lengths(["hello"])


def test_predict_breaks_ties_in_favor_of_the_smallest_class_id():
    # Identical training data for both classes guarantees a tied
    # description length for any input text.
    df = pd.DataFrame(
        [
            {"text": "i am so happy today", "label": 3},
            {"text": "i am so happy today", "label": 2},
        ]
    )
    model = MDLClassifier(n=3)
    model.fit(df)

    scores = model.description_lengths(["completely unrelated text"])[0]
    assert scores[2] == pytest.approx(scores[3])
    assert model.predict(["completely unrelated text"]) == [2]
