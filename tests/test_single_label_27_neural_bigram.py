"""Tests for the 27-class neural bigram MDL classifier (emotion_ait.single_label_27.neural_bigram)."""

from __future__ import annotations

import dataclasses
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from emotion_ait.single_label_27.data import CLASS_NAMES, LABEL_TO_INDEX
from emotion_ait.single_label_27.neural_bigram import NeuralBigramMDL27

REPO_ROOT = Path(__file__).resolve().parents[1]
HAVE_REAL_DATA = (REPO_ROOT / "data" / "subsets" / "single_label_27_ladder" / "manifest.json").exists()
FAST = dict(embedding_dim=8, learning_rate=0.05, batch_size=16, epochs=15, patience=2, device="cpu", seed=0)


def _all_classes_df(per_class: int = 3) -> pd.DataFrame:
    return pd.DataFrame([{"text": f"i feel {name} {name} today", "label": i}
                         for i, name in enumerate(CLASS_NAMES) for _ in range(per_class)])


@pytest.fixture(scope="module")
def fitted():
    train = _all_classes_df()
    return NeuralBigramMDL27(**FAST).fit(train, validation_data=train), train


# --- probability model ----------------------------------------------------------------

def test_softmax_is_a_proper_distribution_over_the_prediction_vocabulary(fitted):
    model, _ = fitted
    v = model.prediction_vocab_size
    assert v == len(model.vocabulary) + 2  # words + </s> + <unk>; <s> is not an outcome
    contexts = torch.arange(len(model._token_to_id))  # every input id, including <s>
    for lm in (model._models[0], model._models[26]):
        probabilities = torch.softmax(lm(contexts), dim=-1)
        assert probabilities.shape == (len(contexts), v)
        assert torch.all(probabilities > 0)
        assert torch.allclose(probabilities.sum(-1), torch.ones(len(contexts)), atol=1e-5)


def test_training_reduces_cross_entropy_and_all_27_models_are_trained(fitted):
    model, _ = fitted
    per_class = model.training_metadata["per_class"]
    assert list(per_class) == CLASS_NAMES and model.missing_classes == []
    for info in per_class.values():
        assert info["train_examples"] == 3 and info["train_tokens"] == 3 * 6  # "i feel x x today" -> 6 transitions
        assert info["epochs_trained"] >= 1
        assert info["train_loss_bits_per_token"][-1] < info["train_loss_bits_per_token"][0]
        assert info["best_validation_bits_per_token"] < math.log2(model.prediction_vocab_size)  # better than uniform
    assert model.training_metadata["parameters_total"] == 27 * model.training_metadata["parameters_per_model"]


def test_bos_eos_and_unk_handling(fitted):
    model, _ = fitted
    ids = model._token_to_id
    previous, following = model._transitions("i feel joy")
    assert previous == [ids["<s>"], ids["i"], ids["feel"], ids["joy"]]
    assert following == [ids["i"], ids["feel"], ids["joy"], ids["</s>"]]
    assert model._transitions("") == ([ids["<s>"]], [ids["</s>"]])
    assert model._transitions("zzz") == ([ids["<s>"], ids["<unk>"]], [ids["<unk>"], ids["</s>"]])
    assert ids["<s>"] == model.prediction_vocab_size  # input-only id, outside the softmax outcomes


def test_description_length_is_summed_bits_of_the_softmax(fitted):
    model, _ = fitted
    text = "i feel joy today"
    previous, following = model._transitions(text)
    expected = []
    for lm in model._models:
        log_p = torch.log_softmax(lm(torch.tensor(previous)), dim=-1)[torch.arange(len(following)), torch.tensor(following)]
        expected.append(float(-log_p.sum()) / math.log(2))
    np.testing.assert_allclose(model.class_description_lengths([text])[0], expected, rtol=1e-5)


def test_scores_finite_and_predictions_valid(fitted):
    model, _ = fitted
    texts = ["", "zzz qqq", "i feel joy today", "!!!"]
    scores = model.class_description_lengths(texts)
    assert scores.shape == (4, 27) and np.isfinite(scores).all()
    predictions = model.predict(texts)
    assert len(predictions) == 4 and all(isinstance(p, int) and 0 <= p < 27 for p in predictions)
    assert model.predict([f"i feel {name} today" for name in CLASS_NAMES]) == list(range(27))


def test_prior_variant_is_separate_and_adds_class_code():
    train = pd.concat([_all_classes_df(), pd.DataFrame([{"text": "i feel great", "label": 0}] * 6)], ignore_index=True)
    plain = NeuralBigramMDL27(**FAST).fit(train)
    prior = NeuralBigramMDL27(**FAST, use_prior=True).fit(train)
    texts = ["i feel joy today", "hello"]
    np.testing.assert_allclose(prior.predict_scores(texts) - plain.predict_scores(texts),
                               np.tile([-math.log2(p) for p in prior.class_priors.values()], (2, 1)), rtol=1e-6)


# --- data use, determinism, edge cases ----------------------------------------------------

def test_vocabulary_and_parameters_come_from_training_data_only():
    train = _all_classes_df()
    held_out = pd.DataFrame([{"text": "brand new validation words", "label": LABEL_TO_INDEX["joy"]}])
    with_val = NeuralBigramMDL27(**{**FAST, "patience": 1000}).fit(train, validation_data=held_out)
    without = NeuralBigramMDL27(**FAST).fit(train)
    assert "brand" not in with_val.vocabulary and with_val.vocabulary == without.vocabulary
    # Without validation data the full epoch budget is used, and training pairs come from train_data only.
    a = NeuralBigramMDL27(**{**FAST, "patience": 1000}).fit(train, validation_data=None)
    b = NeuralBigramMDL27(**{**FAST, "patience": 1000}).fit(train, validation_data=None)
    texts = ["i feel joy", "zzz"]
    np.testing.assert_array_equal(a.class_description_lengths(texts), b.class_description_lengths(texts))
    assert all(i["epochs_trained"] == FAST["epochs"] and not i["stopped_early"] for i in a.training_metadata["per_class"].values())
    assert sum(i["train_tokens"] for i in a.training_metadata["per_class"].values()) == sum(
        len(a._transitions(t)[0]) for t in train["text"])


def test_fresh_models_for_every_fit_and_seed_dependence():
    train = _all_classes_df()
    first = NeuralBigramMDL27(**FAST).fit(train)
    second = NeuralBigramMDL27(**FAST).fit(train)
    other_seed = NeuralBigramMDL27(**{**FAST, "seed": 1}).fit(train)
    assert first._models[0] is not second._models[0]
    texts = ["i feel joy today"]
    np.testing.assert_array_equal(first.class_description_lengths(texts), second.class_description_lengths(texts))
    assert not np.array_equal(first.class_description_lengths(texts), other_seed.class_description_lengths(texts))


def test_missing_classes_are_reported_and_never_predicted():
    train = pd.DataFrame([{"text": "so happy", "label": LABEL_TO_INDEX["joy"]}, {"text": "so angry", "label": LABEL_TO_INDEX["anger"]}])
    with pytest.warns(UserWarning, match="no training examples"):
        model = NeuralBigramMDL27(**FAST).fit(train)
    assert len(model.missing_classes) == 25 and len(model._models) == 27
    assert model.training_metadata["per_class"]["grief"]["epochs_trained"] == 0
    assert np.isfinite(model.class_description_lengths(["so sad"])).all()
    assert set(model.predict(["so sad", "", "zzz"])) <= {LABEL_TO_INDEX["joy"], LABEL_TO_INDEX["anger"]}


def test_invalid_arguments():
    for kwargs in ({"embedding_dim": 0}, {"epochs": 0}, {"learning_rate": 0}, {"device": "tpu"}):
        with pytest.raises(ValueError):
            NeuralBigramMDL27(**kwargs)
    with pytest.raises(ValueError):
        NeuralBigramMDL27(**FAST).fit(pd.DataFrame([{"text": "x", "label": 27}]))


def test_save_load_round_trip(fitted, tmp_path):
    model, _ = fitted
    model.save(tmp_path / "nb.pkl")
    restored = NeuralBigramMDL27.load(tmp_path / "nb.pkl")
    texts = ["i feel joy today", ""]
    np.testing.assert_allclose(restored.class_description_lengths(texts), model.class_description_lengths(texts), rtol=1e-6)


@pytest.mark.skipif(not HAVE_REAL_DATA, reason="27-class data/manifests not generated locally")
def test_runner_gives_neural_model_the_same_training_ids_as_count_mdl():
    from emotion_ait.single_label_27.config import load_single_label_27_config
    from emotion_ait.single_label_27.experiments import run_single_label_27_ladder
    from emotion_ait.single_label_27.mdl import MDL27Classifier

    seen = {}

    def spy(tag, factory):
        def build(seed):
            model = factory(seed)
            original = model.fit

            def fit(train_data, validation_data=None):
                seen[tag] = train_data["example_id"].tolist()
                seen[tag + "_val_ids"] = None if validation_data is None else set(validation_data["original_split"])
                return original(train_data, validation_data)

            model.fit = fit
            return model, {}
        return build

    config = load_single_label_27_config()
    with tempfile.TemporaryDirectory() as tmp:
        base = dataclasses.replace(config.base, paths=dataclasses.replace(config.base.paths, results_dir=Path(tmp)))
        cfg = dataclasses.replace(config, base=base)
        run_single_label_27_ladder(spy("neural", lambda s: NeuralBigramMDL27(embedding_dim=8, epochs=1, device="cpu", seed=s)),
                                   "neural_test", cfg, sizes=[500], seeds=[43])
        run_single_label_27_ladder(spy("mdl", lambda s: MDL27Classifier()), "mdl_test", cfg, sizes=[500], seeds=[43])
    assert seen["neural"] == seen["mdl"] and len(seen["neural"]) == 500
    assert seen["neural_val_ids"] == {"validation"}  # only validation rows reach fit(), never test
