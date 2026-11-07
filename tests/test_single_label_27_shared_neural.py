"""Tests for the shared class-conditioned neural bigram MDL (emotion_ait.single_label_27.shared_neural)."""

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
from emotion_ait.single_label_27.shared_neural import SharedNeuralBigramMDL27

REPO_ROOT = Path(__file__).resolve().parents[1]
HAVE_REAL_DATA = (REPO_ROOT / "data" / "subsets" / "single_label_27_ladder" / "manifest.json").exists()
FAST = dict(embedding_dim=16, class_dim=8, hidden_dim=16, learning_rate=0.02, batch_size=32, epochs=200,
            patience=1000, device="cpu", seed=0)


def _all_classes_df(per_class: int = 4) -> pd.DataFrame:
    return pd.DataFrame([{"text": f"i feel {name} {name} today", "label": i}
                         for i, name in enumerate(CLASS_NAMES) for _ in range(per_class)])


@pytest.fixture(scope="module")
def fitted():
    train = _all_classes_df()
    return SharedNeuralBigramMDL27(**FAST).fit(train, validation_data=train), train


def test_one_shared_model_with_small_class_specific_part(fitted):
    model, _ = fitted
    meta = model.training_metadata
    lm = model._model
    assert model._models is None  # no per-class networks
    assert meta["parameters_total"] == sum(p.numel() for p in lm.parameters())
    assert meta["parameters_class_specific"] == 27 * FAST["class_dim"] + 27 * model.prediction_vocab_size
    assert lm.class_embedding.num_embeddings == 27


def test_softmax_is_proper_for_every_class(fitted):
    model, _ = fitted
    contexts = torch.arange(len(model._token_to_id))
    for c in (0, 13, 26):
        p = torch.softmax(model._model(contexts, torch.full_like(contexts, c)), dim=-1)
        assert p.shape == (len(contexts), model.prediction_vocab_size)
        assert torch.all(p > 0) and torch.allclose(p.sum(-1), torch.ones(len(contexts)), atol=1e-5)


def test_class_conditioning_changes_the_distribution(fitted):
    model, _ = fitted
    ids = model._token_to_id
    feel = torch.tensor([ids["feel"]] * 2)
    p = torch.softmax(model._model(feel, torch.tensor([LABEL_TO_INDEX["joy"], LABEL_TO_INDEX["anger"]])), -1)
    assert p[0, ids["joy"]] > 0.5 > p[1, ids["joy"]]


def test_training_reduces_loss_and_classifies_separable_data(fitted):
    model, _ = fitted
    history = model.training_metadata["train_loss_bits_per_token"]
    assert history[-1] < history[0] / 2
    assert model.predict([f"i feel {name} today" for name in CLASS_NAMES]) == list(range(27))


def test_description_length_matches_softmax_bits(fitted):
    model, _ = fitted
    text = "i feel joy today"
    previous, following = model._transitions(text)
    expected = []
    for c in range(27):
        log_p = torch.log_softmax(model._model(torch.tensor(previous), torch.full((len(previous),), c)), -1)
        expected.append(float(-log_p[torch.arange(len(following)), torch.tensor(following)].sum()) / math.log(2))
    np.testing.assert_allclose(model.class_description_lengths([text])[0], expected, rtol=1e-5)


def test_scores_finite_for_empty_and_oov_text(fitted):
    model, _ = fitted
    scores = model.class_description_lengths(["", "zzz qqq", "!!!"])
    assert scores.shape == (3, 27) and np.isfinite(scores).all()
    assert all(0 <= p < 27 for p in model.predict(["", "zzz qqq"]))


def test_training_data_only_and_cpu_determinism():
    train = _all_classes_df()
    held_out = pd.DataFrame([{"text": "brand new words joy", "label": LABEL_TO_INDEX["joy"]}])
    a = SharedNeuralBigramMDL27(**{**FAST, "epochs": 8}).fit(train, validation_data=held_out)
    assert "brand" not in a.vocabulary
    assert a.training_metadata["train_tokens"] == sum(len(a._transitions(t)[0]) for t in train["text"])
    # Validation only selects the stopping epoch: retraining without validation for exactly
    # that many epochs reproduces the same weights bit for bit (CPU determinism).
    k = a.training_metadata["best_epoch"]
    b = SharedNeuralBigramMDL27(**{**FAST, "epochs": k}).fit(train, validation_data=None)
    assert a.vocabulary == b.vocabulary and b.training_metadata["epochs_trained"] == k
    np.testing.assert_array_equal(a.class_description_lengths(["i feel joy", "zzz"]), b.class_description_lengths(["i feel joy", "zzz"]))


def test_missing_classes_excluded_and_prior_variant():
    train = pd.DataFrame([{"text": "so happy", "label": LABEL_TO_INDEX["joy"]}] * 3
                         + [{"text": "so angry", "label": LABEL_TO_INDEX["anger"]}])
    with pytest.warns(UserWarning, match="no training examples"):
        model = SharedNeuralBigramMDL27(**{**FAST, "epochs": 5}).fit(train)
    assert set(model.predict(["so sad", "", "zzz"])) <= {LABEL_TO_INDEX["joy"], LABEL_TO_INDEX["anger"]}
    with pytest.warns(UserWarning):
        prior = SharedNeuralBigramMDL27(**{**FAST, "epochs": 5}, use_prior=True).fit(train)
    diff = prior.predict_scores(["so sad"])[0] - prior.class_description_lengths(["so sad"])[0]
    assert diff[LABEL_TO_INDEX["joy"]] == pytest.approx(-math.log2(0.75))


def test_invalid_arguments_and_save_load(fitted, tmp_path):
    for kwargs in ({"class_dim": 0}, {"hidden_dim": 0}):
        with pytest.raises(ValueError):
            SharedNeuralBigramMDL27(**kwargs)
    model, _ = fitted
    model.save(tmp_path / "m.pkl")
    restored = SharedNeuralBigramMDL27.load(tmp_path / "m.pkl")
    texts = ["i feel joy today", ""]
    np.testing.assert_allclose(restored.class_description_lengths(texts), model.class_description_lengths(texts), rtol=1e-6)


@pytest.mark.skipif(not HAVE_REAL_DATA, reason="27-class data/manifests not generated locally")
def test_runner_gives_shared_model_the_same_training_ids_as_count_mdl():
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
                seen[tag + "_val"] = set(validation_data["original_split"])
                return original(train_data, validation_data)

            model.fit = fit
            return model, {}
        return build

    config = load_single_label_27_config()
    with tempfile.TemporaryDirectory() as tmp:
        base = dataclasses.replace(config.base, paths=dataclasses.replace(config.base.paths, results_dir=Path(tmp)))
        cfg = dataclasses.replace(config, base=base)
        run_single_label_27_ladder(spy("shared", lambda s: SharedNeuralBigramMDL27(embedding_dim=8, class_dim=4, hidden_dim=8,
                                                                                    epochs=1, device="cpu", seed=s)),
                                   "shared_test", cfg, sizes=[500], seeds=[44])
        run_single_label_27_ladder(spy("mdl", lambda s: MDL27Classifier()), "mdl_test", cfg, sizes=[500], seeds=[44])
    assert seen["shared"] == seen["mdl"] and len(seen["shared"]) == 500
    assert seen["shared_val"] == {"validation"}
