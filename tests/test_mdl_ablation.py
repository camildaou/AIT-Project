"""Tests for the MDL n-gram-order / smoothing ablation (emotion_ait.ablation)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from emotion_ait.ablation import (
    REFERENCE_CONFIG_SUMMARY,
    STUDIES,
    ablation_model_name,
    config_summary,
    study_configs,
)
from emotion_ait.mdl_classifier import MDLClassifier, _laplace_probability, _ngrams, _pad

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDERS = [2, 3, 4]
KS = [0.1, 0.5, 1.0, 2.0]


def _train_df() -> pd.DataFrame:
    rows = [("i am so happy today", 2), ("happy happy joyful day", 2), ("what a happy moment", 2),
            ("i am so sad today", 3), ("sad sad gloomy day", 3)]
    return pd.DataFrame([{"text": t, "label": y} for t, y in rows])


def _local_ladder():
    """(config, processed_df) when the real processed data is available locally."""
    if not (REPO_ROOT / "data" / "processed" / "test.csv").exists():
        pytest.skip("processed data not available")
    from emotion_ait.config import load_config

    config = load_config(None)
    processed = pd.concat(
        [pd.read_csv(config.paths.processed_dir / f"{s}.csv") for s in config.dataset.splits], ignore_index=True
    )
    return config, processed


# --- configuration naming ------------------------------------------------------

def test_study_grids_and_names():
    assert [n for n, _ in STUDIES["ngram"]] == ORDERS and {k for _, k in STUDIES["ngram"]} == {1.0}
    assert [k for _, k in STUDIES["smoothing"]] == KS and {n for n, _ in STUDIES["smoothing"]} == {3}
    names = [name for study in STUDIES for name, _, _ in study_configs(study)]
    assert set(names) == {"mdl_n2_k1", "mdl", "mdl_n4_k1", "mdl_n3_k0p1", "mdl_n3_k0p5", "mdl_n3_k2"}
    assert ablation_model_name(3, 1) == "mdl"
    assert config_summary(3, 1) == REFERENCE_CONFIG_SUMMARY


def test_classifier_defaults_are_the_baseline():
    model = MDLClassifier()
    assert (model.n, model.k, model.smoothing) == (3, 1.0, "laplace")


# --- context construction, BOS / EOS ----------------------------------------------

@pytest.mark.parametrize("n", ORDERS)
def test_context_construction_bos_padding_and_eos(n):
    tokens = ["a", "b", "c"]
    padded = _pad(tokens, n)
    assert padded == ["<s>"] * (n - 1) + tokens + ["</s>"]
    grams = _ngrams(padded, n)
    # One n-gram per predicted token: every word plus </s>, never <s>.
    assert len(grams) == len(tokens) + 1
    assert [g[-1] for g in grams] == tokens + ["</s>"]
    assert grams[0] == tuple(["<s>"] * (n - 1) + ["a"])
    assert all(len(g) == n for g in grams)
    assert all("</s>" not in g[:-1] for g in grams)


@pytest.mark.parametrize("n", ORDERS)
def test_fitted_contexts_have_length_n_minus_one(n):
    model = MDLClassifier(n=n).fit(_train_df())
    for per_class in model._per_class_models.values():
        assert all(len(c) == n - 1 for c in per_class["context_counts"])
        assert all(len(g) == n for g in per_class["ngram_counts"])
        assert sum(per_class["context_counts"].values()) == sum(per_class["ngram_counts"].values())


def test_bigram_description_length_by_hand():
    model = MDLClassifier(n=2, k=0.5).fit(pd.DataFrame([{"text": "a b", "label": 0}]))
    v = model._prediction_vocab_size  # {a, b} + </s> + <unk> = 4
    assert v == 4
    # "a": P(a|<s>) = (1+.5)/(1+2); "</s>": P(</s>|a) = (0+.5)/(1+2)
    expected = -math.log2(1.5 / 3) - math.log2(0.5 / 3)
    assert model.description_lengths(["a"])[0][0] == pytest.approx(expected)


# --- smoothing normalisation & unseen n-grams ---------------------------------------

@pytest.mark.parametrize("n", ORDERS)
@pytest.mark.parametrize("k", KS)
def test_probabilities_normalise_over_prediction_vocabulary(n, k):
    model = MDLClassifier(n=n, k=k).fit(_train_df())
    vocab = sorted(model._word_vocab) + ["</s>", "<unk>"]
    assert "<s>" not in vocab and len(vocab) == model._prediction_vocab_size
    seen_context = tuple(["<s>"] * (n - 1))
    unseen_context = tuple(["zzz"] * (n - 1))
    for per_class in model._per_class_models.values():
        for context in (seen_context, unseen_context):
            total = sum(
                _laplace_probability(per_class["ngram_counts"].get(context + (w,), 0),
                                     per_class["context_counts"].get(context, 0), k, len(vocab))
                for w in vocab
            )
            assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("k", KS)
def test_unseen_ngrams_get_k_over_denominator(k):
    model = MDLClassifier(n=3, k=k).fit(_train_df())
    v = model._prediction_vocab_size
    per_class = model._per_class_models[2]
    context = ("<s>", "<s>")
    c = per_class["context_counts"][context]
    assert _laplace_probability(per_class["ngram_counts"].get(context + ("gloomy",), 0), c, k, v) == pytest.approx(k / (c + k * v))
    # Unseen context: uniform over the prediction vocabulary.
    assert _laplace_probability(0, 0, k, v) == pytest.approx(1 / v)


# --- vocabulary & OOV ------------------------------------------------------------

def test_vocabulary_is_identical_across_orders_and_k():
    vocabularies = {(n, k): MDLClassifier(n=n, k=k).fit(_train_df())._word_vocab for n in ORDERS for k in KS}
    assert len({frozenset(v) for v in vocabularies.values()}) == 1


@pytest.mark.parametrize("n", ORDERS)
def test_oov_words_map_to_unk_and_validation_never_enters_vocabulary(n):
    df = _train_df()
    validation = pd.DataFrame([{"text": "brand new words", "label": 3}])
    model = MDLClassifier(n=n).fit(df, validation_data=validation)
    assert model._word_vocab == MDLClassifier(n=n).fit(df)._word_vocab
    assert "brand" not in model._word_vocab
    # Two different OOV words are both <unk>, hence scored identically.
    assert model.description_lengths(["happy xyzzy"]) == model.description_lengths(["happy qwerty"])


@pytest.mark.parametrize("n", ORDERS)
@pytest.mark.parametrize("k", KS)
def test_results_are_deterministic(n, k):
    texts = ["happy sad day", "unseen words here", ""]
    first = MDLClassifier(n=n, k=k).fit(_train_df()).description_lengths(texts)
    second = MDLClassifier(n=n, k=k).fit(_train_df().sample(frac=1.0, random_state=0)).description_lengths(texts)
    assert first == pytest.approx(second)


# --- real ladder (skipped without local data) ------------------------------------------

def test_training_ids_are_identical_across_configurations():
    config, processed = _local_ladder()
    from emotion_ait.sampling import load_ladder_subset_dataframe, load_ladder_subset_ids

    for seed in config.sampling.ladder_seeds:
        ids = set(load_ladder_subset_dataframe(seed, 100, processed, config)["example_id"])
        assert ids == set(load_ladder_subset_ids(seed, 100, config))
    # Every finished ablation run trained on the same realised subset size as the baseline.
    metrics = config.paths.results_dir / "metrics"
    reference = pd.read_csv(metrics / "mdl_ladder_results.csv").set_index(["train_subset_name", "seed"])["train_subset_size"]
    for name in ("mdl_n2_k1", "mdl_n4_k1", "mdl_n3_k0p1", "mdl_n3_k0p5", "mdl_n3_k2"):
        path = metrics / f"{name}_ladder_results.csv"
        if path.exists():
            sizes = pd.read_csv(path).set_index(["train_subset_name", "seed"])["train_subset_size"]
            assert sizes.equals(reference.loc[sizes.index])


def test_reference_results_verify_and_baseline_predictions_unchanged():
    config, processed = _local_ladder()
    from emotion_ait.ablation import verify_reference_results
    from emotion_ait.sampling import load_ladder_subset_dataframe

    reference = verify_reference_results(config)
    assert {json.dumps(json.loads(s), sort_keys=True) for s in reference["config_summary"]} == {
        json.dumps(REFERENCE_CONFIG_SUMMARY, sort_keys=True)
    }
    test_df = processed[processed["original_split"] == "test"]
    for seed, size in ((43, 500), (46, 3125)):
        model = MDLClassifier().fit(load_ladder_subset_dataframe(seed, size, processed, config))
        expected = pd.read_csv(config.paths.results_dir / "predictions" / f"mdl_ladder_n{size}_{seed}.csv")
        assert model.predict(test_df["text"].tolist()) == expected["predicted_label"].tolist()
