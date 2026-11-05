"""Tests for emotion_ait.mdl_variants (mdl_prior, mdl_two_part, mdl_prequential)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from emotion_ait.mdl_classifier import MDLClassifier, _pad, _tokenize
from emotion_ait.mdl_variants import (
    MDLPrequentialClassifier,
    MDLPriorClassifier,
    MDLTwoPartClassifier,
    _elias_gamma_bits,
    _log2_binomial,
    _model_description_length,
)
from emotion_ait.sampling import nested_prefix_order

REPO_ROOT = Path(__file__).resolve().parents[1]

HELD_OUT = ["what a happy wonderful day", "what a sad terrible day", "an unseen sentence", ""]


def _train_df() -> pd.DataFrame:
    # Imbalanced on purpose (5 joy vs 3 sadness) so priors are not uniform.
    joy = ["i am so happy today", "this makes me very happy", "happy happy joyful day",
           "what a happy wonderful moment", "so very happy now"]
    sadness = ["i am so sad today", "this makes me very sad", "sad sad gloomy day"]
    rows = [{"text": t, "label": 2} for t in joy] + [{"text": t, "label": 3} for t in sadness]
    df = pd.DataFrame(rows)
    df.insert(0, "example_id", [f"ex{i:02d}" for i in range(len(df))])
    return df


def _held_out_df(label: int) -> pd.DataFrame:
    return pd.DataFrame({"example_id": [f"val{i}" for i in range(len(HELD_OUT))], "text": HELD_OUT,
                         "label": [label] * len(HELD_OUT)})


def _counts(model):
    return {c: (dict(m["ngram_counts"]), dict(m["context_counts"])) for c, m in model._per_class_models.items()}


# --- mdl_prior ---------------------------------------------------------------

def test_priors_sum_to_one_and_match_training_frequencies():
    model = MDLPriorClassifier().fit(_train_df())
    assert sum(model.class_priors.values()) == pytest.approx(1.0)
    assert model.class_priors == pytest.approx({2: 5 / 8, 3: 3 / 8})


def test_prior_costs_are_in_bits():
    model = MDLPriorClassifier().fit(_train_df())
    assert model._class_cost_bits == pytest.approx({2: -math.log2(5 / 8), 3: -math.log2(3 / 8)})


def test_prior_scores_equal_conditional_plus_prior_cost():
    df = _train_df()
    baseline = MDLClassifier().fit(df).description_lengths(HELD_OUT)
    model = MDLPriorClassifier().fit(df)
    for base, breakdown, total in zip(baseline, model.score_breakdown(HELD_OUT), model.description_lengths(HELD_OUT)):
        for class_id in (2, 3):
            parts = breakdown[class_id]
            assert parts["conditional_bits"] == pytest.approx(base[class_id])
            assert parts["prior_bits"] == pytest.approx(-math.log2(model.class_priors[class_id]))
            assert parts["total_bits"] == pytest.approx(parts["conditional_bits"] + parts["prior_bits"])
            assert total[class_id] == pytest.approx(parts["total_bits"])
    assert model.predict(HELD_OUT) == [min(s, key=s.get) for s in model.description_lengths(HELD_OUT)]


def test_validation_labels_do_not_influence_priors_or_scores():
    df = _train_df()
    plain = MDLPriorClassifier().fit(df)
    with_val = MDLPriorClassifier().fit(df, validation_data=_held_out_df(label=3))
    assert with_val.class_priors == plain.class_priors
    assert with_val.description_lengths(HELD_OUT) == plain.description_lengths(HELD_OUT)


def test_prior_breaks_ties_in_favour_of_smallest_class_id():
    df = pd.DataFrame([{"text": "same text", "label": 3}, {"text": "same text", "label": 2}])
    assert MDLPriorClassifier().fit(df).predict(["anything"]) == [2]


def test_variants_save_load_round_trip(tmp_path):
    for model in (MDLPriorClassifier(), MDLTwoPartClassifier(), MDLPrequentialClassifier(use_prior=True)):
        model.fit(_train_df())
        path = tmp_path / f"{type(model).__name__}.pkl"
        model.save(path)
        assert type(model).load(path).description_lengths(HELD_OUT) == model.description_lengths(HELD_OUT)


# --- baseline untouched --------------------------------------------------------

def test_baseline_mdl_reproduces_committed_ladder_predictions():
    # Re-fits the unchanged baseline on ladder seed 42 / n100 and compares
    # with the committed predictions file; skipped without local data.
    predictions_path = REPO_ROOT / "results" / "predictions" / "mdl_ladder_n100_42.csv"
    processed_dir = REPO_ROOT / "data" / "processed"
    if not predictions_path.exists() or not (processed_dir / "test.csv").exists():
        pytest.skip("processed data or committed baseline predictions not available")
    from emotion_ait.config import load_config
    from emotion_ait.sampling import load_ladder_subset_dataframe

    config = load_config(None)
    processed = pd.concat([pd.read_csv(processed_dir / f"{s}.csv") for s in config.dataset.splits], ignore_index=True)
    test_df = processed[processed["original_split"] == "test"]
    model = MDLClassifier(n=3, smoothing="laplace", k=1.0).fit(load_ladder_subset_dataframe(42, 100, processed, config))
    expected = pd.read_csv(predictions_path)
    assert model.predict(test_df["text"].tolist()) == expected["predicted_label"].tolist()


# --- mdl_two_part ------------------------------------------------------------

@pytest.mark.parametrize("x, bits", [(1, 1), (2, 3), (3, 3), (4, 5), (7, 5), (8, 7)])
def test_elias_gamma_bits(x, bits):
    assert _elias_gamma_bits(x) == bits


def test_log2_binomial_matches_exact_value():
    assert _log2_binomial(10, 3) == pytest.approx(math.log2(math.comb(10, 3)))


def test_model_description_length_of_hand_computed_table():
    # 2 distinct bigrams over |W| = 3 words -> U = 4 ** 2 = 16 possible bigrams.
    counts = {("<s>", "a"): 1, ("a", "</s>"): 3}
    expected = _elias_gamma_bits(3) + math.log2(math.comb(16, 2)) + _elias_gamma_bits(1) + _elias_gamma_bits(3)
    assert _model_description_length(counts, word_vocab_size=3, n=2) == pytest.approx(expected)


def test_two_part_model_cost_grows_with_class_training_data():
    model = MDLTwoPartClassifier().fit(_train_df())
    assert model._class_cost_bits[2] > model._class_cost_bits[3] > 0


# --- mdl_prequential -----------------------------------------------------------

def test_first_prediction_is_made_with_empty_models():
    df = _train_df()
    model = MDLPrequentialClassifier(seed=0).fit(df)
    first = model.prequential_trace[0]
    text = df.loc[df["example_id"] == first["example_id"], "text"].iloc[0]
    # Empty add-k model: every padded token costs exactly log2 V bits.
    uniform = (len(_tokenize(text)) + 1) * math.log2(model._prediction_vocab_size)
    assert first["bits_by_class"] == pytest.approx({2: uniform, 3: uniform})
    assert first["label_bits"] == pytest.approx(math.log2(2))


def test_each_observation_updates_only_its_true_class():
    df = _train_df()
    order = ["ex00", "ex05", "ex01"] + [i for i in df["example_id"] if i not in ("ex00", "ex05", "ex01")]
    model = MDLPrequentialClassifier(order=order).fit(df)
    trace = model.prequential_trace
    vocab_bits = math.log2(model._prediction_vocab_size)
    # ex05 (sadness) comes after only ex00 (joy): the sadness model must still be empty.
    tokens = _tokenize(df.loc[df["example_id"] == "ex05", "text"].iloc[0])
    assert trace[1]["bits_by_class"][3] == pytest.approx((len(tokens) + 1) * vocab_bits)
    assert trace[1]["bits_by_class"][2] != pytest.approx(trace[1]["bits_by_class"][3])
    assert model.prequential_summary["update_counts"] == {2: 5, 3: 3}


def test_cumulative_code_length_equals_sum_of_individual_lengths():
    model = MDLPrequentialClassifier(seed=1).fit(_train_df())
    trace, summary = model.prequential_trace, model.prequential_summary
    assert [r["position"] for r in trace] == list(range(1, len(trace) + 1))
    assert trace[-1]["cumulative_text_bits"] == pytest.approx(sum(r["text_bits"] for r in trace))
    assert summary["total_text_bits"] == pytest.approx(sum(r["text_bits"] for r in trace))
    assert summary["total_label_bits"] == pytest.approx(sum(r["label_bits"] for r in trace))
    assert summary["total_joint_bits"] == pytest.approx(trace[-1]["cumulative_joint_bits"])
    assert sum(summary["text_bits_by_class"].values()) == pytest.approx(summary["total_text_bits"])
    assert all(r["text_bits"] == r["bits_by_class"][r["true_label"]] for r in trace)


def test_order_is_deterministic_and_runs_are_reproducible():
    df = _train_df()
    first = MDLPrequentialClassifier(seed=7).fit(df)
    second = MDLPrequentialClassifier(seed=7).fit(df.sample(frac=1.0, random_state=3).reset_index(drop=True))
    assert first.prequential_trace == second.prequential_trace
    assert first.prequential_summary == second.prequential_summary
    other = MDLPrequentialClassifier(seed=8).fit(df)
    assert [r["example_id"] for r in other.prequential_trace] != [r["example_id"] for r in first.prequential_trace]


def test_final_models_equal_batch_fit_and_reflect_all_updates():
    df = _train_df()
    batch = MDLClassifier().fit(df)
    model = MDLPrequentialClassifier(seed=0).fit(df)
    assert _counts(model) == _counts(batch)
    assert model._prediction_vocab_size == batch._prediction_vocab_size
    assert model.prequential_summary["n_updates"] == len(df)
    assert model.predict(HELD_OUT) == batch.predict(HELD_OUT)
    assert model.description_lengths(HELD_OUT) == batch.description_lengths(HELD_OUT)
    with_prior = MDLPrequentialClassifier(seed=0, use_prior=True).fit(df)
    assert with_prior.description_lengths(HELD_OUT) == pytest.approx(MDLPriorClassifier().fit(df).description_lengths(HELD_OUT))


def test_validation_and_test_data_never_update_models():
    df = _train_df()
    model = MDLPrequentialClassifier(seed=0).fit(df, validation_data=_held_out_df(label=3))
    reference = MDLPrequentialClassifier(seed=0).fit(df)
    assert _counts(model) == _counts(reference)
    assert model.prequential_trace == reference.prequential_trace
    before = _counts(model)
    model.predict(HELD_OUT)
    model.description_lengths(HELD_OUT)
    assert _counts(model) == before


def test_order_with_foreign_or_missing_ids_is_rejected():
    df = _train_df()
    ids = df["example_id"].tolist()
    with pytest.raises(ValueError):
        MDLPrequentialClassifier(order=ids[:-1] + ["val0"]).fit(df)
    with pytest.raises(ValueError):
        MDLPrequentialClassifier(order=ids[:-1]).fit(df)


def test_nested_prefix_order_is_a_prefix_across_sizes():
    sets = [["a", "b"], ["a", "b", "c", "d"], ["a", "b", "c", "d", "e", "f", "g"]]
    full = nested_prefix_order(sets, seed=42)
    assert sorted(full) == sorted(sets[-1])
    for i, ids in enumerate(sets):
        prefix = nested_prefix_order(sets[: i + 1], seed=42)
        assert full[: len(ids)] == prefix
        assert set(prefix) == set(ids)
    assert nested_prefix_order(sets, seed=42) == full
    with pytest.raises(ValueError):
        nested_prefix_order([["a", "b"], ["a", "c"]], seed=0)


def test_pad_is_shared_with_baseline():
    # Guards that variants and baseline code text identically.
    assert _pad(["a"], 3) == ["<s>", "<s>", "a", "</s>"]
