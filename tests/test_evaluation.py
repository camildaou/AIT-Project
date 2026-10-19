"""Tests for emotion_ait.evaluation."""

from __future__ import annotations

import pytest

from emotion_ait.evaluation import (  # noqa: E402
    ResultRecord,
    Timer,
    append_result,
    compute_metrics,
    save_predictions,
    warn_if_missing_classes,
)

LABEL_NAMES = {0: "anger", 1: "fear", 2: "joy", 3: "sadness"}


def test_compute_metrics_basic_correctness():
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 0, 1, 0, 2, 2, 3, 2]
    metrics = compute_metrics(y_true, y_pred, LABEL_NAMES)

    assert metrics["accuracy"] == pytest.approx(6 / 8)
    assert set(metrics["per_class"].keys()) == {"anger", "fear", "joy", "sadness"}
    assert metrics["per_class"]["anger"]["recall"] == pytest.approx(1.0)
    assert metrics["per_class"]["fear"]["recall"] == pytest.approx(0.5)
    assert len(metrics["confusion_matrix"]) == 4
    assert metrics["confusion_matrix_labels"] == ["anger", "fear", "joy", "sadness"]


def test_compute_metrics_handles_missing_predicted_classes_without_raising():
    # Model never predicts "sadness"; must not crash, must report 0s.
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 2]
    metrics = compute_metrics(y_true, y_pred, LABEL_NAMES)
    assert metrics["per_class"]["sadness"]["precision"] == 0.0
    assert metrics["per_class"]["sadness"]["recall"] == 0.0
    assert metrics["per_class"]["sadness"]["f1"] == 0.0


def test_compute_metrics_handles_tiny_subsets_with_a_single_class_present():
    y_true = [2, 2, 2]
    y_pred = [2, 2, 0]
    metrics = compute_metrics(y_true, y_pred, LABEL_NAMES)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["fear"]["support"] == 0
    assert metrics["per_class"]["fear"]["f1"] == 0.0


def test_compute_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_metrics([], [], LABEL_NAMES)


def test_warn_if_missing_classes_warns_exactly_for_absent_classes():
    with pytest.warns(UserWarning, match="sadness"):
        warn_if_missing_classes([0, 1, 2], LABEL_NAMES)


def test_warn_if_missing_classes_silent_when_all_present():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_missing_classes([0, 1, 2, 3], LABEL_NAMES)


def test_timer_records_a_nonnegative_elapsed_time():
    with Timer() as t:
        total = sum(range(1000))
    assert total == sum(range(1000))
    assert t.elapsed is not None
    assert t.elapsed >= 0.0


def test_append_result_writes_csv_and_json(tmp_path):
    record = ResultRecord(
        model_name="majority_baseline",
        train_subset_name="train_pct10",
        train_subset_percentage=10.0,
        train_subset_size=50,
        seed=42,
        config_summary={"note": "unit test"},
        test_accuracy=0.4,
        test_macro_f1=0.3,
        test_per_class={"joy": {"precision": 0.4, "recall": 1.0, "f1": 0.57, "support": 10}},
        train_time_seconds=0.01,
        inference_time_seconds=0.001,
        validation_macro_f1=0.25,
    )
    csv_path = append_result(record, tmp_path)
    assert csv_path.exists()

    import pandas as pd

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["model_name"] == "majority_baseline"

    json_files = list((tmp_path / "json").glob("*.json"))
    assert len(json_files) == 1

    per_run_csv = tmp_path / "csv" / "majority_baseline_train_pct10_42.csv"
    assert per_run_csv.exists()
    per_run_df = pd.read_csv(per_run_csv)
    assert len(per_run_df) == 1
    assert per_run_df.iloc[0]["train_subset_name"] == "train_pct10"
    assert per_run_df.iloc[0]["test_accuracy"] == pytest.approx(0.4)


def test_append_result_appends_multiple_rows(tmp_path):
    for subset_pct in (5.0, 10.0):
        record = ResultRecord(
            model_name="majority_baseline",
            train_subset_name=f"train_pct{int(subset_pct)}",
            train_subset_percentage=subset_pct,
            train_subset_size=25,
            seed=42,
            config_summary={},
            test_accuracy=0.3,
            test_macro_f1=0.2,
            test_per_class={},
            train_time_seconds=0.01,
            inference_time_seconds=0.001,
        )
        append_result(record, tmp_path)

    import pandas as pd

    df = pd.read_csv(tmp_path / "results.csv")
    assert len(df) == 2

    # The combined CSV accumulates rows, but each subset also gets its
    # own standalone per-run metrics CSV (one row each, not merged).
    per_run_csvs = sorted(p.name for p in (tmp_path / "csv").glob("*.csv"))
    assert per_run_csvs == ["majority_baseline_train_pct10_42.csv", "majority_baseline_train_pct5_42.csv"]


def test_save_predictions_writes_one_row_per_example(tmp_path):
    out_path = save_predictions(
        model_name="mdl",
        subset_name="train_abs100",
        seed=42,
        example_ids=["ex1", "ex2"],
        texts=["i am happy", "i am sad"],
        y_true=[2, 3],
        y_pred=[2, 2],
        label_names=LABEL_NAMES,
        results_dir=tmp_path,
    )

    assert out_path == tmp_path / "predictions" / "mdl_train_abs100_42.csv"
    assert out_path.exists()

    import pandas as pd

    df = pd.read_csv(out_path)
    assert len(df) == 2
    assert list(df["true_label_name"]) == ["joy", "sadness"]
    assert list(df["predicted_label_name"]) == ["joy", "joy"]
    assert list(df["correct"]) == [True, False]


def test_save_predictions_includes_description_length_columns_when_provided(tmp_path):
    out_path = save_predictions(
        model_name="mdl",
        subset_name="train_abs100",
        seed=42,
        example_ids=["ex1"],
        texts=["i am happy"],
        y_true=[2],
        y_pred=[2],
        label_names=LABEL_NAMES,
        results_dir=tmp_path,
        description_lengths=[{0: 12.5, 1: 15.0, 2: 3.2, 3: 10.1}],
    )

    import pandas as pd

    df = pd.read_csv(out_path)
    assert df.iloc[0]["bits_anger"] == pytest.approx(12.5)
    assert df.iloc[0]["bits_joy"] == pytest.approx(3.2)


def test_save_predictions_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        save_predictions(
            model_name="mdl",
            subset_name="train_abs100",
            seed=42,
            example_ids=["ex1", "ex2"],
            texts=["only one text"],
            y_true=[2, 3],
            y_pred=[2, 3],
            label_names=LABEL_NAMES,
            results_dir=tmp_path,
        )


def test_save_predictions_overwrites_previous_run_for_same_key(tmp_path):
    save_predictions(
        model_name="mdl",
        subset_name="train_abs100",
        seed=42,
        example_ids=["ex1", "ex2", "ex3"],
        texts=["a", "b", "c"],
        y_true=[0, 1, 2],
        y_pred=[0, 1, 2],
        label_names=LABEL_NAMES,
        results_dir=tmp_path,
    )
    out_path = save_predictions(
        model_name="mdl",
        subset_name="train_abs100",
        seed=42,
        example_ids=["ex1"],
        texts=["a"],
        y_true=[0],
        y_pred=[0],
        label_names=LABEL_NAMES,
        results_dir=tmp_path,
    )

    import pandas as pd

    df = pd.read_csv(out_path)
    assert len(df) == 1
