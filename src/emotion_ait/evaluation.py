"""Shared evaluation protocol and results schema.

Both the MDL and LSTM classifiers are scored with exactly the same
function (:func:`compute_metrics`) and logged with the same schema
(:class:`ResultRecord`), so that results are directly comparable
regardless of which collaborator produced them.
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Dict[int, str],
) -> Dict[str, object]:
    """Compute the shared evaluation metrics for one set of predictions.

    Parameters
    ----------
    y_true, y_pred:
        True and predicted integer class ids.
    label_names:
        Mapping from integer class id to a human-readable name, used to
        label per-class metrics and the confusion matrix. All ids in
        ``label_names`` are always reported, even if a class is absent
        from both ``y_true`` and ``y_pred`` (e.g. for tiny subsets) --
        in that case precision/recall/F1 are reported as 0.0 rather
        than raising, using scikit-learn's ``zero_division=0``.

    Returns
    -------
    dict
        ``accuracy``, ``macro_f1``, ``per_class`` (precision/recall/f1/
        support per class), and ``confusion_matrix`` (list of lists,
        row = true class, column = predicted class, ordered by sorted
        class id).
    """
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute metrics on an empty prediction set.")

    ordered_ids = sorted(label_names.keys())
    accuracy = float(np.mean(y_true_arr == y_pred_arr))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=ordered_ids, average=None, zero_division=0
    )
    macro_f1 = float(np.mean(f1))

    per_class = {}
    for idx, class_id in enumerate(ordered_ids):
        per_class[label_names[class_id]] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=ordered_ids)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [label_names[i] for i in ordered_ids],
    }


class Timer:
    """Tiny context manager for recording wall-clock durations in seconds.

    Usage::

        with Timer() as t:
            model.fit(train_data)
        train_time_seconds = t.elapsed
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed: Optional[float] = None
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed = time.perf_counter() - self._start


@dataclasses.dataclass
class ResultRecord:
    """One row of the shared, machine-readable results schema.

    Every field required by the project spec is represented explicitly;
    ``extra`` can hold additional free-form metadata without breaking
    the fixed schema.
    """

    model_name: str
    train_subset_name: str
    train_subset_percentage: Optional[float]
    train_subset_size: int
    seed: int
    config_summary: Dict[str, object]
    test_accuracy: float
    test_macro_f1: float
    test_per_class: Dict[str, Dict[str, float]]
    train_time_seconds: float
    inference_time_seconds: float
    validation_macro_f1: Optional[float] = None
    timestamp: str = dataclasses.field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())

    def to_flat_dict(self) -> Dict[str, object]:
        """Flatten nested fields to JSON strings so the record can be
        written as a single CSV row."""
        flat = dataclasses.asdict(self)
        flat["config_summary"] = json.dumps(flat["config_summary"], sort_keys=True)
        flat["test_per_class"] = json.dumps(flat["test_per_class"], sort_keys=True)
        return flat


def append_result(record: ResultRecord, results_dir: Path, filename: str = "results.csv") -> Path:
    """Append one result record to the shared, combined results CSV,
    creating it (with header) on first use. Also mirrors every record
    individually under ``results_dir``:

    * ``json/<model>_<subset>_<seed>.json`` -- the full record as JSON.
    * ``csv/<model>_<subset>_<seed>.csv`` -- the same flattened row as a
      single-row CSV, so one subset's metrics can be inspected (or
      diffed/reloaded) without reading the combined CSV.

    Both per-run mirrors are overwritten on re-run (unlike the combined
    CSV, which only ever grows)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / filename
    flat = record.to_flat_dict()
    row_df = pd.DataFrame([flat])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(csv_path, index=False)

    run_name = f"{record.model_name}_{record.train_subset_name}_{record.seed}"

    json_dir = results_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    with open(json_dir / f"{run_name}.json", "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(record), fh, indent=2)

    per_run_csv_dir = results_dir / "csv"
    per_run_csv_dir.mkdir(parents=True, exist_ok=True)
    row_df.to_csv(per_run_csv_dir / f"{run_name}.csv", index=False)

    return csv_path


def warn_if_missing_classes(y_true: Sequence[int], label_names: Dict[int, str]) -> None:
    """Emit an explicit warning when some expected classes are absent
    from a (typically tiny) evaluation set, rather than silently
    computing misleading macro-averages."""
    present = set(y_true)
    missing = [label_names[i] for i in sorted(label_names) if i not in present]
    if missing:
        warnings.warn(f"Evaluation set is missing classes: {missing}. Their metrics will be 0.0.")


def save_predictions(
    model_name: str,
    subset_name: str,
    seed: int,
    example_ids: Sequence[str],
    texts: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Dict[int, str],
    results_dir: Path,
    description_lengths: Optional[Sequence[Dict[int, float]]] = None,
) -> Path:
    """Persist one row per evaluated example for qualitative/error
    analysis: ``example_id``, ``text``, true/predicted label (id and
    name), a ``correct`` flag, and -- when the model exposes a
    ``description_lengths`` diagnostic (e.g. :class:`MDLClassifier`) --
    one ``bits_<label_name>`` column per candidate class, so individual
    misclassifications can be inspected alongside their scores.

    Written to ``results_dir / "predictions" / "<model_name>_<subset_name>_<seed>.csv"``,
    overwriting any previous file for the same (model, subset, seed)
    combination (unlike the aggregate metrics CSV, this is a per-run
    snapshot rather than an append-only log).
    """
    if not (len(example_ids) == len(texts) == len(y_true) == len(y_pred)):
        raise ValueError("example_ids, texts, y_true, and y_pred must all have the same length.")

    rows = []
    for i, example_id in enumerate(example_ids):
        true_id = int(y_true[i])
        pred_id = int(y_pred[i])
        row = {
            "example_id": example_id,
            "text": texts[i],
            "true_label": true_id,
            "true_label_name": label_names[true_id],
            "predicted_label": pred_id,
            "predicted_label_name": label_names[pred_id],
            "correct": true_id == pred_id,
        }
        if description_lengths is not None:
            for class_id, bits in sorted(description_lengths[i].items()):
                row[f"bits_{label_names[class_id]}"] = bits
        rows.append(row)

    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    out_path = predictions_dir / f"{model_name}_{subset_name}_{seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
