"""Shared experiment runner for the 27-class single-label track.

Both the MDL and the LSTM implementation are run through this one function,
so they are trained on the same ladder subsets and scored by the same
evaluation code (:func:`emotion_ait.experiments.run_experiment` with the
27-class mapping): validation and test macro-F1, accuracy (= micro-F1 for
single-label predictions), per-class precision/recall/F1 for all 27 classes,
train/inference time, and a per-run test predictions file.

Naming keeps every file distinct from the other tracks: model names are
prefixed ``sl27_`` and results go to
``results/metrics/sl27_<name>_ladder_results[_<tag>].csv`` plus the usual
per-run ``sl27_<name>_ladder_n<size>_<seed>.*`` files. A run refuses to start
if its combined results CSV already exists.

With ``diagnostics=True``, a model exposing
``diagnostics(train_df, {"validation": texts, "test": texts}) -> dict`` (e.g.
:class:`~emotion_ait.single_label_27.mdl.MDL27Classifier`) also gets one JSON
report per run in ``results/diagnostics/sl27_<name>_ladder_n<size>_<seed>.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..evaluation import ResultRecord
from ..experiments import run_experiment
from ..interfaces import EmotionClassifier
from .config import SingleLabel27Config
from .data import (
    INDEX_TO_LABEL,
    load_single_label_27_dataframe,
    load_single_label_27_ladder_manifest,
    load_single_label_27_ladder_subset_dataframe,
)

logger = logging.getLogger(__name__)

#: ``build(seed) -> (fresh unfitted model, config_summary dict)``.
ModelBuilder = Callable[[int], Tuple[EmotionClassifier, Dict[str, object]]]
MODEL_PREFIX = "sl27_"


def run_name(name: str) -> str:
    """Prefixed model name used in every results file of this track."""
    return name if name.startswith(MODEL_PREFIX) else f"{MODEL_PREFIX}{name}"


def results_path(config: SingleLabel27Config, name: str, tag: Optional[str] = None) -> Path:
    suffix = f"_{tag}" if tag else ""
    return config.base.paths.results_dir / "metrics" / f"{run_name(name)}_ladder_results{suffix}.csv"


def load_all_splits(config: SingleLabel27Config) -> pd.DataFrame:
    """The three processed partitions, concatenated (``original_split`` kept)."""
    return pd.concat([load_single_label_27_dataframe(s, config) for s in config.splits], ignore_index=True)


def run_single_label_27_ladder(
    build: ModelBuilder,
    name: str,
    config: SingleLabel27Config,
    sizes: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
    results_tag: Optional[str] = None,
    diagnostics: bool = False,
) -> List[ResultRecord]:
    """Train and evaluate one model configuration on every (seed, size)."""
    path = results_path(config, name, results_tag)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite/append existing results {path}; move it away or use a results tag.")
    sizes = list(sizes or config.ladder_sizes)
    seeds = list(seeds or config.ladder_seeds)
    unknown = set(sizes) - set(config.ladder_sizes)
    if unknown:
        raise ValueError(f"Sizes {sorted(unknown)} are not ladder sizes {config.ladder_sizes}.")

    processed = load_all_splits(config)
    held_out = {s: processed.loc[processed["original_split"] == s, "text"].tolist() for s in ("validation", "test")}
    records = []
    for seed in seeds:
        for size in sizes:
            train_df = load_single_label_27_ladder_subset_dataframe(seed, size, config)
            manifest = load_single_label_27_ladder_manifest(seed, size, config)
            model, summary = build(seed)
            summary = {
                **summary,
                "track": "single_label_27",
                "dataset_id": config.dataset_id,
                "subset_ids_sha256": manifest["example_ids_sha256"],
            }
            record = run_experiment(
                model=model,
                model_name=run_name(name),
                subset_name=f"ladder_n{size}",
                processed_df=processed,
                config=config.base,
                config_summary=summary,
                train_df=train_df,
                seed=seed,
                results_filename=path.name,
                id_to_label=INDEX_TO_LABEL,
            )
            logger.info("%s seed=%d size=%d: val_macro_f1=%.4f test_macro_f1=%.4f test_accuracy=%.4f",
                        run_name(name), seed, size, record.validation_macro_f1, record.test_macro_f1, record.test_accuracy)
            if diagnostics and hasattr(model, "diagnostics"):
                out = config.base.paths.results_dir / "diagnostics" / f"{run_name(name)}_ladder_n{size}_{seed}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w", encoding="utf-8") as fh:
                    json.dump(model.diagnostics(train_df, held_out), fh, indent=2)
                logger.info("Wrote diagnostics %s", out)
            records.append(record)
    return records
