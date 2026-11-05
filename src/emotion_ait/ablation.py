"""Configuration of the MDL n-gram-order and smoothing ablations.

Every configuration is a plain :class:`~emotion_ait.mdl_classifier.MDLClassifier`
with a different ``n`` or ``k``; nothing else changes (tokenization,
shared vocabulary, ``<s>``/``</s>``/``<unk>`` handling, decision rule,
ladder subsets, validation/test splits).

The trigram ``k=1`` configuration *is* the existing baseline, so its
results are reused from ``results/metrics/mdl_ladder_results.csv`` (after
:func:`verify_reference_results` checks that they match the current
ladder) instead of being retrained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import Config
from .sampling import load_ladder_subset_ids

REFERENCE_MODEL_NAME = "mdl"
REFERENCE_CONFIG = (3, 1.0)
REFERENCE_CONFIG_SUMMARY = {"k": 1.0, "n": 3, "smoothing": "laplace"}

STUDIES: Dict[str, List[Tuple[int, float]]] = {
    "ngram": [(2, 1.0), (3, 1.0), (4, 1.0)],
    "smoothing": [(3, 0.1), (3, 0.5), (3, 1.0), (3, 2.0)],
}


def ablation_model_name(n: int, k: float) -> str:
    """Results name of one configuration: ``"mdl"`` for the reference
    (trigram, ``k=1``), else ``mdl_n<n>_k<k>`` with ``.`` written as ``p``
    (e.g. ``mdl_n3_k0p1``), so names never collide with each other or
    with the baseline's files."""
    if (n, float(k)) == REFERENCE_CONFIG:
        return REFERENCE_MODEL_NAME
    return f"mdl_n{n}_k{f'{float(k):g}'.replace('.', 'p')}"


def config_summary(n: int, k: float) -> Dict[str, object]:
    """Per-run configuration saved with the metrics (same keys as the baseline)."""
    return {"k": float(k), "n": int(n), "smoothing": "laplace"}


def study_configs(study: str) -> List[Tuple[str, int, float]]:
    """``(model_name, n, k)`` for every configuration of one study."""
    if study not in STUDIES:
        raise ValueError(f"Unknown study {study!r}; expected one of {sorted(STUDIES)}.")
    return [(ablation_model_name(n, k), n, k) for n, k in STUDIES[study]]


def results_path(config: Config, model_name: str, tag: Optional[str] = None) -> Path:
    """Combined held-out metrics CSV of one configuration."""
    suffix = f"_{tag}" if tag else ""
    return config.paths.results_dir / "metrics" / f"{model_name}_ladder_results{suffix}.csv"


def verify_reference_results(config: Config) -> pd.DataFrame:
    """Load the baseline ladder results and check they can stand in for
    the trigram ``k=1`` configuration: exactly one row per configured
    (size, seed), baseline config summary, training sizes equal to the
    current ladder manifests, validation macro-F1 present, and a test
    predictions file for every run. Raises ``ValueError`` otherwise."""
    path = results_path(config, REFERENCE_MODEL_NAME)
    if not path.exists():
        raise ValueError(f"Reference results {path} not found; run the baseline ladder first.")
    df = pd.read_csv(path)
    problems = []
    if set(df["model_name"]) != {REFERENCE_MODEL_NAME}:
        problems.append(f"unexpected model names {sorted(set(df['model_name']))}")
    summaries = {json.dumps(json.loads(s), sort_keys=True) for s in df["config_summary"]}
    if summaries != {json.dumps(REFERENCE_CONFIG_SUMMARY, sort_keys=True)}:
        problems.append(f"config summaries {sorted(summaries)} != {REFERENCE_CONFIG_SUMMARY}")
    expected = {(f"ladder_n{size}", seed) for size in config.sampling.ladder_sizes for seed in config.sampling.ladder_seeds}
    observed = list(zip(df["train_subset_name"], df["seed"]))
    if len(observed) != len(set(observed)) or set(observed) != expected:
        problems.append("rows are not exactly one per configured (size, seed)")
    for size in config.sampling.ladder_sizes:
        for seed in config.sampling.ladder_seeds:
            rows = df[(df["train_subset_name"] == f"ladder_n{size}") & (df["seed"] == seed)]
            manifest_size = len(load_ladder_subset_ids(seed, size, config))
            if len(rows) == 1 and int(rows["train_subset_size"].iloc[0]) != manifest_size:
                problems.append(f"size {size} seed {seed}: trained on {int(rows['train_subset_size'].iloc[0])} != manifest {manifest_size}")
            predictions = config.paths.results_dir / "predictions" / f"{REFERENCE_MODEL_NAME}_ladder_n{size}_{seed}.csv"
            if not predictions.exists():
                problems.append(f"missing {predictions.name}")
    if df["validation_macro_f1"].isna().any():
        problems.append("missing validation_macro_f1 values")
    if problems:
        raise ValueError(f"Baseline results in {path} cannot be reused as the reference: " + "; ".join(problems))
    return df
