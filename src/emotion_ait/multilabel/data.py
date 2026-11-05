"""Shared data API of the multi-label track -- the only module that model
code (MDL or LSTM) should import from :mod:`emotion_ait.multilabel`.

Every function returns plain pandas / numpy objects with no model-specific
processing (no tokenization, no vocabulary, no tensors), so both
developers get byte-identical inputs::

    from emotion_ait.multilabel.data import (
        TARGET_LABELS, get_multilabel_example_ids, get_multilabel_targets,
        load_multilabel_dataframe, load_multilabel_ladder_subset_dataframe,
    )

    train = load_multilabel_ladder_subset_dataframe(seed=42, size=500)
    val = load_multilabel_dataframe("validation")
    test = load_multilabel_dataframe("test")

    texts = train["text"].tolist()          # list[str]
    Y = get_multilabel_targets(train)       # (500, 4) int64, columns TARGET_LABELS
    ids = get_multilabel_example_ids(train) # list[str], == manifest order

Row order is deterministic: processed splits keep the raw GoEmotions order;
ladder subsets follow their manifest's ``example_ids`` order (the seeded,
stratified prefix order -- shuffle it yourself if your training needs it).
Use the training ladder only for fitting; validation for tuning and
threshold selection; test only for final reporting.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import MultilabelConfig, load_multilabel_config
from .preprocessing import multilabel_columns
from .sampling import manifest_path

#: Column order of every target matrix.
TARGET_LABELS: List[str] = ["joy", "sadness", "anger", "fear"]


def _config(config: Optional[MultilabelConfig]) -> MultilabelConfig:
    config = config if config is not None else load_multilabel_config()
    if config.target_labels != TARGET_LABELS:
        raise ValueError(f"Config target_labels {config.target_labels} != TARGET_LABELS {TARGET_LABELS}.")
    return config


def load_multilabel_dataframe(split: str, config: Optional[MultilabelConfig] = None) -> pd.DataFrame:
    """One full processed partition (``"train"``, ``"validation"`` or ``"test"``).

    Columns: ``example_id`` (str), ``text`` (str), ``joy``/``sadness``/
    ``anger``/``fear`` (int 0/1), ``label_combination`` (str),
    ``original_labels`` (str), ``original_split`` (str)."""
    config = _config(config)
    if split not in config.splits:
        raise ValueError(f"Unknown split {split!r}; expected one of {config.splits}.")
    path = config.processed_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/prepare_multilabel_data.py` first.")
    dtypes: Dict[str, object] = {c: str for c in multilabel_columns(TARGET_LABELS)}
    dtypes.update({label: "int64" for label in TARGET_LABELS})
    return pd.read_csv(path, dtype=dtypes, keep_default_na=False)


def load_multilabel_ladder_manifest(seed: int, size: int, config: Optional[MultilabelConfig] = None) -> Dict[str, object]:
    """The full manifest of one (seed, size) training subset."""
    config = _config(config)
    path = manifest_path(config, seed, size)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/generate_multilabel_ladder.py` first.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_multilabel_ladder_ids(seed: int, size: int, config: Optional[MultilabelConfig] = None) -> List[str]:
    """Ordered example ids of one (seed, size) training subset."""
    return list(load_multilabel_ladder_manifest(seed, size, config)["example_ids"])


def load_multilabel_ladder_subset_dataframe(seed: int, size: int, config: Optional[MultilabelConfig] = None) -> pd.DataFrame:
    """Rows of one (seed, size) training subset, in manifest order.

    Raises if the processed training data no longer matches the manifest
    (different ids or targets), so a stale dataset can never be used."""
    config = _config(config)
    manifest = load_multilabel_ladder_manifest(seed, size, config)
    train = load_multilabel_dataframe("train", config).set_index("example_id", drop=False)
    ids = manifest["example_ids"]
    missing = [i for i in ids if i not in train.index]
    if missing:
        raise ValueError(f"{len(missing)} manifest ids are missing from the processed training data; regenerate the ladder.")
    subset = train.loc[ids].reset_index(drop=True)
    if get_multilabel_targets(subset).tolist() != manifest["targets"]:
        raise ValueError("Processed training targets do not match the manifest; regenerate the ladder.")
    return subset


def get_multilabel_targets(df: pd.DataFrame) -> np.ndarray:
    """``(len(df), 4)`` int64 0/1 matrix, columns in :data:`TARGET_LABELS` order."""
    return df[TARGET_LABELS].to_numpy(dtype=np.int64)


def get_multilabel_example_ids(df: pd.DataFrame) -> List[str]:
    """Example ids in row order."""
    return df["example_id"].astype(str).tolist()
