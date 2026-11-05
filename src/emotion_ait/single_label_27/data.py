"""Shared data API of the 27-class single-label track -- the only module that
model code (MDL or LSTM) should import from :mod:`emotion_ait.single_label_27`.

Returns plain pandas / numpy objects with no model-specific processing::

    from emotion_ait.single_label_27.data import (
        CLASS_NAMES, LABEL_TO_INDEX, INDEX_TO_LABEL,
        load_single_label_27_dataframe, load_single_label_27_ladder_subset_dataframe,
        get_single_label_27_labels, get_single_label_27_example_ids,
    )

    train = load_single_label_27_ladder_subset_dataframe(seed=42, size=2000)
    val = load_single_label_27_dataframe("validation")
    texts = train["text"].tolist()             # list[str]
    y = get_single_label_27_labels(train)      # (2000,) int64 class indices 0..26
    ids = get_single_label_27_example_ids(train)

Class indices are fixed: the 27 GoEmotions emotions in original dataset order
with ``neutral`` removed (``admiration`` = 0 ... ``surprise`` = 26), identical
for every split, subset and experiment. Row order is deterministic: processed
splits keep the raw order, ladder subsets follow their manifest order
(stratified prefix order -- shuffle in your own training code if needed).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import SingleLabel27Config, load_single_label_27_config
from .preprocessing import PROCESSED_COLUMNS
from .sampling import manifest_path

#: The 27 classes, in class-index order.
CLASS_NAMES: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise",
]
LABEL_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL: Dict[int, str] = dict(enumerate(CLASS_NAMES))


def _config(config: Optional[SingleLabel27Config]) -> SingleLabel27Config:
    config = config if config is not None else load_single_label_27_config()
    if config.class_names != CLASS_NAMES:
        raise ValueError(f"Config classes {config.class_names} != CLASS_NAMES {CLASS_NAMES}.")
    return config


def load_single_label_27_dataframe(split: str, config: Optional[SingleLabel27Config] = None) -> pd.DataFrame:
    """One full processed partition (``"train"``, ``"validation"`` or ``"test"``).

    Columns: ``example_id`` (str), ``text`` (str), ``label`` (int 0..26),
    ``label_name`` (str), ``original_labels`` (str), ``original_split`` (str)."""
    config = _config(config)
    if split not in config.splits:
        raise ValueError(f"Unknown split {split!r}; expected one of {config.splits}.")
    path = config.processed_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/prepare_single_label_27_data.py` first.")
    dtypes: Dict[str, object] = {c: str for c in PROCESSED_COLUMNS}
    dtypes["label"] = "int64"
    return pd.read_csv(path, dtype=dtypes, keep_default_na=False)


def load_single_label_27_ladder_manifest(seed: int, size: int, config: Optional[SingleLabel27Config] = None) -> Dict[str, object]:
    """The full manifest of one (seed, size) training subset."""
    config = _config(config)
    path = manifest_path(config, seed, size)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/generate_single_label_27_ladder.py` first.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_single_label_27_ladder_ids(seed: int, size: int, config: Optional[SingleLabel27Config] = None) -> List[str]:
    """Ordered example ids of one (seed, size) training subset."""
    return list(load_single_label_27_ladder_manifest(seed, size, config)["example_ids"])


def load_single_label_27_ladder_subset_dataframe(seed: int, size: int, config: Optional[SingleLabel27Config] = None) -> pd.DataFrame:
    """Rows of one (seed, size) training subset, in manifest order. Raises if
    the processed training data no longer matches the manifest."""
    config = _config(config)
    manifest = load_single_label_27_ladder_manifest(seed, size, config)
    train = load_single_label_27_dataframe("train", config).set_index("example_id", drop=False)
    ids = manifest["example_ids"]
    missing = [i for i in ids if i not in train.index]
    if missing:
        raise ValueError(f"{len(missing)} manifest ids are missing from the processed training data; regenerate the ladder.")
    subset = train.loc[ids].reset_index(drop=True)
    if subset["label_name"].tolist() != manifest["labels"]:
        raise ValueError("Processed training labels do not match the manifest; regenerate the ladder.")
    return subset


def get_single_label_27_labels(df: pd.DataFrame) -> np.ndarray:
    """``(len(df),)`` int64 class indices (see :data:`CLASS_NAMES`)."""
    return df["label"].to_numpy(dtype=np.int64)


def get_single_label_27_example_ids(df: pd.DataFrame) -> List[str]:
    """Example ids in row order."""
    return df["example_id"].astype(str).tolist()
