"""Reproducible loading of the raw GoEmotions dataset.

This module is the *only* place that talks to the Hugging Face ``datasets``
library. Downstream code (preprocessing, sampling, and eventually the MDL
and LSTM classifiers) should never call ``load_dataset`` directly -- this
keeps the dataset source a single, auditable source of truth.

Dataset identity (verified against the Hugging Face Hub on 2026-10-19):

* Hub repository: ``google-research-datasets/go_emotions``
* Config name: ``simplified`` (the config that exposes official
  train/validation/test splits with multi-label integer ids; the sibling
  ``raw`` config has only a single ``train`` split of per-rater rows and
  is not used here).
* Splits: ``train`` (43,410 rows), ``validation`` (5,426 rows), ``test``
  (5,427 rows) -- confirmed via the Hugging Face datasets-server API and
  by actually downloading the dataset with ``datasets==4.5.0``.
* Columns: ``text`` (str), ``labels`` (list[int], indices into the
  28-name ``original_label_names`` list in ``configs/default.yaml``),
  ``id`` (str, a stable GoEmotions/Reddit comment id suitable as a
  unique source identifier).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def _raw_split_path(raw_dir: Path, split: str) -> Path:
    return raw_dir / f"{split}.parquet"


def download_raw_dataset(config: Config, force: bool = False) -> Dict[str, pd.DataFrame]:
    """Download (or load from the local HF cache) the raw GoEmotions splits
    and persist them as parquet files under ``data/raw`` for fast, offline
    reuse and auditability.

    Parameters
    ----------
    config:
        Shared project configuration (see :mod:`emotion_ait.config`).
    force:
        If True, re-download and overwrite any cached parquet files even if
        they already exist.

    Returns
    -------
    dict
        Mapping from split name to a DataFrame with columns
        ``text``, ``labels`` (list[int]), ``id``, ``original_split``.
    """
    # Local import so that merely importing this module does not require
    # the (optional, heavier) `datasets` dependency unless actually needed.
    from datasets import load_dataset

    config.paths.raw_dir.mkdir(parents=True, exist_ok=True)

    cached_paths = {split: _raw_split_path(config.paths.raw_dir, split) for split in config.dataset.splits}
    if not force and all(p.exists() for p in cached_paths.values()):
        logger.info("Using cached raw data in %s", config.paths.raw_dir)
        return load_raw_dataset(config)

    logger.info(
        "Downloading %s (config=%s) from the Hugging Face Hub",
        config.dataset.hub_id,
        config.dataset.hub_config,
    )
    hf_dataset = load_dataset(config.dataset.hub_id, config.dataset.hub_config)

    frames: Dict[str, pd.DataFrame] = {}
    for split in config.dataset.splits:
        if split not in hf_dataset:
            raise KeyError(
                f"Expected split '{split}' not present in dataset "
                f"'{config.dataset.hub_id}' (config '{config.dataset.hub_config}'). "
                f"Available splits: {list(hf_dataset.keys())}"
            )
        df = hf_dataset[split].to_pandas()
        df["original_split"] = split
        frames[split] = df
        out_path = _raw_split_path(config.paths.raw_dir, split)
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d raw rows for split '%s' to %s", len(df), split, out_path)

    return frames


def load_raw_dataset(config: Config) -> Dict[str, pd.DataFrame]:
    """Load the raw GoEmotions splits from the local parquet cache.

    Raises
    ------
    FileNotFoundError
        If the cache has not been populated yet; call
        :func:`download_raw_dataset` first (e.g. via
        ``scripts/prepare_data.py``).
    """
    frames: Dict[str, pd.DataFrame] = {}
    for split in config.dataset.splits:
        path = _raw_split_path(config.paths.raw_dir, split)
        if not path.exists():
            raise FileNotFoundError(
                f"Raw cache not found for split '{split}' at {path}. "
                "Run `python scripts/prepare_data.py` first."
            )
        frames[split] = pd.read_parquet(path)
    return frames
