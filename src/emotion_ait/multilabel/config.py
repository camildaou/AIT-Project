"""Typed loading of ``configs/multilabel.yaml``."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..config import REPO_ROOT, Config, load_config

DEFAULT_MULTILABEL_CONFIG_PATH = REPO_ROOT / "configs" / "multilabel.yaml"


@dataclasses.dataclass(frozen=True)
class MultilabelConfig:
    dataset_id: str
    target_labels: List[str]
    policy: str
    splits: List[str]
    processed_dir: Path
    subsets_dir: Path
    ladder_sizes: List[int]
    ladder_seeds: List[int]
    expected_counts: Dict[str, Dict[str, object]]
    #: The single-label project config, used only for the raw data location
    #: and the 28 original GoEmotions label names.
    base: Config


def load_multilabel_config(path: Optional[Path] = None, base_config_path: Optional[Path] = None) -> MultilabelConfig:
    """Load the multi-label config (default ``configs/multilabel.yaml``).

    Relative paths are resolved against the repository root."""
    path = Path(path) if path is not None else DEFAULT_MULTILABEL_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw["policy"] != "option_a":
        raise ValueError(f"Only policy 'option_a' is implemented, got {raw['policy']!r}.")

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    return MultilabelConfig(
        dataset_id=str(raw["dataset_id"]),
        target_labels=list(raw["target_labels"]),
        policy=raw["policy"],
        splits=list(raw["splits"]),
        processed_dir=_resolve(raw["paths"]["processed_dir"]),
        subsets_dir=_resolve(raw["paths"]["subsets_dir"]),
        ladder_sizes=[int(s) for s in raw["sampling"]["ladder_sizes"]],
        ladder_seeds=[int(s) for s in raw["sampling"]["ladder_seeds"]],
        expected_counts=dict(raw.get("expected_counts") or {}),
        base=load_config(base_config_path),
    )
