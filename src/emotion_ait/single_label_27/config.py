"""Typed loading of ``configs/single_label_27.yaml``."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..config import REPO_ROOT, Config, load_config

DEFAULT_SINGLE_LABEL_27_CONFIG_PATH = REPO_ROOT / "configs" / "single_label_27.yaml"
NEUTRAL = "neutral"


@dataclasses.dataclass(frozen=True)
class SingleLabel27Config:
    dataset_id: str
    #: The 27 classes: original GoEmotions order with ``neutral`` removed.
    class_names: List[str]
    splits: List[str]
    processed_dir: Path
    subsets_dir: Path
    ladder_sizes: List[int]
    ladder_seeds: List[int]
    require_all_classes: bool
    expected_counts: Dict[str, Dict[str, object]]
    #: The project base config, used only for the raw data location and the
    #: 28 original GoEmotions label names.
    base: Config


def load_single_label_27_config(path: Optional[Path] = None, base_config_path: Optional[Path] = None) -> SingleLabel27Config:
    """Load the 27-class config (default ``configs/single_label_27.yaml``);
    relative paths are resolved against the repository root."""
    path = Path(path) if path is not None else DEFAULT_SINGLE_LABEL_27_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    base = load_config(base_config_path)

    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    return SingleLabel27Config(
        dataset_id=str(raw["dataset_id"]),
        class_names=[name for name in base.dataset.original_label_names if name != NEUTRAL],
        splits=list(raw["splits"]),
        processed_dir=_resolve(raw["paths"]["processed_dir"]),
        subsets_dir=_resolve(raw["paths"]["subsets_dir"]),
        ladder_sizes=[int(s) for s in raw["sampling"]["ladder_sizes"]],
        ladder_seeds=[int(s) for s in raw["sampling"]["ladder_seeds"]],
        require_all_classes=bool(raw["sampling"].get("require_all_classes", True)),
        expected_counts=dict(raw.get("expected_counts") or {}),
        base=base,
    )
