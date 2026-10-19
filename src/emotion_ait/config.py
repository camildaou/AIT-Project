"""Typed configuration loading for the emotion-ait pipeline.

Both collaborators (MDL and LSTM) should obtain all shared paths,
dataset identifiers, label mappings and sampling parameters through
:func:`load_config` so that the two implementations always agree on
the same data contract.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Repository root, computed from this file's location (src/emotion_ait/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Identifies the source dataset and the splits we expect it to expose."""

    hub_id: str
    hub_config: str
    splits: List[str]
    original_label_names: List[str]


@dataclasses.dataclass(frozen=True)
class TargetTaskConfig:
    """The fixed 4-class single-label subset of GoEmotions used in this project."""

    label_mapping: Dict[str, int]

    @property
    def target_emotions(self) -> List[str]:
        """Target emotion names ordered by their assigned integer id."""
        return [name for name, _ in sorted(self.label_mapping.items(), key=lambda kv: kv[1])]

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {idx: name for name, idx in self.label_mapping.items()}


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    subsets_dir: Path
    results_dir: Path


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    seed: int
    percentages: List[float]
    absolute_sizes: List[int]


@dataclasses.dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    target_task: TargetTaskConfig
    paths: PathsConfig
    sampling: SamplingConfig
    repo_root: Path


def load_config(path: Optional[Path] = None) -> Config:
    """Load the shared project configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to a YAML config file. Defaults to ``configs/default.yaml``
        at the repository root.

    Returns
    -------
    Config
        A fully-typed, immutable configuration object.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    dataset_cfg = DatasetConfig(
        hub_id=raw["dataset"]["hub_id"],
        hub_config=raw["dataset"]["hub_config"],
        splits=list(raw["dataset"]["splits"]),
        original_label_names=list(raw["dataset"]["original_label_names"]),
    )
    target_task_cfg = TargetTaskConfig(
        label_mapping=dict(raw["target_task"]["label_mapping"]),
    )

    def _resolve(rel: str) -> Path:
        candidate = Path(rel)
        if candidate.is_absolute():
            return candidate
        return REPO_ROOT / candidate

    paths_cfg = PathsConfig(
        raw_dir=_resolve(raw["paths"]["raw_dir"]),
        processed_dir=_resolve(raw["paths"]["processed_dir"]),
        subsets_dir=_resolve(raw["paths"]["subsets_dir"]),
        results_dir=_resolve(raw["paths"]["results_dir"]),
    )
    sampling_cfg = SamplingConfig(
        seed=int(raw["sampling"]["seed"]),
        percentages=[float(p) for p in raw["sampling"]["percentages"]],
        absolute_sizes=[int(s) for s in raw["sampling"]["absolute_sizes"]],
    )

    return Config(
        dataset=dataset_cfg,
        target_task=target_task_cfg,
        paths=paths_cfg,
        sampling=sampling_cfg,
        repo_root=REPO_ROOT,
    )
