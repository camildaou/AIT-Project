"""Nested, class-stratified training subsets for the 27-class single-label track.

Same mechanism as the multi-label track (:mod:`emotion_ait.multilabel.sampling`),
with the 27 classes as strata, in class-index order:

1. within each class, ids are sorted and permuted with
   ``numpy.random.RandomState(seed)``;
2. one global order of all training ids is built by sequential Webster /
   Sainte-Lague allocation (slot goes to the class maximising
   ``N_c / (a_c + 0.5)``; ties to the lower class index);
3. the subset of size ``n`` is the first ``n`` ids of that order.

So subsets have exactly ``n`` examples, are nested within a seed, follow the
training class distribution (each class within +-1 of ``n * N_c / N``), and
the full size is the whole training partition. There is no per-class
minimum: with 39 grief examples out of 23,485, all 27 classes are present
from ``n = 303`` onwards, and the ladder starts above that (validated when
``require_all_classes`` is set).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..multilabel.sampling import webster_sequence
from .config import SingleLabel27Config

STRATIFICATION = "class_webster_sequential"


def class_permutations(train_df: pd.DataFrame, seed: int, class_names: Sequence[str]) -> Dict[str, List[str]]:
    """Seeded order of example ids within each class present in ``train_df``."""
    permutations = {}
    for name in class_names:
        ids = sorted(train_df.loc[train_df["label_name"] == name, "example_id"].tolist())
        if ids:
            order = np.random.RandomState(seed).permutation(len(ids))
            permutations[name] = [ids[i] for i in order]
    return permutations


def nested_order(train_df: pd.DataFrame, seed: int, class_names: Sequence[str]) -> List[str]:
    """Global seeded order of all training ids; subset ``n`` = first ``n``."""
    permutations = class_permutations(train_df, seed, class_names)
    classes = list(permutations)
    next_index = {c: 0 for c in classes}
    order = []
    for c in webster_sequence({c: len(ids) for c, ids in permutations.items()}, classes):
        order.append(permutations[c][next_index[c]])
        next_index[c] += 1
    return order


def _sha256_lines(lines: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def train_partition_fingerprint(train_df: pd.DataFrame) -> str:
    """Hash of the processed training ids and labels, in file order."""
    return _sha256_lines([f"{i}\t{label}" for i, label in zip(train_df["example_id"], train_df["label"])])


def build_ladder_manifests(train_df: pd.DataFrame, config: SingleLabel27Config) -> Dict[int, Dict[int, Dict[str, object]]]:
    """``{seed: {size: manifest}}`` for every configured seed and size."""
    if max(config.ladder_sizes) != len(train_df):
        raise ValueError(f"The largest ladder size must be the full training partition ({len(train_df)}), got {config.ladder_sizes}.")
    label_by_id = dict(zip(train_df["example_id"], train_df["label_name"]))
    fingerprint = train_partition_fingerprint(train_df)
    manifests: Dict[int, Dict[int, Dict[str, object]]] = {}
    for seed in config.ladder_seeds:
        order = nested_order(train_df, seed, config.class_names)
        manifests[seed] = {}
        for size in sorted(config.ladder_sizes):
            ids = order[:size]
            labels = [label_by_id[i] for i in ids]
            counts = {name: labels.count(name) for name in config.class_names}
            manifests[seed][size] = {
                "dataset_id": config.dataset_id,
                "class_names": list(config.class_names),
                "stratification": STRATIFICATION,
                "seed": seed,
                "requested_size": size,
                "actual_size": len(ids),
                "train_partition_size": len(train_df),
                "train_partition_sha256": fingerprint,
                "example_ids_sha256": _sha256_lines(ids),
                "class_counts": counts,
                "classes_present": sum(1 for v in counts.values() if v > 0),
                "example_ids": ids,
                "labels": labels,
            }
    return manifests


def validate_ladder_manifests(manifests: Dict[int, Dict[int, Dict[str, object]]], train_df: pd.DataFrame, config: SingleLabel27Config) -> None:
    """Raise ``ValueError`` listing every violated ladder invariant."""
    train_ids = set(train_df["example_id"])
    label_by_id = dict(zip(train_df["example_id"], train_df["label_name"]))
    present_in_train = set(train_df["label_name"])
    problems = []
    for seed, per_size in manifests.items():
        previous: List[str] = []
        for size in sorted(per_size):
            m = per_size[size]
            ids = m["example_ids"]
            if len(ids) != size or m["actual_size"] != size:
                problems.append(f"seed {seed} n{size}: {len(ids)} ids")
            if len(set(ids)) != len(ids):
                problems.append(f"seed {seed} n{size}: duplicate ids")
            if not set(ids) <= train_ids:
                problems.append(f"seed {seed} n{size}: {len(set(ids) - train_ids)} ids outside the training partition")
            if ids[: len(previous)] != previous:
                problems.append(f"seed {seed} n{size}: smaller subset is not a prefix")
            if m["labels"] != [label_by_id.get(i) for i in ids]:
                problems.append(f"seed {seed} n{size}: labels do not match the processed training data")
            missing = present_in_train - {name for name, count in m["class_counts"].items() if count > 0}
            if config.require_all_classes and missing:
                problems.append(f"seed {seed} n{size}: classes missing {sorted(missing)}")
            previous = ids
        full = per_size[max(per_size)]["example_ids"]
        if set(full) != train_ids or len(full) != len(train_ids):
            problems.append(f"seed {seed}: full-size subset does not contain every training example exactly once")
    if problems:
        raise ValueError("Invalid 27-class ladder: " + "; ".join(problems))


def manifest_path(config: SingleLabel27Config, seed: int, size: int) -> Path:
    return config.subsets_dir / f"seed_{seed}" / f"n{size}.json"


def write_ladder_manifests(manifests: Dict[int, Dict[int, Dict[str, object]]], config: SingleLabel27Config) -> None:
    """Write per-(seed, size) manifests plus per-seed and combined summaries
    (the summaries omit ids/labels)."""
    combined = {}
    for seed, per_size in manifests.items():
        seed_dir = config.subsets_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for size, manifest in per_size.items():
            with open(manifest_path(config, seed, size), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=1)
        combined[str(seed)] = {
            str(size): {k: v for k, v in m.items() if k not in ("example_ids", "labels")} for size, m in per_size.items()
        }
        with open(seed_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(combined[str(seed)], fh, indent=2)
    with open(config.subsets_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2)
