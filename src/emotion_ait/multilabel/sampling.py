"""Nested, label-combination-stratified training subsets for the multi-label track.

Strata
------
Each observed non-empty target combination in the processed training
partition (e.g. ``joy``, ``sadness+fear``) is one stratum. Strata are put
in a fixed *canonical order*: fewer labels first, then by the target
order of their labels (``joy``, ``sadness``, ``anger``, ``fear``,
``joy+sadness``, ``joy+anger``, ...).

Ordering and allocation
-----------------------
For one seed:

1. Within each stratum, ids are sorted and then permuted with
   ``numpy.random.RandomState(seed)`` (same convention as the single-label
   sampler), giving a fixed seeded order per stratum.
2. A single global order of all ``N`` training ids is built by sequential
   **Webster / Sainte-Lague** allocation: slot ``t`` goes to the stratum
   ``s`` (with ids left) maximising ``N_s / (a_s + 0.5)``, where ``N_s`` is
   the stratum size and ``a_s`` the number of slots it already has; that
   stratum's next id fills the slot. Ties go to the stratum earliest in
   canonical order. Priorities are compared as exact fractions.
3. The subset of size ``n`` is the first ``n`` ids of the global order.

Consequences: every subset has exactly ``n`` examples; smaller subsets are
prefixes (hence subsets) of larger ones; the full size contains every
training example once; and each stratum's count at size ``n`` is the
Webster apportionment of ``n`` -- the proportional share ``n * N_s / N``
rounded at 0.5, adjusted only as needed to sum to ``n``. Allocation is
sequential, so a stratum's count never decreases as ``n`` grows (no
largest-remainder "Alabama paradox"). Rare combinations therefore only
enter once their proportional share reaches about one half (e.g. a
4-example combination out of 4,883 first appears at ``n`` ~ 600).

The interleaving of strata depends only on the stratum sizes; the seed
decides *which* examples of each stratum come first.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import MultilabelConfig
from .preprocessing import COMBINATION_SEPARATOR

STRATIFICATION = "label_combination_webster_sequential"


def canonical_stratum_order(combinations: Sequence[str], target_labels: Sequence[str]) -> List[str]:
    position = {label: i for i, label in enumerate(target_labels)}

    def key(combination: str):
        labels = combination.split(COMBINATION_SEPARATOR)
        return (len(labels), [position[label] for label in labels])

    return sorted(set(combinations), key=key)


def stratum_permutations(train_df: pd.DataFrame, seed: int, target_labels: Sequence[str]) -> Dict[str, List[str]]:
    """Seeded order of example ids within each label-combination stratum."""
    permutations = {}
    for combination in canonical_stratum_order(train_df["label_combination"], target_labels):
        ids = sorted(train_df.loc[train_df["label_combination"] == combination, "example_id"].tolist())
        order = np.random.RandomState(seed).permutation(len(ids))
        permutations[combination] = [ids[i] for i in order]
    return permutations


def webster_sequence(stratum_sizes: Dict[str, int], strata: Sequence[str]) -> List[str]:
    """Stratum chosen for each of the ``sum(stratum_sizes)`` slots (see module docstring)."""
    allocated = {s: 0 for s in strata}
    sequence = []
    for _ in range(sum(stratum_sizes[s] for s in strata)):
        best, best_priority = None, None
        for s in strata:
            if allocated[s] >= stratum_sizes[s]:
                continue
            priority = Fraction(2 * stratum_sizes[s], 2 * allocated[s] + 1)
            if best is None or priority > best_priority:
                best, best_priority = s, priority
        sequence.append(best)
        allocated[best] += 1
    return sequence


def nested_order(train_df: pd.DataFrame, seed: int, target_labels: Sequence[str]) -> List[str]:
    """The global seeded order of all training ids; subset ``n`` = first ``n``."""
    permutations = stratum_permutations(train_df, seed, target_labels)
    strata = list(permutations)
    next_index = {s: 0 for s in strata}
    order = []
    for s in webster_sequence({s: len(ids) for s, ids in permutations.items()}, strata):
        order.append(permutations[s][next_index[s]])
        next_index[s] += 1
    return order


def _sha256_lines(lines: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def train_partition_fingerprint(train_df: pd.DataFrame, target_labels: Sequence[str]) -> str:
    """Hash of the processed training ids and targets, in file order."""
    return _sha256_lines(
        [f"{i}\t{''.join(str(int(v)) for v in row)}" for i, row in zip(train_df["example_id"], train_df[list(target_labels)].values)]
    )


def build_ladder_manifests(train_df: pd.DataFrame, config: MultilabelConfig) -> Dict[int, Dict[int, Dict[str, object]]]:
    """``{seed: {size: manifest}}`` for every configured seed and size."""
    labels = config.target_labels
    if len(train_df) not in config.ladder_sizes or max(config.ladder_sizes) != len(train_df):
        raise ValueError(f"The largest ladder size must be the full training partition ({len(train_df)}), got {config.ladder_sizes}.")
    targets_by_id = {i: [int(v) for v in row] for i, row in zip(train_df["example_id"], train_df[labels].values)}
    combination_by_id = dict(zip(train_df["example_id"], train_df["label_combination"]))
    fingerprint = train_partition_fingerprint(train_df, labels)
    strata = canonical_stratum_order(train_df["label_combination"], labels)

    manifests: Dict[int, Dict[int, Dict[str, object]]] = {}
    for seed in config.ladder_seeds:
        order = nested_order(train_df, seed, labels)
        manifests[seed] = {}
        for size in sorted(config.ladder_sizes):
            ids = order[:size]
            targets = [targets_by_id[i] for i in ids]
            combinations = [combination_by_id[i] for i in ids]
            manifests[seed][size] = {
                "dataset_id": config.dataset_id,
                "policy": config.policy,
                "target_labels": list(labels),
                "stratification": STRATIFICATION,
                "seed": seed,
                "requested_size": size,
                "actual_size": len(ids),
                "train_partition_size": len(train_df),
                "train_partition_sha256": fingerprint,
                "example_ids_sha256": _sha256_lines(ids),
                "positive_counts": {label: int(sum(t[j] for t in targets)) for j, label in enumerate(labels)},
                "combination_counts": {s: combinations.count(s) for s in strata},
                "example_ids": ids,
                "targets": targets,
            }
    return manifests


def validate_ladder_manifests(manifests: Dict[int, Dict[int, Dict[str, object]]], train_df: pd.DataFrame, config: MultilabelConfig) -> None:
    """Raise ``ValueError`` listing every violated ladder invariant."""
    labels = config.target_labels
    train_ids = set(train_df["example_id"])
    targets_by_id = {i: [int(v) for v in row] for i, row in zip(train_df["example_id"], train_df[labels].values)}
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
            if any(count == 0 for count in m["positive_counts"].values()):
                problems.append(f"seed {seed} n{size}: an emotion has no positives {m['positive_counts']}")
            if m["targets"] != [targets_by_id.get(i) for i in ids]:
                problems.append(f"seed {seed} n{size}: targets do not match the processed training data")
            previous = ids
        full = per_size[max(per_size)]["example_ids"]
        if set(full) != train_ids or len(full) != len(train_ids):
            problems.append(f"seed {seed}: full-size subset does not contain every training example exactly once")
    if problems:
        raise ValueError("Invalid multi-label ladder: " + "; ".join(problems))


def manifest_path(config: MultilabelConfig, seed: int, size: int) -> Path:
    return config.subsets_dir / f"seed_{seed}" / f"n{size}.json"


def write_ladder_manifests(manifests: Dict[int, Dict[int, Dict[str, object]]], config: MultilabelConfig) -> None:
    """Write per-(seed, size) manifests plus per-seed and combined summaries
    (the summaries omit ids/targets)."""
    summary_keys = lambda m: {k: v for k, v in m.items() if k not in ("example_ids", "targets")}  # noqa: E731
    combined = {}
    for seed, per_size in manifests.items():
        seed_dir = config.subsets_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for size, manifest in per_size.items():
            with open(manifest_path(config, seed, size), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=1)
        combined[str(seed)] = {str(size): summary_keys(m) for size, m in per_size.items()}
        with open(seed_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(combined[str(seed)], fh, indent=2)
    with open(config.subsets_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2)
