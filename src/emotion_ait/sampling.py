"""Deterministic, leakage-free, stratified training-subset sampling.

Design summary
--------------
For every target emotion we build **one** deterministic permutation of
that class's training example ids, seeded from the shared random seed.
A subset of size *N* for that class is always the first *N* ids of that
fixed permutation. This is the key trick that makes nesting automatic
and exact: because the same permutation is reused for every requested
subset size, a smaller per-class quota is always a prefix of a larger
per-class quota for that class, for the *same* seed. Consequently, the
whole-dataset subset for a smaller size is a subset (in the mathematical
sense) of the subset for any larger size, as long as every class's quota
is non-decreasing between the two sizes.

Rounding rule
-------------
Per-class quotas are computed independently per requested size using
**round-half-up** (``floor(x + 0.5)``), i.e. ``quota_c = round_half_up(
fraction_or_pct_of_full_train * count_c)``, then clipped to
``count_c`` (a class cannot contribute more examples than it has).
Because round-half-up and ``min(..., count_c)`` are both non-decreasing
functions of their input, and the requested fraction increases
monotonically across our configured percentage/size sequences, per-class
quotas never shrink as the requested size grows. This guarantees nesting
across the standard percentage sequence (5/10/20/40/70/100%) and across
the absolute-size sequence (100/250/500), at the cost of the realized
subset total only approximately (not exactly) matching the requested
percentage/size -- the realized size is reported alongside the request.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

TRAIN_SPLIT_NAME = "train"


def _round_half_up(x: float) -> int:
    """Round-half-up to the nearest integer (never Python's banker's rounding)."""
    return math.floor(x + 0.5)


@dataclasses.dataclass(frozen=True)
class SubsetRequest:
    """One requested training-subset size.

    Exactly one of ``percentage`` / ``absolute_size`` should be set.
    """

    name: str
    percentage: Optional[float] = None
    absolute_size: Optional[int] = None

    @property
    def kind(self) -> str:
        return "percentage" if self.percentage is not None else "absolute"


def default_subset_requests(config: Config) -> List[SubsetRequest]:
    """Build the standard list of subset requests from the config:
    one per configured percentage, plus one per configured absolute size.
    """
    requests = [
        SubsetRequest(name=f"train_pct{int(p) if float(p).is_integer() else p}", percentage=float(p))
        for p in config.sampling.percentages
    ]
    requests += [
        SubsetRequest(name=f"train_abs{n}", absolute_size=int(n)) for n in config.sampling.absolute_sizes
    ]
    return requests


def _class_permutations(train_df: pd.DataFrame, label_names: List[str], seed: int) -> Dict[str, List[str]]:
    """Build one fixed, deterministic permutation of example ids per class.

    Ids are sorted before permuting so that the result depends only on
    the (id, seed) pair and not on row order in ``train_df``.
    """
    permutations: Dict[str, List[str]] = {}
    for label_name in label_names:
        class_ids = sorted(train_df.loc[train_df["label_name"] == label_name, "example_id"].tolist())
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(class_ids))
        permutations[label_name] = [class_ids[i] for i in order]
    return permutations


def _class_quotas(
    class_counts: Dict[str, int],
    *,
    percentage: Optional[float] = None,
    absolute_size: Optional[int] = None,
    total_train: int,
) -> Dict[str, int]:
    """Compute the per-class quota for one subset request.

    See the module docstring for the rounding / clipping rule and why it
    guarantees nesting.
    """
    quotas: Dict[str, int] = {}
    for label_name, count in class_counts.items():
        if percentage is not None:
            target = (percentage / 100.0) * count
        else:
            # Absolute size: allocate proportionally to each class's
            # share of the *full filtered training partition*.
            class_fraction = count / total_train if total_train > 0 else 0.0
            target = class_fraction * float(absolute_size)
        quota = min(_round_half_up(target), count)
        quotas[label_name] = max(quota, 0)
    return quotas


def build_subset(
    train_df: pd.DataFrame,
    request: SubsetRequest,
    label_names: List[str],
    seed: int,
    class_permutations: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, object]:
    """Select one deterministic, stratified training subset.

    Parameters
    ----------
    train_df:
        The *filtered* training partition only (never validation/test).
    request:
        Which size/percentage to build.
    label_names:
        The target emotion names, in a fixed documented order.
    seed:
        Global random seed (default 42 throughout the project).
    class_permutations:
        Optional precomputed output of :func:`_class_permutations`, so
        that callers generating many subsets only pay the permutation
        cost once per seed.

    Returns
    -------
    dict
        JSON-serialisable metadata, including the sorted list of selected
        ``example_ids`` and any stratification warnings encountered.
    """
    if not (train_df["original_split"] == TRAIN_SPLIT_NAME).all():
        raise ValueError("build_subset must only be called with rows from the 'train' split.")

    class_counts = {name: int((train_df["label_name"] == name).sum()) for name in label_names}
    total_train = int(sum(class_counts.values()))

    if class_permutations is None:
        class_permutations = _class_permutations(train_df, label_names, seed)

    quotas = _class_quotas(
        class_counts,
        percentage=request.percentage,
        absolute_size=request.absolute_size,
        total_train=total_train,
    )

    warning_messages: List[str] = []
    selected_ids: List[str] = []
    for label_name in label_names:
        quota = quotas[label_name]
        available = class_permutations[label_name]
        if quota == 0 and class_counts[label_name] > 0:
            msg = (
                f"Subset '{request.name}': quota for class '{label_name}' rounded down to 0 "
                f"(class has {class_counts[label_name]} training examples); perfect "
                "stratification is not possible for this subset size."
            )
            warning_messages.append(msg)
            warnings.warn(msg)
        selected_ids.extend(available[:quota])

    selected_ids = sorted(selected_ids)  # deterministic order before saving

    requested_size: Union[int, float]
    if request.percentage is not None:
        requested_size = (request.percentage / 100.0) * total_train
    else:
        requested_size = request.absolute_size

    return {
        "name": request.name,
        "kind": request.kind,
        "percentage": request.percentage,
        "absolute_size": request.absolute_size,
        "seed": seed,
        "train_total_count": total_train,
        "requested_size": requested_size,
        "actual_size": len(selected_ids),
        "class_counts": {name: quotas[name] for name in label_names},
        "warnings": warning_messages,
        "example_ids": selected_ids,
    }


def generate_all_subsets(
    processed_df: pd.DataFrame,
    config: Config,
    requests: Optional[List[SubsetRequest]] = None,
) -> Dict[str, Dict[str, object]]:
    """Generate every configured training subset and write each one to
    ``data/subsets/<name>.json``, plus a combined ``manifest.json``.
    """
    train_df = processed_df[processed_df["original_split"] == TRAIN_SPLIT_NAME].reset_index(drop=True)
    label_names = config.target_task.target_emotions
    seed = config.sampling.seed

    if requests is None:
        requests = default_subset_requests(config)

    class_permutations = _class_permutations(train_df, label_names, seed)

    config.paths.subsets_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, object]] = {}
    for request in requests:
        subset = build_subset(train_df, request, label_names, seed, class_permutations)
        results[request.name] = subset
        out_path = config.paths.subsets_dir / f"{request.name}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(subset, fh, indent=2)
        logger.info(
            "Subset '%s': requested=%s actual=%d classes=%s",
            request.name,
            subset["requested_size"],
            subset["actual_size"],
            subset["class_counts"],
        )

    manifest = {
        name: {k: v for k, v in subset.items() if k != "example_ids"} for name, subset in results.items()
    }
    manifest_path = config.paths.subsets_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return results


def load_subset_ids(name: str, config: Config) -> List[str]:
    """Load the sorted list of example ids previously saved for a subset."""
    path = config.paths.subsets_dir / f"{name}.json"
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return list(payload["example_ids"])


def load_subset_dataframe(name: str, processed_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Load a subset as a dataframe of full rows (joined against the
    processed training partition)."""
    ids = set(load_subset_ids(name, config))
    train_df = processed_df[processed_df["original_split"] == TRAIN_SPLIT_NAME]
    return train_df[train_df["example_id"].isin(ids)].reset_index(drop=True)


# --- Multi-seed training-size "ladder" (MDL-vs-LSTM comparison) ------------
#
# A second, independent subset-generation path, kept separate from the
# single-seed percentage/absolute-size subsets above (which are unaffected
# by any of this). For each of several seeds, generates a nested ladder of
# *absolute* training sizes using the exact same deterministic,
# fixed-permutation mechanism as `build_subset` above (so the same nesting
# guarantee applies, within a seed, across increasing sizes). Both MDL and
# LSTM must load these exclusively through `load_ladder_subset_dataframe`
# (never regenerate ids themselves), which is what guarantees they always
# train on the identical set of example ids for a given (seed, size).
#
# On-disk layout, rooted at `config.paths.subsets_dir / "ladder"`:
#   seed_<seed>/n<size>.json   -- one subset per (seed, size), with
#                                 `example_ids` and a parallel `labels` list
#                                 (same order), `class_counts`, etc.
#   seed_<seed>/manifest.json  -- per-seed summary across all sizes
#                                 (no ids/labels, for quick inspection).
#   manifest.json              -- combined summary across all seeds/sizes.

DEFAULT_LADDER_SIZES: List[int] = [100, 250, 500, 1000, 1500, 2000, 2500, 3125]
DEFAULT_LADDER_SEEDS: List[int] = [42, 43, 44, 45, 46]


def _ladder_dir(subsets_dir: Path) -> Path:
    return subsets_dir / "ladder"


def _ladder_seed_dir(subsets_dir: Path, seed: int) -> Path:
    return _ladder_dir(subsets_dir) / f"seed_{seed}"


def generate_ladder_subsets(
    processed_df: pd.DataFrame,
    config: Config,
    sizes: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
) -> Dict[int, Dict[int, Dict[str, object]]]:
    """Generate the full (seed x size) ladder of nested, stratified
    training subsets and write them to disk (see the layout described
    above), then validate every invariant required by the comparison
    (:func:`validate_ladder_subsets`) before returning.

    Parameters
    ----------
    processed_df:
        The full processed dataset (all splits), as produced by
        :mod:`emotion_ait.preprocessing`. Only its training-split rows
        are ever sampled from; validation/test rows are read only to
        validate that no leakage occurred.
    config:
        Shared project configuration. Defaults for ``sizes``/``seeds``
        come from ``config.sampling.ladder_sizes`` /
        ``config.sampling.ladder_seeds`` when not given explicitly.
    sizes:
        Absolute training sizes to generate, e.g.
        ``[100, 250, 500, 1000, 1500, 2000, 2500, 3125]``. Must not
        exceed the size of the filtered training partition.
    seeds:
        Random seeds to generate an independent nested ladder for.

    Returns
    -------
    dict
        ``{seed: {size: subset_dict}}``, where each ``subset_dict`` is
        the same JSON-serialisable schema written to disk.

    Raises
    ------
    ValueError
        If a requested size exceeds the available training partition,
        or if any post-generation invariant is violated (see
        :func:`validate_ladder_subsets`) -- nothing is silently dropped
        or altered.
    """
    sizes = sorted({int(s) for s in (sizes if sizes is not None else config.sampling.ladder_sizes)})
    seeds = list(seeds if seeds is not None else config.sampling.ladder_seeds)

    train_df = processed_df[processed_df["original_split"] == TRAIN_SPLIT_NAME].reset_index(drop=True)
    label_names = config.target_task.target_emotions
    total_train = len(train_df)
    id_to_label_name = dict(zip(train_df["example_id"], train_df["label_name"]))

    for size in sizes:
        if size > total_train:
            raise ValueError(
                f"Requested ladder size {size} exceeds the filtered training partition "
                f"({total_train} examples); the maximum achievable training subset under "
                "the current 4-class single-label filter is "
                f"{total_train}."
            )

    ladder_dir = _ladder_dir(config.paths.subsets_dir)
    ladder_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[int, Dict[int, Dict[str, object]]] = {}
    for seed in seeds:
        class_permutations = _class_permutations(train_df, label_names, seed)
        seed_dir = _ladder_seed_dir(config.paths.subsets_dir, seed)
        seed_dir.mkdir(parents=True, exist_ok=True)

        per_size: Dict[int, Dict[str, object]] = {}
        for size in sizes:
            request = SubsetRequest(name=f"n{size}", absolute_size=size)
            subset = build_subset(train_df, request, label_names, seed, class_permutations)
            subset["labels"] = [id_to_label_name[example_id] for example_id in subset["example_ids"]]
            per_size[size] = subset
            out_path = seed_dir / f"{request.name}.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(subset, fh, indent=2)
            logger.info(
                "Ladder seed=%d size=%d: requested=%s actual=%d classes=%s",
                seed,
                size,
                subset["requested_size"],
                subset["actual_size"],
                subset["class_counts"],
            )
        results[seed] = per_size

        seed_manifest = {
            str(size): {k: v for k, v in subset.items() if k not in ("example_ids", "labels")}
            for size, subset in per_size.items()
        }
        with open(seed_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(seed_manifest, fh, indent=2)

    combined_manifest = {
        str(seed): {
            str(size): {k: v for k, v in subset.items() if k not in ("example_ids", "labels")}
            for size, subset in per_size.items()
        }
        for seed, per_size in results.items()
    }
    with open(ladder_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(combined_manifest, fh, indent=2)

    validate_ladder_subsets(results, processed_df, config, label_names)

    return results


def validate_ladder_subsets(
    ladder_results: Dict[int, Dict[int, Dict[str, object]]],
    processed_df: pd.DataFrame,
    config: Config,
    label_names: Optional[List[str]] = None,
) -> None:
    """Validate every invariant required by the MDL-vs-LSTM ladder
    comparison, raising ``ValueError`` immediately (never silently
    dropping or altering data) if any of them is violated:

    * every selected id actually belongs to the training split, and none
      belong to validation/test (no leakage);
    * each subset's reported ``actual_size`` matches its id count, and is
      within a small rounding tolerance of the requested size (exactly
      equal when the requested size is the full training partition);
    * all four target classes are represented (non-zero count) at every
      (seed, size);
    * within each seed's ladder, every smaller subset's ids are a subset
      of every next-larger subset's ids (nesting).
    """
    label_names = label_names or config.target_task.target_emotions

    train_ids: Set[str] = set(
        processed_df.loc[processed_df["original_split"] == TRAIN_SPLIT_NAME, "example_id"]
    )
    non_train_ids: Set[str] = set(
        processed_df.loc[processed_df["original_split"] != TRAIN_SPLIT_NAME, "example_id"]
    )
    total_train = len(train_ids)

    for seed, per_size in ladder_results.items():
        previous_ids: Optional[Set[str]] = None
        for size in sorted(per_size.keys()):
            subset = per_size[size]
            ids: Set[str] = set(subset["example_ids"])

            leaked = ids & non_train_ids
            if leaked:
                raise ValueError(
                    f"Ladder subset (seed={seed}, size={size}) leaks {len(leaked)} "
                    f"validation/test id(s) into training, e.g. {sorted(leaked)[:5]}."
                )
            if not ids <= train_ids:
                raise ValueError(
                    f"Ladder subset (seed={seed}, size={size}) contains id(s) that are not "
                    "part of the training split at all."
                )

            actual_size = subset["actual_size"]
            if actual_size != len(ids):
                raise ValueError(
                    f"Ladder subset (seed={seed}, size={size}): 'actual_size' ({actual_size}) "
                    f"does not match the number of selected example ids ({len(ids)})."
                )
            tolerance = max(len(label_names), 1)
            if abs(actual_size - size) > tolerance:
                raise ValueError(
                    f"Ladder subset (seed={seed}, size={size}): requested size is not "
                    f"achievable (actual size {actual_size} differs from the requested "
                    f"{size} by more than the {tolerance}-example rounding tolerance)."
                )
            if size == total_train and actual_size != size:
                raise ValueError(
                    f"Ladder subset (seed={seed}, size={size}): requesting the full training "
                    f"partition ({total_train} examples) must reproduce it exactly, but got "
                    f"{actual_size}."
                )

            class_counts = subset["class_counts"]
            missing = [name for name in label_names if class_counts.get(name, 0) <= 0]
            if missing:
                raise ValueError(
                    f"Ladder subset (seed={seed}, size={size}) is missing class(es) {missing}; "
                    "all four target classes must be represented at every size."
                )

            if previous_ids is not None and not previous_ids <= ids:
                raise ValueError(
                    f"Ladder subset (seed={seed}, size={size}) is not a superset of the "
                    "previous (smaller) size's subset; nesting is violated."
                )
            previous_ids = ids


def load_ladder_subset_ids(seed: int, size: int, config: Config) -> List[str]:
    """Load the sorted list of example ids for one (seed, size) ladder
    subset, previously written by :func:`generate_ladder_subsets`."""
    path = _ladder_seed_dir(config.paths.subsets_dir, seed) / f"n{size}.json"
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return list(payload["example_ids"])


def load_ladder_subset_dataframe(seed: int, size: int, processed_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Load one (seed, size) ladder subset as a dataframe of full rows,
    joined against the processed training partition.

    Both MDL and LSTM must call this (never regenerate ids themselves)
    so they always train on the identical set of examples for a given
    (seed, size)."""
    ids = set(load_ladder_subset_ids(seed, size, config))
    train_df = processed_df[processed_df["original_split"] == TRAIN_SPLIT_NAME]
    return train_df[train_df["example_id"].isin(ids)].reset_index(drop=True)
