#!/usr/bin/env python
"""Run the MDL n-gram-order and/or smoothing ablations over the ladder
(8 sizes x 5 seeds per configuration, same subset ids as every other model).

    n-gram study:    n in {2, 3, 4},       k = 1
    smoothing study: k in {0.1, 0.5, 1, 2}, n = 3

The trigram k=1 configuration is the existing `mdl` baseline: it is not
retrained, its results are verified and reused (see
`emotion_ait.ablation.verify_reference_results`). Pass
`--rerun-reference` to also retrain it under the separate name
`mdl_n3_k1`, e.g. to check reproducibility.

Each configuration writes `results/metrics/<model>_ladder_results[_<tag>].csv`
plus the usual per-run `<model>_ladder_n<size>_<seed>.*` files; the run
refuses to start if any target combined CSV already exists.

Usage
-----
    python scripts/run_mdl_ablation.py                              # both studies, full ladder
    python scripts/run_mdl_ablation.py --study ngram
    python scripts/run_mdl_ablation.py --study smoothing
    python scripts/run_mdl_ablation.py --configs mdl_n4_k1 --sizes 100 --seeds 42 --results-tag debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_ait.ablation import (  # noqa: E402
    REFERENCE_MODEL_NAME,
    STUDIES,
    config_summary,
    results_path,
    study_configs,
    verify_reference_results,
)
from emotion_ait.config import load_config  # noqa: E402
from emotion_ait.experiments import run_experiment  # noqa: E402
from emotion_ait.mdl_classifier import MDLClassifier  # noqa: E402
from emotion_ait.sampling import load_ladder_subset_dataframe  # noqa: E402
from run_ladder_experiments import load_processed_df  # noqa: E402

logger = logging.getLogger("run_mdl_ablation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=[*STUDIES, "all"], default="all", help="Which ablation to run.")
    parser.add_argument("--configs", nargs="+", default=None, help="Only these model names (e.g. mdl_n2_k1).")
    parser.add_argument("--sizes", type=int, nargs="+", default=None, help="Default: config.sampling.ladder_sizes.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: config.sampling.ladder_seeds.")
    parser.add_argument("--results-tag", default=None, help="Suffix for the combined CSVs, e.g. for debug runs.")
    parser.add_argument("--rerun-reference", action="store_true", help="Also retrain trigram k=1 as mdl_n3_k1.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    studies = list(STUDIES) if args.study == "all" else [args.study]
    runs = {}
    for study in studies:
        for name, n, k in study_configs(study):
            if name == REFERENCE_MODEL_NAME:
                if not args.rerun_reference:
                    continue
                name = "mdl_n3_k1"
            runs[name] = (n, k)
    if args.configs:
        unknown = set(args.configs) - set(runs)
        if unknown:
            parser.error(f"Unknown/unselected configs {sorted(unknown)}; available: {sorted(runs)}.")
        runs = {name: runs[name] for name in args.configs}

    verify_reference_results(config)
    logger.info("Reference '%s' results verified against the ladder manifests; reused, not retrained.", REFERENCE_MODEL_NAME)

    existing = [str(results_path(config, name, args.results_tag)) for name in runs if results_path(config, name, args.results_tag).exists()]
    if existing:
        parser.error(f"Refusing to overwrite/append existing results: {existing}. Move them away or use --results-tag.")

    processed_df = load_processed_df(config)
    sizes = args.sizes or config.sampling.ladder_sizes
    seeds = args.seeds or config.sampling.ladder_seeds
    total, index = len(runs) * len(sizes) * len(seeds), 0
    for name, (n, k) in runs.items():
        for seed in seeds:
            for size in sizes:
                index += 1
                train_df = load_ladder_subset_dataframe(seed, size, processed_df, config)
                record = run_experiment(
                    model=MDLClassifier(n=n, smoothing="laplace", k=k),
                    model_name=name,
                    subset_name=f"ladder_n{size}",
                    processed_df=processed_df,
                    config=config,
                    config_summary=config_summary(n, k),
                    train_df=train_df,
                    seed=seed,
                    results_filename=results_path(config, name, args.results_tag).name,
                )
                logger.info(
                    "[%d/%d] %s seed=%d size=%d: val_macro_f1=%.4f test_macro_f1=%.4f",
                    index, total, name, seed, size, record.validation_macro_f1, record.test_macro_f1,
                )
    logger.info("Done: %d runs. Next: python scripts/report_mdl_ablation.py", total)


if __name__ == "__main__":
    main()
