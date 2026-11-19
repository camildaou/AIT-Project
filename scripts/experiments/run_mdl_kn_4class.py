#!/usr/bin/env python
"""Run interpolated Kneser-Ney MDL on the 4-class ladder
(8 sizes x 5 seeds per configuration, same subset ids as every other model).

Configurations (see ``emotion_ait.mdl_kneser_ney.KN_CONFIGS``):

    mdl_n2_knclass   bigram,  per-class lower orders
    mdl_n2_knshared  bigram,  lower orders pooled over the training subset
    mdl_n3_knclass   trigram, per-class lower orders
    mdl_n4_knclass   4-gram,  per-class lower orders

Each writes ``results/metrics/<model>_ladder_results[_<tag>].csv``; the run
refuses to start if any target combined CSV already exists.

Usage
-----
    python scripts/experiments/run_mdl_kn_4class.py
    python scripts/experiments/run_mdl_kn_4class.py --configs mdl_n2_knclass --sizes 100 --seeds 42 --results-tag debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_ait.data_loader import load_processed_dataframe  # noqa: E402
from emotion_ait.ablation import results_path  # noqa: E402
from emotion_ait.config import load_config  # noqa: E402
from emotion_ait.experiments import run_experiment  # noqa: E402
from emotion_ait.mdl_kneser_ney import KN_CONFIGS, MDLKneserNey4Classifier, kn_config_summary  # noqa: E402
from emotion_ait.sampling import load_ladder_subset_dataframe  # noqa: E402

logger = logging.getLogger("run_mdl_kn_4class")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=None, help="Only these model names.")
    parser.add_argument("--sizes", type=int, nargs="+", default=None, help="Default: config.sampling.ladder_sizes.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: config.sampling.ladder_seeds.")
    parser.add_argument("--results-tag", default=None, help="Suffix for the combined CSVs, e.g. for debug runs.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    runs = {name: (n, backoff) for name, n, backoff in KN_CONFIGS}
    if args.configs:
        unknown = set(args.configs) - set(runs)
        if unknown:
            parser.error(f"Unknown configs {sorted(unknown)}; available: {sorted(runs)}.")
        runs = {name: runs[name] for name in args.configs}

    config = load_config(args.config)
    existing = [str(results_path(config, name, args.results_tag)) for name in runs if results_path(config, name, args.results_tag).exists()]
    if existing:
        parser.error(f"Refusing to overwrite/append existing results: {existing}. Move them away or use --results-tag.")

    processed_df = load_processed_dataframe(config)
    sizes = args.sizes or config.sampling.ladder_sizes
    seeds = args.seeds or config.sampling.ladder_seeds
    total, index = len(runs) * len(sizes) * len(seeds), 0
    for name, (n, backoff) in runs.items():
        for seed in seeds:
            for size in sizes:
                index += 1
                train_df = load_ladder_subset_dataframe(seed, size, processed_df, config)
                record = run_experiment(
                    model=MDLKneserNey4Classifier(n=n, kn_backoff=backoff),
                    model_name=name,
                    subset_name=f"ladder_n{size}",
                    processed_df=processed_df,
                    config=config,
                    config_summary=kn_config_summary(n, backoff),
                    train_df=train_df,
                    seed=seed,
                    results_filename=results_path(config, name, args.results_tag).name,
                )
                logger.info(
                    "[%d/%d] %s seed=%d size=%d: val_macro_f1=%.4f test_macro_f1=%.4f",
                    index, total, name, seed, size, record.validation_macro_f1, record.test_macro_f1,
                )
    logger.info("Done: %d runs.", total)


if __name__ == "__main__":
    main()
