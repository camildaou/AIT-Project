#!/usr/bin/env python
"""Run the LSTM on named training subsets (or the full configured schedule).

Usage:
    python scripts/run_lstm_experiments.py --subset train_abs100
    python scripts/run_lstm_experiments.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_ait.config import load_config  # noqa: E402
from emotion_ait.experiments import run_experiment  # noqa: E402
from emotion_ait.lstm_classifier import LSTMClassifier  # noqa: E402


# Order requested for the project's learning-curve comparison.
DEFAULT_SUBSETS = [
    "train_abs100", "train_pct5", "train_pct10", "train_abs250",
    "train_pct20", "train_abs500", "train_pct40", "train_pct70",
    "train_pct100",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", action="append", dest="subsets",
        help="Subset name to run; repeat the flag for multiple subsets. Defaults to all nine.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Maximum training epochs (default: 10).")
    parser.add_argument("--patience", type=int, default=3, help="Validation early-stopping patience (default: 3).")
    parser.add_argument(
        "--predictions-only", action="store_true",
        help="Regenerate prediction CSVs without appending duplicate metric rows.",
    )
    args = parser.parse_args()

    config = load_config()
    subsets = args.subsets or DEFAULT_SUBSETS
    for name in subsets:
        subset_path = config.paths.subsets_dir / f"{name}.json"
        if not subset_path.exists():
            parser.error(f"Subset not found: {subset_path}. Run scripts/prepare_data.py first.")

    processed = pd.concat(
        [pd.read_csv(config.paths.processed_dir / f"{split}.csv") for split in config.dataset.splits],
        ignore_index=True,
    )
    for subset_name in subsets:
        print(f"\nRunning LSTM: {subset_name}", flush=True)
        model = LSTMClassifier(
            max_epochs=args.epochs,
            patience=args.patience,
            seed=config.sampling.seed,
        )
        result = run_experiment(
            model=model,
            model_name="lstm",
            subset_name=subset_name,
            processed_df=processed,
            config=config,
            save_predictions=True,
            record_results=not args.predictions_only,
            config_summary={
                "embedding_dim": model.embedding_dim,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "max_epochs": model.max_epochs,
                "learning_rate": model.learning_rate,
                "batch_size": model.batch_size,
                "patience": model.patience,
                "seed": model.seed,
            },
        )
        print(
            f"{subset_name}: n={result.train_subset_size}, "
            f"test_accuracy={result.test_accuracy:.4f}, "
            f"test_macro_f1={result.test_macro_f1:.4f}, "
            f"validation_macro_f1={result.validation_macro_f1:.4f}, "
            f"train_seconds={result.train_time_seconds:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
