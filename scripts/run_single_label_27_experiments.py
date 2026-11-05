#!/usr/bin/env python
"""Run one model configuration over the 27-class single-label ladder
(8 sizes x 5 seeds), with the shared evaluation of every other track.

Built-in models:
    mdl   -- MDL27Classifier(n, k)              (--n, --k, --prior; 27 class models,
             shared vocabulary; --prior adds -log2 P(e) and is named mdl_prior_*)
    lstm  -- LSTMClassifier(num_classes=27, ...) (--epochs, --patience); requires
             the LSTM to accept a `num_classes` argument (it is 4-class only today).
Or plug in any EmotionClassifier without editing this script:
    --factory package.module:build --name my_model
where `build(seed: int) -> (model, config_summary_dict)` returns a fresh model.

Results: results/metrics/sl27_<name>_ladder_results[_<tag>].csv and per-run
sl27_<name>_ladder_n<size>_<seed>.* files; refuses to overwrite existing results.

Usage
-----
    python scripts/run_single_label_27_experiments.py --model mdl --n 2 --k 1
    python scripts/run_single_label_27_experiments.py --model mdl --n 2 --k 1 --prior
    python scripts/run_single_label_27_experiments.py --model mdl --smoothing kneser_ney --kn-backoff shared   # -> sl27_mdl_n2_knshared
    python scripts/run_single_label_27_experiments.py --model mdl --sizes 23485 --seeds 42 --diagnostics
    python scripts/run_single_label_27_experiments.py --model lstm
    python scripts/run_single_label_27_experiments.py --factory my_pkg.models:build --name lstm_v2
    python scripts/run_single_label_27_experiments.py --model mdl --sizes 500 --seeds 42 --results-tag debug
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_ait.single_label_27.config import load_single_label_27_config  # noqa: E402
from emotion_ait.single_label_27.data import CLASS_NAMES  # noqa: E402
from emotion_ait.single_label_27.experiments import results_path, run_name, run_single_label_27_ladder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_single_label_27_experiments")


def _format_k(k: float) -> str:
    return f"{k:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", choices=["mdl", "lstm"], help="Built-in model.")
    source.add_argument("--factory", help="Custom builder as 'package.module:function'.")
    parser.add_argument("--name", help="Results name (required with --factory; default derived from the model).")
    parser.add_argument("--n", type=int, default=2, help="MDL n-gram order (default 2, the best 4-class setting).")
    parser.add_argument("--k", type=float, default=1.0, help="MDL add-k smoothing (default 1).")
    parser.add_argument("--prior", action="store_true", help="MDL: add the training class prior -log2 P(e).")
    parser.add_argument("--smoothing", choices=["laplace", "kneser_ney"], default="laplace", help="MDL smoothing.")
    parser.add_argument("--kn-backoff", choices=["class", "shared"], default="shared",
                        help="MDL Kneser-Ney lower-order distribution: per class or pooled over the training subset.")
    parser.add_argument("--kn-discount", type=float, default=None, help="MDL Kneser-Ney fixed discount (default: estimated).")
    parser.add_argument("--diagnostics", action="store_true", help="Write a per-run diagnostics JSON (if the model supports it).")
    parser.add_argument("--epochs", type=int, default=10, help="LSTM max epochs.")
    parser.add_argument("--patience", type=int, default=3, help="LSTM early-stopping patience.")
    parser.add_argument("--sizes", type=int, nargs="+", default=None, help="Subset of ladder sizes (debugging).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Subset of ladder seeds (debugging).")
    parser.add_argument("--results-tag", default=None, help="Suffix for the combined results CSV, e.g. 'debug'.")
    parser.add_argument("--config", type=Path, default=None, help="Path to the 27-class YAML config.")
    args = parser.parse_args()
    config = load_single_label_27_config(args.config)

    if args.factory:
        if not args.name:
            parser.error("--name is required with --factory.")
        module_name, _, attr = args.factory.partition(":")
        build = getattr(importlib.import_module(module_name), attr)
        name = args.name
    elif args.model == "mdl":
        from emotion_ait.single_label_27.mdl import MDL27Classifier

        kneser_ney = args.smoothing == "kneser_ney"

        def build(seed):
            model = MDL27Classifier(n=args.n, smoothing=args.smoothing, k=args.k, use_prior=args.prior,
                                    kn_backoff=args.kn_backoff, kn_discount=args.kn_discount)
            summary = {"model": "MDL27Classifier", "n": model.n, "smoothing": model.smoothing, "use_prior": model.use_prior}
            if kneser_ney:
                summary.update(kn_backoff=model.kn_backoff, kn_discount=model.kn_discount or "estimated")
            else:
                summary["k"] = model.k
            return model, summary

        variant = "mdl_prior" if args.prior else "mdl"
        if kneser_ney:
            discount = f"_d{_format_k(args.kn_discount)}" if args.kn_discount else ""
            default_name = f"{variant}_n{args.n}_kn{args.kn_backoff}{discount}"
        else:
            default_name = f"{variant}_n{args.n}_k{_format_k(args.k)}"
        name = args.name or default_name
    else:
        from emotion_ait.lstm_classifier import LSTMClassifier

        if "num_classes" not in inspect.signature(LSTMClassifier.__init__).parameters:
            parser.error("LSTMClassifier has no `num_classes` argument yet (it only supports the 4-class task). "
                         "Add one, or pass a 27-class model via --factory.")

        def build(seed):
            model = LSTMClassifier(num_classes=len(CLASS_NAMES), max_epochs=args.epochs, patience=args.patience, seed=seed)
            return model, {"num_classes": len(CLASS_NAMES), "max_epochs": args.epochs, "patience": args.patience, "seed": seed}

        name = args.name or "lstm"

    if results_path(config, name, args.results_tag).exists():
        parser.error(f"{results_path(config, name, args.results_tag)} already exists; move it away or use --results-tag.")
    records = run_single_label_27_ladder(build, name, config, args.sizes, args.seeds, args.results_tag, args.diagnostics)
    logger.info("Done: %d runs of %s. Results: %s", len(records), run_name(name), results_path(config, name, args.results_tag))


if __name__ == "__main__":
    main()
