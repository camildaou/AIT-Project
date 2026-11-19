#!/usr/bin/env python
"""Dataset-level two-part MDL model selection on the 4-class track.

Configurations (four emotion-specific models each, same training comments,
same tokenization and shared training-only vocabulary):

    bigram_add1   MDLClassifier(n=2, k=1)
    trigram_add1  MDLClassifier(n=3, k=1)
    bigram_kn     MDLKneserNey4Classifier(n=2, kn_backoff="class")

For each one: L(M) = sum_e L(M_e) (fixed-width count code, see
``emotion_ait.two_part_mdl``), L(D|M) = sum_i L(x_i | M_{y_i}) on the *same*
training comments (in-sample), and L_total = L(M) + L(D|M). The configuration
with the smallest L_total is selected **before** any test metric is looked at;
test macro-F1 / accuracy / confusion matrix are then reported for all three
(classification rule unchanged: argmin_e L(x | M_e)).

Main run: the full 4-class training set (3,125 comments). Robustness: the same
selection on every ladder subset (8 sizes x 5 seeds).

Writes ``results/two_part_mdl/{main,ladder}.csv`` and ``report.md``. Nothing
else in ``results/`` is touched.

Usage
-----
    python scripts/experiments/run_two_part_mdl_4class.py
    python scripts/experiments/run_two_part_mdl_4class.py --bits 16 --no-ladder
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_ait.data_loader import load_processed_dataframe  # noqa: E402
from emotion_ait.config import load_config  # noqa: E402
from emotion_ait.evaluation import compute_metrics  # noqa: E402
from emotion_ait.mdl_classifier import MDLClassifier, _tokenize  # noqa: E402
from emotion_ait.mdl_kneser_ney import MDLKneserNey4Classifier  # noqa: E402
from emotion_ait.sampling import load_ladder_subset_dataframe  # noqa: E402
from emotion_ait.two_part_mdl import two_part_summary  # noqa: E402

CONFIGS = {
    "bigram_add1": lambda: MDLClassifier(n=2, smoothing="laplace", k=1.0),
    "trigram_add1": lambda: MDLClassifier(n=3, smoothing="laplace", k=1.0),
    "bigram_kn": lambda: MDLKneserNey4Classifier(n=2, kn_backoff="class"),
}


def evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, id_to_label, bits: int):
    """Fit every configuration on train_df; return selection rows and test metrics."""
    rows, metrics = [], {}
    for name, build in CONFIGS.items():
        model = build().fit(train_df)
        row = {"config": name, **two_part_summary(model, train_df.text.tolist(), train_df.label.tolist(), bits)}
        metrics[name] = compute_metrics(test_df.label.tolist(), model.predict(test_df.text.tolist()), id_to_label)
        rows.append(row)
    table = pd.DataFrame(rows)
    # Selection uses the training-set objective only; test columns are added afterwards.
    selected = table.loc[table.L_total_bits.idxmin(), "config"]
    table["selected"] = table.config == selected
    table["rank_total"] = table.L_total_bits.rank().astype(int)
    table["rank_total_with_keys"] = table.L_total_with_keys_bits.rank().astype(int)
    table["test_macro_f1"] = [metrics[c]["macro_f1"] for c in table.config]
    table["test_accuracy"] = [metrics[c]["accuracy"] for c in table.config]
    return table, selected, metrics


def uniform_code_bits(train_df: pd.DataFrame) -> float:
    """Reference with no model at all (L(M) = 0): every token incl. </s> costs log2 V,
    V = the same training-only prediction vocabulary (words + </s> + <unk>)."""
    tokens = [_tokenize(t) for t in train_df.text]
    vocab_size = len({w for ts in tokens for w in ts}) + 2
    return sum(len(ts) + 1 for ts in tokens) * math.log2(vocab_size)


def fmt_bits(x: float) -> str:
    return f"{x:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, default=32, help="Bits per stored count entry (B).")
    parser.add_argument("--no-ladder", action="store_true", help="Skip the ladder robustness check.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "two_part_mdl")
    args = parser.parse_args()

    config = load_config(None)
    processed = load_processed_dataframe(config)
    train_df = processed[processed.original_split == "train"].reset_index(drop=True)
    test_df = processed[processed.original_split == "test"].reset_index(drop=True)
    id_to_label = config.target_task.id_to_label
    args.out.mkdir(parents=True, exist_ok=True)

    table, selected, metrics = evaluate(train_df, test_df, id_to_label, args.bits)
    table.to_csv(args.out / "main.csv", index=False)
    (args.out / "main_test_metrics.json").write_text(json.dumps(metrics, indent=2))

    cols = ["config", "entries_sufficient", "L_model_bits", "L_data_bits", "L_total_bits", "model_share",
            "data_bits_per_comment", "L_total_with_keys_bits", "L_total_stored_bits", "test_macro_f1", "test_accuracy"]
    print(f"\nFull training set: {len(train_df)} comments, B = {args.bits} bits/entry")
    print(table[cols].to_string(index=False))
    uniform = uniform_code_bits(train_df)
    print(f"\nReference, no model (uniform code over the vocabulary): {uniform:,.0f} bits")
    print(f"\nSelected by minimum L_total: {selected}")
    m = metrics[selected]
    print(f"Test macro-F1 {m['macro_f1']:.4f}, accuracy {m['accuracy']:.4f}")
    print("Confusion matrix (rows = true, cols = predicted):", m["confusion_matrix_labels"])
    for label, row in zip(m["confusion_matrix_labels"], m["confusion_matrix"]):
        print(f"  {label:>8}: {row}")

    ladder = None
    if not args.no_ladder:
        parts = []
        for seed in config.sampling.ladder_seeds:
            for size in config.sampling.ladder_sizes:
                sub = load_ladder_subset_dataframe(seed, size, processed, config)
                t, _, _ = evaluate(sub, test_df, id_to_label, args.bits)
                t["seed"], t["size"] = seed, size
                parts.append(t)
        ladder = pd.concat(parts, ignore_index=True)
        ladder.to_csv(args.out / "ladder.csv", index=False)
        sel = ladder[ladder.selected].groupby("size").config.value_counts().unstack(fill_value=0)
        best_f1 = ladder.loc[ladder.groupby(["seed", "size"]).test_macro_f1.idxmax()]
        agree = (ladder[ladder.selected].set_index(["seed", "size"]).config
                 == best_f1.set_index(["seed", "size"]).config).mean()
        print("\nLadder: configuration selected by L_total (count over 5 seeds)")
        print(sel.to_string())
        print(f"Selected config is also the best on test macro-F1 in {agree:.0%} of {len(best_f1)} runs")

    write_report(args.out / "report.md", table, selected, metrics, ladder, len(train_df), args.bits, uniform)


def write_report(path, table, selected, metrics, ladder, n_train, bits, uniform):
    lines = [
        "# Two-part MDL model selection (4-class, in-sample)",
        "",
        f"Training data D: {n_train} comments (full 4-class training set). B = {bits} bits per stored count entry.",
        "L(D|M) is measured on the same comments the models were fitted on (in-sample),",
        "so it is a dataset-description length, not a held-out estimate.",
        "",
        "| config | count entries | L(M) bits | L(D|M) bits | L_total bits | L(M) share | bits/comment | test macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table.itertuples():
        mark = " **(selected)**" if r.selected else ""
        lines.append(
            f"| {r.config}{mark} | {r.entries_sufficient:,} | {fmt_bits(r.L_model_bits)} | {fmt_bits(r.L_data_bits)} | "
            f"{fmt_bits(r.L_total_bits)} | {r.model_share:.1%} | {r.data_bits_per_comment:.1f} | {r.test_macro_f1:.4f} |"
        )
    lines += [f"| uniform code (reference, not a candidate) | 0 | 0 | {fmt_bits(uniform)} | {fmt_bits(uniform)} | 0% | "
              f"{uniform / n_train:.1f} | – |"]
    m = metrics[selected]
    lines += ["", f"Selected: **{selected}** — test macro-F1 {m['macro_f1']:.4f}, accuracy {m['accuracy']:.4f}.", "",
              "Confusion matrix (rows true, columns predicted): " + ", ".join(m["confusion_matrix_labels"]), ""]
    lines += ["    " + f"{lab:>8} " + " ".join(f"{v:5d}" for v in row)
              for lab, row in zip(m["confusion_matrix_labels"], m["confusion_matrix"])]
    if ladder is not None:
        sel = ladder[ladder.selected].groupby("size").config.value_counts().unstack(fill_value=0)
        lines += ["", "## Ladder robustness (selections over 5 seeds per size)", "", sel.to_markdown()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
