#!/usr/bin/env python
"""Paired significance tests over the 5 seeds (reads results CSVs, writes nothing).

For each pair of models and each training size, the 5 seeds give 5 paired test
macro-F1 differences (same training ids for both models). We report the mean
difference, how many seeds favour model A, and a two-sided paired t-test.
Pooled over all sizes (40 pairs) we also give a Wilcoxon signed-rank test.

Usage: python scripts/analysis/paired_seed_tests.py
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
M = ROOT / "results" / "metrics"


def load(name):
    d = pd.read_csv(M / f"{name}_ladder_results.csv")
    d["size"] = d.train_subset_name.str.extract(r"n(\d+)").astype(int)
    return d.set_index(["size", "seed"]).test_macro_f1


PAIRS = {
    "4-class": [
        ("KN bigram", "mdl_n2_knclass", "LSTM", "lstm"),
        ("add-1 bigram", "mdl_n2_k1", "LSTM", "lstm"),
        ("KN bigram", "mdl_n2_knclass", "add-1 bigram", "mdl_n2_k1"),
        ("KN trigram", "mdl_n3_knclass", "KN bigram", "mdl_n2_knclass"),
        ("KN 4-gram", "mdl_n4_knclass", "KN bigram", "mdl_n2_knclass"),
    ],
    "27-class": [
        ("KN bigram", "sl27_mdl_n2_knclass", "LSTM", "sl27_lstm"),
        ("shared neural", "sl27_shared_neural_mdl", "LSTM", "sl27_lstm"),
        ("shared neural", "sl27_shared_neural_mdl", "KN bigram", "sl27_mdl_n2_knclass"),
        ("KN bigram", "sl27_mdl_n2_knclass", "add-1 bigram", "sl27_mdl_n2_k1"),
        ("27 separate nets", "sl27_neural_bigram_mdl", "add-1 bigram", "sl27_mdl_n2_k1"),
        ("KN trigram", "sl27_mdl_n3_knclass", "KN bigram", "sl27_mdl_n2_knclass"),
        ("KN 4-gram", "sl27_mdl_n4_knclass", "KN bigram", "sl27_mdl_n2_knclass"),
    ],
}

for track, pairs in PAIRS.items():
    print(f"\n===== {track}: test macro-F1, A - B, paired by seed =====")
    for a_label, a, b_label, b in pairs:
        diff = (load(a) - load(b)).dropna()
        print(f"\n{a_label}  vs  {b_label}")
        for size, d in diff.groupby(level="size"):
            wins = int((d > 0).sum())
            ties = int((d == 0).sum())
            p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else float("nan")
            flag = "*" if p < 0.05 else " "
            print(f"  {size:>6}: mean {d.mean():+.3f}  A better in {wins}/{len(d)}"
                  f"{f' ({ties} ties)' if ties else ''}  t-test p={p:.3g}{flag}")
        w = stats.wilcoxon(diff)
        print(f"  pooled {len(diff)} pairs: mean {diff.mean():+.4f}, A better in {(diff > 0).sum()}/{len(diff)}, "
              f"Wilcoxon p={w.pvalue:.3g}")
