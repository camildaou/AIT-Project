#!/usr/bin/env python
"""Summarise the MDL n-gram-order and smoothing ablations (tables + learning
curves) from the results written by `scripts/run_mdl_ablation.py`.

Model selection uses **validation** macro-F1 only: per study, the selected
configuration is the one with the highest validation macro-F1 averaged over
all sizes and seeds (and, per size, the one with the highest mean validation
macro-F1 at that size). Test metrics are reported for the selected
configurations afterwards and never used to choose between them.

Outputs (refuses to overwrite existing ones unless `--overwrite`):
    results/ablation/mdl_ablation_report.md                  all tables, markdown
    results/ablation/mdl_ablation_<study>_by_size.csv        mean/std per config x size
    results/ablation/mdl_ablation_<study>_summary.csv        one row per config
    results/ablation/mdl_ablation_confusion_matrices.csv     test confusion matrices, long format
    results/figures/mdl_ablation_<study>.png                 validation | test learning curves

Usage
-----
    python scripts/report_mdl_ablation.py
    python scripts/report_mdl_ablation.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from emotion_ait.ablation import (  # noqa: E402
    REFERENCE_MODEL_NAME,
    STUDIES,
    results_path,
    study_configs,
    verify_reference_results,
)
from emotion_ait.config import load_config  # noqa: E402

# Ordinal one-hue ramps (n and k are ordered quantities), validated light->dark.
RAMPS = {3: ["#86b6ef", "#2a78d6", "#0d366b"], 4: ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]}
MARKERS = ["o", "s", "^", "D"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _label(n: int, k: float) -> str:
    return f"n={n}, k={k:g}" + (" (baseline)" if (n, k) == (3, 1.0) else "")


def load_study(config, study: str) -> pd.DataFrame:
    frames = []
    for name, n, k in study_configs(study):
        df = verify_reference_results(config) if name == REFERENCE_MODEL_NAME else pd.read_csv(results_path(config, name))
        expected = len(config.sampling.ladder_sizes) * len(config.sampling.ladder_seeds)
        if len(df) != expected:
            raise ValueError(f"{name}: {len(df)} runs, expected {expected}; is the ablation run complete?")
        frames.append(df.assign(config=_label(n, k), n=n, k=k, size=df["train_subset_name"].str.removeprefix("ladder_n").astype(int)))
    out = pd.concat(frames, ignore_index=True)
    # Same example ids for every configuration => same realised subset sizes.
    sizes = out.groupby(["size", "seed"])["train_subset_size"].nunique()
    if (sizes != 1).any():
        raise ValueError(f"{study}: configurations trained on different subset sizes for {sizes[sizes != 1].index.tolist()}")
    return out


def by_size(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["config", "n", "k", "size"]).agg(
        val_macro_f1_mean=("validation_macro_f1", "mean"), val_macro_f1_std=("validation_macro_f1", "std"),
        test_macro_f1_mean=("test_macro_f1", "mean"), test_macro_f1_std=("test_macro_f1", "std"),
        test_accuracy_mean=("test_accuracy", "mean"), test_accuracy_std=("test_accuracy", "std"),
        train_time_s_mean=("train_time_seconds", "mean"), inference_time_s_mean=("inference_time_seconds", "mean"),
    )
    return agg.reset_index().sort_values(["n", "k", "size"])


def summary(df: pd.DataFrame, per_size: pd.DataFrame) -> pd.DataFrame:
    smallest, largest = per_size["size"].min(), per_size["size"].max()
    rows = []
    for (config, n, k), g in df.groupby(["config", "n", "k"]):
        s = per_size[per_size["config"] == config].set_index("size")
        rows.append({
            "config": config, "n": n, "k": k,
            "val_macro_f1_overall": g["validation_macro_f1"].mean(),
            "test_macro_f1_overall": g["test_macro_f1"].mean(),
            "test_accuracy_overall": g["test_accuracy"].mean(),
            f"test_macro_f1_n{smallest}": s.loc[smallest, "test_macro_f1_mean"],
            f"test_macro_f1_n{smallest}_std": s.loc[smallest, "test_macro_f1_std"],
            f"test_macro_f1_n{largest}": s.loc[largest, "test_macro_f1_mean"],
            f"test_macro_f1_n{largest}_std": s.loc[largest, "test_macro_f1_std"],
            "mean_seed_std_test_macro_f1": s["test_macro_f1_std"].mean(),
        })
    out = pd.DataFrame(rows).sort_values(["n", "k"]).reset_index(drop=True)
    out["selected_by_validation"] = out["val_macro_f1_overall"] == out["val_macro_f1_overall"].max()
    return out


def per_size_selection(per_size: pd.DataFrame) -> pd.DataFrame:
    best = per_size.loc[per_size.groupby("size")["val_macro_f1_mean"].idxmax()]
    return best[["size", "config", "val_macro_f1_mean", "test_macro_f1_mean", "test_macro_f1_std"]].reset_index(drop=True)


def per_class_at(df: pd.DataFrame, size: int) -> pd.DataFrame:
    rows = []
    for config, g in df[df["size"] == size].groupby("config", sort=False):
        per_class = [json.loads(p) for p in g["test_per_class"]]
        for label in per_class[0]:
            rows.append({"config": config, "class": label, **{
                m: sum(p[label][m] for p in per_class) / len(per_class) for m in ("precision", "recall", "f1")
            }})
    return pd.DataFrame(rows)


def confusion_matrices(config, df: pd.DataFrame, study: str) -> pd.DataFrame:
    """Test confusion matrices rebuilt from each run's saved predictions."""
    labels = [config.target_task.id_to_label[i] for i in sorted(config.target_task.id_to_label)]
    rows = []
    for (model_name, config_label, size, seed), _ in df.groupby(["model_name", "config", "size", "seed"]):
        pred = pd.read_csv(config.paths.results_dir / "predictions" / f"{model_name}_ladder_n{size}_{seed}.csv")
        counts = pred.groupby(["true_label_name", "predicted_label_name"]).size()
        for true in labels:
            for predicted in labels:
                rows.append({"study": study, "model_name": model_name, "config": config_label, "size": size, "seed": seed,
                             "true": true, "predicted": predicted, "count": int(counts.get((true, predicted), 0))})
    return pd.DataFrame(rows)


def plot_study(per_size: pd.DataFrame, study: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = per_size.drop_duplicates("config").sort_values(["n", "k"])["config"].tolist()
    colors = dict(zip(configs, RAMPS[len(configs)]))
    sizes = sorted(per_size["size"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, facecolor=SURFACE)
    for ax, split, title in ((axes[0], "val", "Validation (used for selection)"), (axes[1], "test", "Test (final reporting)")):
        ax.set_facecolor(SURFACE)
        ends = []
        for i, config in enumerate(configs):
            s = per_size[per_size["config"] == config].sort_values("size")
            mean, std = s[f"{split}_macro_f1_mean"], s[f"{split}_macro_f1_std"]
            ax.fill_between(s["size"], mean - std, mean + std, color=colors[config], alpha=0.12, linewidth=0)
            ax.plot(s["size"], mean, color=colors[config], linewidth=2, marker=MARKERS[i], markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=config, zorder=3)
            ends.append([mean.iloc[-1], config])
        # Direct labels at the right end, dodged so they never overlap.
        ends.sort()
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.012)
        for y, config in ends:
            ax.annotate(config, (sizes[-1], y), xytext=(10, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK_2, annotation_clip=False)
        ax.set_xscale("log")
        # Sizes >= 1500 crowd together on a log axis; label every other one there.
        ticks = [s for s in sizes if s < 1500 or s in (2000, sizes[-1])]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(s) for s in ticks], fontsize=9, color=MUTED)
        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=9, colors=MUTED)
        ax.tick_params(colors=AXIS)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Training-set size (log scale)", fontsize=9, color=INK_2)
    axes[0].set_ylabel("Macro-F1 (mean ± 1 sd over 5 seeds)", fontsize=9, color=INK_2)
    name = {"ngram": "n-gram order (k = 1)", "smoothing": "add-k smoothing (n = 3)"}[study]
    fig.suptitle(f"MDL ablation: {name}", x=0.01, ha="left", fontsize=13, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 0.94, 0.93))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _markdown(df: pd.DataFrame, digits: int = 3) -> str:
    def fmt(v):
        if isinstance(v, bool):
            return "**yes**" if v else ""
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing report outputs.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file.")
    args = parser.parse_args()
    config = load_config(args.config)

    out_dir = config.paths.results_dir / "ablation"
    fig_dir = config.paths.results_dir / "figures"
    outputs = [out_dir / "mdl_ablation_report.md", out_dir / "mdl_ablation_confusion_matrices.csv"]
    for study in STUDIES:
        outputs += [out_dir / f"mdl_ablation_{study}_by_size.csv", out_dir / f"mdl_ablation_{study}_summary.csv",
                    fig_dir / f"mdl_ablation_{study}.png"]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not args.overwrite:
        parser.error(f"Report outputs already exist (pass --overwrite to replace): {existing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    largest = max(config.sampling.ladder_sizes)
    report: List[str] = [
        "# MDL ablation: n-gram order and add-k smoothing", "",
        "Generated by `scripts/report_mdl_ablation.py`. Every configuration uses the same ladder subset ids, "
        "tokenization, shared vocabulary and decision rule as `mdl`; the trigram k=1 row *is* the reused `mdl` "
        "baseline. Selection uses validation macro-F1 only; test numbers are reported after selection. "
        "Micro-F1 equals accuracy for single-label multiclass predictions, so it is reported as accuracy.", "",
    ]
    confusions: List[pd.DataFrame] = []
    for study in STUDIES:
        df = load_study(config, study)
        per_size = by_size(df)
        summ = summary(df, per_size)
        per_size.to_csv(out_dir / f"mdl_ablation_{study}_by_size.csv", index=False)
        summ.to_csv(out_dir / f"mdl_ablation_{study}_summary.csv", index=False)
        plot_study(per_size, study, fig_dir / f"mdl_ablation_{study}.png")
        confusions.append(confusion_matrices(config, df, study))

        selected = summ[summ["selected_by_validation"]].iloc[0]
        title = {"ngram": "Experiment A: n-gram order (k = 1)", "smoothing": "Experiment B: smoothing k (n = 3)"}[study]
        table = per_size.assign(
            val=per_size["val_macro_f1_mean"].map("{:.3f}".format) + " ± " + per_size["val_macro_f1_std"].map("{:.3f}".format),
            test=per_size["test_macro_f1_mean"].map("{:.3f}".format) + " ± " + per_size["test_macro_f1_std"].map("{:.3f}".format),
        ).pivot(index="size", columns="config", values=["val", "test"])
        report += [
            f"## {title}", "", f"![{title}](../figures/mdl_ablation_{study}.png)", "",
            f"**Selected by validation:** {selected['config']} (overall validation macro-F1 "
            f"{selected['val_macro_f1_overall']:.3f}; its test macro-F1 {selected['test_macro_f1_overall']:.3f}).", "",
            "### Summary per configuration", "", _markdown(summ), "",
            "### Validation macro-F1 by size (mean ± sd over 5 seeds)", "",
            _markdown(table["val"].reset_index()), "",
            "### Test macro-F1 by size (mean ± sd over 5 seeds)", "",
            _markdown(table["test"].reset_index()), "",
            "### Best configuration per size, chosen on validation", "", _markdown(per_size_selection(per_size)), "",
            f"### Per-class test metrics at n={largest} (mean over seeds)", "",
            _markdown(per_class_at(df, largest)), "",
            "### Mean train / inference time (s)", "",
            _markdown(per_size.groupby("config", sort=False)[["train_time_s_mean", "inference_time_s_mean"]].mean().reset_index()), "",
        ]
    confusion = pd.concat(confusions, ignore_index=True).drop_duplicates(["model_name", "size", "seed", "true", "predicted"])
    confusion.to_csv(out_dir / "mdl_ablation_confusion_matrices.csv", index=False)
    report += [f"## Test confusion matrices at n={largest} (summed over seeds; rows = true, columns = predicted)", ""]
    for config_label, g in confusion[confusion["size"] == largest].groupby("config", sort=False):
        m = g.pivot_table(index="true", columns="predicted", values="count", aggfunc="sum")
        report += [f"**{config_label}**", "", _markdown(m.reset_index()), ""]
    (out_dir / "mdl_ablation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out_dir / 'mdl_ablation_report.md'} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
