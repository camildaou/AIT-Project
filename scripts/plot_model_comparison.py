"""Create separate comparison charts for initial runs and the ladder runs.

Run from the project root with:
    python scripts/plot_model_comparison.py

The script reads existing result CSVs and writes new artifacts under
results/figures and results/metrics. It does not modify the source result files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
MODELS = {
    "MDL": METRICS_DIR / "mdl_ladder_results.csv",
    "LSTM": METRICS_DIR / "lstm_ladder_results.csv",
}
INITIAL_MODELS = {
    "MDL": METRICS_DIR / "mdl_results.csv",
    "LSTM": METRICS_DIR / "lstm_results.csv",
}
METRICS = {
    "Macro-F1": "test_macro_f1",
    "Accuracy": "test_accuracy",
    "Training time (seconds)": "train_time_seconds",
}


def load_results(files: dict[str, Path], experiment: str) -> pd.DataFrame:
    frames = []
    for model, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected {experiment} results file: {path}")
        frame = pd.read_csv(path)
        frame["model"] = model
        frame["test_per_class"] = frame["test_per_class"].map(json.loads)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    required = {"model", "train_subset_size", "seed", *METRICS.values(), "test_per_class"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"{experiment} results are missing columns: {sorted(missing)}")
    return results


def save_learning_curve(
    results: pd.DataFrame, label: str, column: str, filename: str, title_prefix: str
) -> None:
    grouped = (
        results.groupby(["model", "train_subset_size"])[column]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"MDL": "#2878B5", "LSTM": "#E07A24"}
    has_repeats = grouped["count"].max() > 1
    plotted_models = set(grouped.loc[grouped["mean"].notna(), "model"])
    for model in ("MDL", "LSTM"):
        data = grouped[grouped["model"] == model].sort_values("train_subset_size")
        data = data[data["mean"].notna()]
        if data.empty:
            continue
        if has_repeats:
            ax.errorbar(
                data["train_subset_size"], data["mean"], yerr=data["std"].fillna(0),
                marker="o", markersize=5, capsize=4, linewidth=2,
                color=colors[model], label=model,
            )
        else:
            ax.plot(
                data["train_subset_size"], data["mean"], marker="o", markersize=5,
                linewidth=2, color=colors[model], label=model,
            )
    ax.set_title(f"{title_prefix}: {label} by training set size")
    ax.set_xlabel("Number of training examples")
    ax.set_ylabel(label)
    if column in {"test_macro_f1", "test_accuracy"}:
        spread = grouped["std"].fillna(0)
        low = (grouped["mean"] - spread).min()
        high = (grouped["mean"] + spread).max()
        padding = max((high - low) * 0.12, 0.02)
        ax.set_ylim(max(0, low - padding), min(1, high + padding))
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    if plotted_models:
        ax.legend(frameon=False)
    missing_models = {"MDL", "LSTM"} - plotted_models
    if missing_models and column == "train_time_seconds":
        missing_text = ", ".join(sorted(missing_models))
        ax.text(
            0.02, 0.98, f"No recorded training-time values for {missing_text}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_full_data_per_class(results: pd.DataFrame, filename: str, title_prefix: str) -> None:
    full_size = results["train_subset_size"].max()
    full = results[results["train_subset_size"] == full_size]
    labels = ("anger", "fear", "joy", "sadness")
    rows = []
    for model in ("MDL", "LSTM"):
        entries = full[full["model"] == model]
        for label in labels:
            values = [scores[label]["f1"] for scores in entries["test_per_class"]]
            rows.append({"model": model, "emotion": label, "mean_f1": sum(values) / len(values)})
    class_scores = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    pivot = class_scores.pivot(index="emotion", columns="model", values="mean_f1").reindex(labels)
    pivot.plot(kind="bar", ax=ax, color=["#2878B5", "#E07A24"], width=0.72)
    ax.set_title(f"{title_prefix}: per-class test F1 at full training size (n={full_size})")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", frameon=False)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_summary(results: pd.DataFrame, filename: str) -> None:
    rows = []
    for (size, model), group in results.groupby(["train_subset_size", "model"]):
        row = {"train_subset_size": size, "model": model, "runs": len(group)}
        for label, column in METRICS.items():
            row[f"{label.lower().replace('-', '_').replace(' ', '_')}_mean"] = group[column].mean()
            row[f"{label.lower().replace('-', '_').replace(' ', '_')}_std"] = group[column].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["train_subset_size", "model"])
    summary.to_csv(METRICS_DIR / filename, index=False)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    first_training = load_results(INITIAL_MODELS, "first training")
    for label, column in METRICS.items():
        if column == "train_time_seconds":
            continue
        name = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_learning_curve(
            first_training, label, column,
            f"first_training_{name}_learning_curve.png", "First training"
        )
    save_full_data_per_class(
        first_training, "first_training_per_class_f1_full_training.png", "First training"
    )
    save_comparison_summary(first_training, "first_training_model_comparison_summary.csv")

    ladder = load_results(MODELS, "ladder")
    for seed in sorted(ladder["seed"].unique()):
        seed_results = ladder[ladder["seed"] == seed]
        save_learning_curve(
            seed_results,
            "Macro-F1",
            "test_macro_f1",
            f"ladder_seed_{int(seed)}_macro_f1_learning_curve.png",
            f"Ladder, seed {int(seed)}",
        )
        save_learning_curve(
            seed_results,
            "Accuracy",
            "test_accuracy",
            f"ladder_seed_{int(seed)}_accuracy_learning_curve.png",
            f"Ladder, seed {int(seed)}",
        )
        save_full_data_per_class(
            seed_results,
            f"ladder_seed_{int(seed)}_per_class_f1_full_training.png",
            f"Ladder, seed {int(seed)}",
        )
    save_comparison_summary(ladder, "ladder_model_comparison_summary.csv")
    print("Created comparison artifacts:")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")
    for slug in ("first_training", "ladder"):
        print(f"  {(METRICS_DIR / f'{slug}_model_comparison_summary.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
