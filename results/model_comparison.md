# MDL and LSTM ladder comparison

The project has two sets of experiments, so the charts are split into separate groups. Both use the same test set for model evaluation.

## First training runs

- [Macro-F1 curve](figures/first_training_macro-f1_learning_curve.png)
- [Accuracy curve](figures/first_training_accuracy_learning_curve.png)
- [Per-class F1 at full training size](figures/first_training_per_class_f1_full_training.png)

These runs use the original training subsets and one seed (42), so their charts show one line of results without error bars. The values are in [first_training_model_comparison_summary.csv](metrics/first_training_model_comparison_summary.csv).

## Ladder runs

The ladder charts are separated by seed. Each seed has Macro-F1 and accuracy curves, plus a grouped bar chart comparing per-class F1 for anger, fear, joy, and sadness at the full training size:

- **Seed 42:** [Macro-F1](figures/ladder_seed_42_macro_f1_learning_curve.png) · [Accuracy](figures/ladder_seed_42_accuracy_learning_curve.png) · [Per-class F1](figures/ladder_seed_42_per_class_f1_full_training.png)
- **Seed 43:** [Macro-F1](figures/ladder_seed_43_macro_f1_learning_curve.png) · [Accuracy](figures/ladder_seed_43_accuracy_learning_curve.png) · [Per-class F1](figures/ladder_seed_43_per_class_f1_full_training.png)
- **Seed 44:** [Macro-F1](figures/ladder_seed_44_macro_f1_learning_curve.png) · [Accuracy](figures/ladder_seed_44_accuracy_learning_curve.png) · [Per-class F1](figures/ladder_seed_44_per_class_f1_full_training.png)
- **Seed 45:** [Macro-F1](figures/ladder_seed_45_macro_f1_learning_curve.png) · [Accuracy](figures/ladder_seed_45_accuracy_learning_curve.png) · [Per-class F1](figures/ladder_seed_45_per_class_f1_full_training.png)
- **Seed 46:** [Macro-F1](figures/ladder_seed_46_macro_f1_learning_curve.png) · [Accuracy](figures/ladder_seed_46_accuracy_learning_curve.png) · [Per-class F1](figures/ladder_seed_46_per_class_f1_full_training.png)

The overall ladder averages remain in [ladder_model_comparison_summary.csv](metrics/ladder_model_comparison_summary.csv).

Across the five-seed ladder average, MDL has higher mean macro-F1 at 100 and 249 examples, and LSTM higher from 500 onward. At 3,125 examples, LSTM reaches 0.645 versus 0.495 for MDL. Individual seed charts make the run-to-run differences visible.

## Recreate the artifacts

From the `AIT-Project` directory, run:

```powershell
python scripts/plot_model_comparison.py
```

The script reads `results/metrics/mdl_results.csv`, `lstm_results.csv`, and the two `*_ladder_results.csv` files. It writes separate first-training and ladder charts and summary tables, leaving the source results untouched.
