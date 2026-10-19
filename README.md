# Emotion Classification Through Algorithmic Information Theory

This repository is the **shared infrastructure** for a research project that
compares two emotion classifiers on a 4-class subset of GoEmotions:

* **MDL classifier** (Person A) -- a Minimum Description Length classifier
  built from emotion-specific smoothed n-gram language models.
* **LSTM classifier** (Person B) -- a conventional supervised LSTM sequence
  classifier (PyTorch).

**Research question.** Can a classical, non-neural, compression-based (MDL)
classifier match a conventional supervised LSTM on emotion classification,
especially in **low-data regimes** (small training subsets)? This repo does
not answer that question yet -- it only provides the shared data pipeline,
model interface, and evaluation protocol that both collaborators' answers
will be built on and compared with.

The shared infrastructure and the **MDL classifier** are implemented; the
**LSTM classifier** is still left as a documented stub (see
[`src/emotion_ait/mdl_classifier.py`](src/emotion_ait/mdl_classifier.py) and
[`src/emotion_ait/lstm_classifier.py`](src/emotion_ait/lstm_classifier.py)).

## 1. Dataset

* Source: [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions)
  on the Hugging Face Hub.
* **Config name: `simplified`** (verified against the Hub on 2026-10-19 via
  the `datasets-server` API and by actually downloading the dataset with
  `datasets==4.5.0`). The project brief referred to a "`go_emotions`
  configuration" -- the repository does not literally expose a config with
  that name; the config that provides the official single-label-friendly
  train/validation/test splits used here is `simplified`. The sibling `raw`
  config exposes only a single `train` split of per-rater annotation rows
  and is **not** used.
* Official splits (as downloaded): `train` = 43,410 rows, `validation` =
  5,426 rows, `test` = 5,427 rows. These match the dataset card exactly, and
  we use them as-is -- **no re-splitting of the full dataset is performed.**
* Columns actually present: `text` (str), `labels` (list of integer indices
  into a fixed 28-name label list: the 27 emotions + `neutral`, see
  `configs/default.yaml`), `id` (str) -- a stable GoEmotions/Reddit comment
  id. We use this `id` column as the `example_id` source whenever present
  (see "Example IDs" below).
* License/usage: GoEmotions is released by Google Research; see the
  [dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions)
  for the exact license and the original paper
  ("GoEmotions: A Dataset of Fine-Grained Emotions", Demszky et al., 2020)
  for citation.

### Loading procedure

All dataset access goes through [`emotion_ait.data_loader`](src/emotion_ait/data_loader.py),
which is the **only** module that calls the Hugging Face `datasets` library.
`download_raw_dataset()` downloads the three official splits once and caches
them as parquet files under `data/raw/` (git-ignored); subsequent runs reuse
the cache via `load_raw_dataset()`.

## 2. Filtering to a 4-class single-label subset

Implemented in [`emotion_ait.preprocessing`](src/emotion_ait/preprocessing.py).
A comment is retained if and only if:

1. Its complete original `labels` list has length exactly 1 (comments with
   zero or multiple labels -- including a target emotion co-occurring with
   any other label, e.g. `neutral` -- are excluded).
2. That single label is one of the four target emotions: `joy`, `sadness`,
   `anger`, `fear`.

Retained labels are mapped to a **fixed, documented** integer id (see
`configs/default.yaml`, `target_task.label_mapping`):

| emotion | id |
|---|---|
| anger | 0 |
| fear | 1 |
| joy | 2 |
| sadness | 3 |

The original comment text and the original split assignment are preserved
unchanged; **no new train/validation/test split is created** -- a comment
that was in the official GoEmotions `validation` split stays in
`validation` after filtering.

### Example IDs

Each processed example gets an `example_id`:

* If the raw GoEmotions `id` (stable Reddit comment id) is present and
  non-empty, it is used directly.
* Otherwise, a deterministic id `"{original_split}-{row_index}"` is
  generated from the original split name and the row's index within that
  split. The comment text is **never** used as an identifier, since it is
  not guaranteed unique.

### Actual retained counts (real run, seed-independent; filtering has no
randomness)

This table was produced by actually running `python scripts/prepare_data.py`
against the real dataset -- these are **observed**, not invented, counts.

| split | anger | fear | joy | sadness | total | retained / original |
|---|---|---|---|---|---|---|
| train | 1025 | 430 | 853 | 817 | **3125** | 3125 / 43410 (7.2%) |
| validation | 109 | 58 | 106 | 84 | **357** | 357 / 5426 (6.6%) |
| test | 131 | 65 | 93 | 102 | **391** | 391 / 5427 (7.2%) |

(Also saved machine-readably at `data/processed/dataset_stats.json` after
running the pipeline.) The class imbalance (`fear` is consistently the
smallest class) is a real property of GoEmotions' single-label subset and
should be kept in mind when interpreting macro-F1 scores, especially on the
smallest training subsets.

## 3. Shared, leakage-free data pipeline

Single source of truth, consumed identically by both collaborators:

* **Dataset loader**: [`emotion_ait.data_loader`](src/emotion_ait/data_loader.py)
* **Preprocessing / filtering**: [`emotion_ait.preprocessing`](src/emotion_ait/preprocessing.py)
* **Fixed splits**: the official GoEmotions train/validation/test partitions
  are reused as-is; validation and test are never resampled or touched by
  the subset sampler.
* **Stratified training-subset sampler**: [`emotion_ait.sampling`](src/emotion_ait/sampling.py)
* **Shared evaluation interface**: [`emotion_ait.evaluation`](src/emotion_ait/evaluation.py)

## 4. Training subsets

Generated by `generate_all_subsets()` in
[`emotion_ait.sampling`](src/emotion_ait/sampling.py) and written to
`data/subsets/<name>.json` (one subset artifact per name, plus a
`manifest.json` with metadata for all subsets but no example ids).

Requested sizes (configurable in `configs/default.yaml`):

* Percentages of the **filtered training partition** (3,125 examples):
  5%, 10%, 20%, 40%, 70%, 100%.
* Absolute sizes: 100, 250, 500 (all comfortably below the 3,125 available
  filtered training examples).

### How sampling works

* One random seed (default **42**) drives everything.
* For each target class, we build **one** deterministic permutation of
  that class's training `example_id`s (`numpy.random.RandomState(seed)`
  over a lexicographically sorted id list, so the permutation depends only
  on the (ids, seed) pair, never on row order).
* A subset of a given requested size selects, for each class, a **quota**
  computed independently from that class's full training count:
  `quota = min(round_half_up(fraction * class_count), class_count)`, where
  `fraction` is `percentage / 100` for percentage-based subsets or
  `requested_size * (class_count / total_train_count)` for absolute-size
  subsets. The first `quota` ids of that class's fixed permutation are
  selected.
* **Rounding**: round-half-up (`floor(x + 0.5)`), not Python's default
  banker's rounding, applied per class. This means the realized subset size
  only approximately (not exactly) matches the requested percentage/size --
  the realized size is always recorded alongside the request (e.g. the
  "pct5" subset requested 156.25 and realized 157 examples; see
  `data/subsets/manifest.json` after running the pipeline).
* **Nesting**: because round-half-up and the `min(..., class_count)` clip
  are both non-decreasing functions of the requested fraction, and every
  class reuses the *same* fixed permutation across all requested sizes,
  each class's quota is non-decreasing as the requested size grows. The
  smaller subset's selected ids for that class are therefore always a
  prefix of the larger subset's -- which makes the whole subset nested too.
  This was verified for the real data: the 5/10/20/40/70/100% subsets and
  the 100/250/500 absolute subsets are both exactly nested chains (see
  `scripts/verify_data.py` output and `tests/test_sampling.py`).
* IDs are **sorted** before being written to the subset JSON file, so saved
  artifacts are byte-for-byte reproducible.
* Subsets are only ever drawn from the training partition; validation and
  test rows are never visible to the sampler (enforced with a runtime
  check in `build_subset`).
* When a class's quota rounds down to 0 for a tiny subset (e.g. a very rare
  class in a 5% subset), a `UserWarning` is raised explaining that perfect
  stratification was not possible for that subset/class combination, and
  that class is simply left out of that subset rather than failing.

Both the MDL and LSTM classifiers must load a subset exclusively through
`emotion_ait.sampling.load_subset_dataframe(name, processed_df, config)` --
this guarantees they always train on **exactly** the same examples at a
given subset size.

## 5. Model interface

Defined once in [`emotion_ait.interfaces.EmotionClassifier`](src/emotion_ait/interfaces.py)
(an `abc.ABC`):

```python
fit(train_data, validation_data=None) -> EmotionClassifier
predict(texts: Sequence[str]) -> list[int]
save(path: Path) -> None
load(path: Path) -> EmotionClassifier   # classmethod
```

`mdl_classifier.MDLClassifier` and `lstm_classifier.LSTMClassifier` both
implement this interface and nothing more is shared between them -- their
internal representations (n-gram language models vs. a PyTorch `nn.Module`)
are completely private to each module.

* **`MDLClassifier` (implemented, Person A)**: fits one smoothed n-gram
  (default: trigram, `n=3`, Laplace/add-`k` smoothing) word-level language
  model per target emotion on that class's training texts only, then
  predicts the class whose model assigns the shortest description length
  (summed negative log2-probability) to the input text -- see the module
  docstring in [`mdl_classifier.py`](src/emotion_ait/mdl_classifier.py) for
  the full smoothing/tokenization contract.
* **`LSTMClassifier` (stub, Person B)**: every method raises
  `NotImplementedError` with a docstring explaining what still needs to be
  built.

`emotion_ait.baselines` provides two trivial, fully-working reference
implementations of the same interface (`MajorityClassBaseline`,
`RandomBaseline`) purely so the rest of the pipeline (sampling, evaluation,
result logging) can be exercised end-to-end today, without pretending any
real training has happened for the two research models.

## 6. Evaluation

[`emotion_ait.evaluation`](src/emotion_ait/evaluation.py) provides:

* `compute_metrics(y_true, y_pred, label_names)` -- accuracy, macro-F1,
  per-class precision/recall/F1/support, and a confusion matrix, using
  scikit-learn with `zero_division=0` so that missing/absent classes (very
  likely in the smallest subsets) produce `0.0` metrics instead of raising
  or emitting scikit-learn's own warnings.
* `warn_if_missing_classes(y_true, label_names)` -- explicit, actionable
  `UserWarning` when an evaluation set is missing a class.
* `Timer` -- a tiny context manager for wall-clock train/inference timing.
* `ResultRecord` / `append_result` -- the shared results schema (model
  name, subset name/percentage/actual size, seed, hyperparameter summary,
  validation macro-F1, test accuracy, test macro-F1, per-class metrics,
  train/inference time, timestamp), appended to `results/metrics/results.csv`
  (one row per run, growing), with each run *also* mirrored individually
  as its own JSON file under `results/metrics/json/` and its own
  single-row metrics CSV under `results/metrics/csv/`
  (`<model>_<subset>_<seed>.{json,csv}`, overwritten on re-run) -- so one
  subset's metrics can be inspected without reading the combined CSV.
* `save_predictions(...)` -- one row per evaluated test example
  (`example_id`, `text`, true/predicted label, a `correct` flag, and --
  when the model exposes a `description_lengths` diagnostic, e.g.
  `MDLClassifier` -- one `bits_<emotion>` column per candidate class),
  written to `results/predictions/<model>_<subset>_<seed>.csv` for
  qualitative/error analysis. Called automatically by `run_experiment`
  (see below) unless `save_test_predictions=False`.

`emotion_ait.experiments.run_experiment(model, model_name, subset_name,
processed_df, config)` wires all of the above together generically and
works today against the baselines; it will work unchanged once the MDL and
LSTM classifiers are implemented.

## 7. Repository structure

```
emotion-ait/ (this repo)
├── README.md                     <- you are here
├── requirements.txt
├── pyproject.toml                <- editable-install config for `emotion_ait`
├── .gitignore
├── configs/
│   └── default.yaml               <- single source of truth for paths, label mapping, seed, subset sizes
├── data/
│   ├── README.md
│   ├── raw/                       <- cached raw GoEmotions parquet (git-ignored)
│   ├── processed/                 <- filtered 4-class CSVs + dataset_stats.json (git-ignored)
│   └── subsets/                   <- per-subset JSON + manifest.json (git-ignored)
├── notebooks/
│   └── dataset_exploration.ipynb  <- exploratory only, no modeling
├── src/emotion_ait/
│   ├── config.py                  <- typed config loading
│   ├── data_loader.py              <- raw GoEmotions download/caching
│   ├── preprocessing.py            <- 4-class single-label filtering
│   ├── sampling.py                 <- deterministic nested stratified subsets
│   ├── interfaces.py               <- shared EmotionClassifier ABC
│   ├── baselines.py                <- fully-working trivial reference models
│   ├── mdl_classifier.py           <- MDL classifier (implemented, Person A)
│   ├── lstm_classifier.py          <- LSTM classifier STUB (Person B)
│   ├── evaluation.py               <- shared metrics + results schema
│   └── experiments.py              <- orchestration glue
├── scripts/
│   ├── prepare_data.py             <- run the whole data pipeline
│   └── verify_data.py              <- integrity checks, human-readable report
├── tests/                          <- synthetic-data unit tests (no network needed)
└── results/
    ├── metrics/                    <- results.csv + per-run JSON/CSV mirrors (small, committable)
    ├── predictions/                <- per-example predictions CSVs (committed)
    └── figures/                    <- git-ignored (regenerable)
```

## 8. Commands

```bash
# 1. Install dependencies (editable install of the emotion_ait package)
pip install -r requirements.txt
pip install -e .

# 2. Prepare the dataset (downloads GoEmotions on first run, then reuses
#    the local cache; also filters, generates subsets, and prints the
#    real retained counts)
python scripts/prepare_data.py

# 3. Verify data integrity (no example in >1 split, correct label mapping,
#    nested subsets, etc.)
python scripts/verify_data.py

# 4. Run the test suite (fast, synthetic data, no network required)
pytest
```

> **Environment note:** this pipeline was developed and verified against a
> real download of GoEmotions using Python 3.9.13 (the only interpreter
> available in the development sandbox used for this repo) with
> `datasets==4.5.0`. The project's declared minimum in `pyproject.toml` is
> Python 3.11+, per the original spec; the code avoids any syntax or
> standard-library features that would not also work on 3.9, so it runs
> unmodified on either version. If your local interpreter is older than
> 3.11, please upgrade before relying on this as a long-term development
> environment -- only the pipeline run documented above has been verified
> here.

## 9. Working independently (Person A / Person B)

* Both of you consume the **same** processed data
  (`data/processed/*.csv`) and the **same** named subsets
  (`data/subsets/*.json`), loaded only via
  `emotion_ait.sampling.load_subset_dataframe`. Never re-filter or re-split
  the data yourselves.
* Implement your classifier entirely inside your own stub module
  (`mdl_classifier.py` for Person A, `lstm_classifier.py` for Person B),
  implementing only the four `EmotionClassifier` methods. You are free to
  add private helper classes/functions within that module (or new, clearly
  owned modules you add alongside it) -- just don't change the shared
  interface, config schema, or preprocessing/sampling logic without
  discussing it, since both of you depend on them being identical.
* Evaluate exclusively through `emotion_ait.evaluation.compute_metrics` /
  `emotion_ait.experiments.run_experiment`, and log results with
  `append_result` so both of your results land in the same
  `results/metrics/results.csv` schema and are directly comparable.
* Do not tune hyperparameters against the `test` split -- use `validation`
  for that, exactly as the shared `run_experiment` helper already does
  (it computes `validation_macro_f1` separately from the test metrics).

## 10. How results will be compared

Once both classifiers are implemented, the plan is to run
`run_experiment` for both models across every subset in
`data/subsets/` (5/10/20/40/70/100% and 100/250/500 absolute), producing
one row per (model, subset) pair in `results/metrics/results.csv`. That
gives a learning-curve-style comparison of MDL vs. LSTM test macro-F1 (and
accuracy, per-class metrics, and train/inference time) as a function of
training-set size -- directly addressing the research question of how the
two approaches compare, especially in the low-data regime.

## Reproducibility & validation

Automated tests ([`tests/`](tests)) verify, against small synthetic
fixtures (no network access required):

* Filtering retains only the four target single-label classes.
* The label &rarr; id mapping matches `configs/default.yaml`.
* Original split assignments are preserved and no example id appears in
  more than one split.
* Training subsets draw exclusively from the training split.
* Percentage subsets (5/10/20/40/70/100%) and absolute-size subsets
  (100/250/500) are each nested in increasing order.
* Repeated runs with the same seed reproduce identical subset selections.
* The evaluation module produces valid, non-crashing metrics for tiny
  evaluation sets and for classes that are absent from the predictions or
  the ground truth.

`scripts/verify_data.py` re-checks the same invariants against the real,
full-size generated dataset and subsets (see the command output in Section
8) -- all checks passed on the real GoEmotions download used for this
repository.
