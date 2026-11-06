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
* **MDL variants** ([`mdl_variants.py`](src/emotion_ait/mdl_variants.py)):
  experimental subclasses of `MDLClassifier` that reuse its tokenization,
  vocabulary, smoothing and scoring unchanged (`mdl_classifier.py` itself
  is untouched). See [section 6.1](#61-mdl-variants) for formulas and
  protocols.
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

### 6.1 MDL variants

All scores are in bits. Ties go to the smallest class id. Every variant
trains on the same ladder subset ids as `mdl` and `lstm` for a given
(seed, size), via `load_ladder_subset_dataframe`.

| Ladder `--model` | Decision rule | Class |
|---|---|---|
| `mdl` | `argmin_e L(x|M_e)` | `MDLClassifier` |
| `mdl_prior` | `argmin_e [L(x|M_e) - log2 P(e)]`, `P(e) = N_e / N` | `MDLPriorClassifier` |
| `mdl_two_part` | `argmin_e [L(M_e) + L(x|M_e)]` | `MDLTwoPartClassifier` |
| `mdl_prequential` | as `mdl`, models built sequentially | `MDLPrequentialClassifier(use_prior=False)` |
| `mdl_prequential_prior` | as `mdl_prior`, models built sequentially | `MDLPrequentialClassifier(use_prior=True)` |

where `L(x|M_e) = -sum_i log2 P(w_i | w_{i-2}, w_{i-1}, M_e)` (add-1 trigram).

**`mdl_prior`.** `N_e / N` is counted from the training subset's labels
only. Validation and test labels are never used. `score_breakdown(texts)`
returns `conditional_bits`, `prior_bits` and `total_bits` per class.

**`mdl_two_part`.** `L(M_e)` is the cost of transmitting class `e`'s
n-gram count table. That cost is paid for the whole dataset, so charging
it on every text is not a proper use of MDL. In practice it dwarfs
`L(x|M_e)` and every text is predicted as the class with the smallest
table (fear). It is kept only as a reference result.

**`mdl_prequential`: protocol.** For one (seed, size) subset of `N`
labelled training texts:

1. *Initial portion:* empty. All `N` observations are coded. An empty
   add-1 model is already a valid code (`log2 V` bits per token), so
   `L_preq` is a complete code for the subset at every size and seed.
2. *Alphabet:* the vocabulary `V` and the class set are those of the
   full training subset, fixed before coding starts, exactly as in `mdl`.
3. *Order:* nested ladder prefix (`sampling.load_ladder_prequential_order`).
   First come the `n100` ids, then the ids that `n250` adds, and so on.
   Each block is shuffled from its sorted ids by one RNG seeded with the
   ladder seed. Within a seed, the order for a smaller size is therefore
   an exact prefix of the order for a larger size. The order's SHA-1 is
   recorded per run.
4. For `t = 1..N`: score `x_t` under every class model *before* updating.
   Record the text code `L(x_t | M_{y_t}^{(t-1)})` and the label code
   `-log2 P_{t-1}(y_t)`, with `P_{t-1}(e) = (N_e^{(t-1)} + 1) / (t-1+K)`.
   Then add `x_t` to the model of its true class `y_t` only.
5. `L_preq_text = sum_t L(x_t | M_{y_t}^{(t-1)})`,
   `L_preq_label = sum_t -log2 P_{t-1}(y_t)`, and
   `L_preq_joint = L_preq_text + L_preq_label`.
6. *Held-out:* validation and test are scored only with the final models
   (plus the final `N_e/N` prior for `mdl_prequential_prior`). They never
   update any model.

Count-based models end up identical whether fitted in one batch or one
example at a time, and the vocabulary is fixed. The final models are
therefore exactly the `mdl` models, so held-out metrics equal those of
`mdl` / `mdl_prior` (checked by a test and on all 40 ladder runs). The
new quantity is `L_preq`, which is reported **separately** from held-out
metrics:

* `results/metrics/<model>_ladder_results.csv`: held-out validation/test
  metrics (same schema as every other model).
* `results/metrics/<model>_ladder_prequential.csv`: one row per run with
  `total_text_bits`, `total_label_bits`, `total_joint_bits`, per-class
  `text_bits_<emotion>` and `updates_<emotion>`, `n_observations`,
  `n_updates`, `online_accuracy` (pre-update argmin on the training
  stream), `order_sha1`, and `cumulative_text_bits_first_n<s>` for every
  smaller ladder size. The last are coded with *this* run's `V`, so they
  are not the `n<s>` run's own total.
* `results/prequential/<model>_ladder_n<size>_<seed>_trace.csv`: one row
  per observation (`position`, `example_id`, pre-update `bits_<emotion>`,
  `text_bits`, `label_bits`, cumulative bits, online prediction).

**Commands.** Each run refuses to start if its combined CSV already
exists, so variants never append into each other's results.

```bash
python scripts/run_ladder_experiments.py --model mdl_prior               # full ladder: 8 sizes x 5 seeds
python scripts/run_ladder_experiments.py --model mdl_prequential
python scripts/run_ladder_experiments.py --model mdl_prequential_prior
python scripts/run_ladder_experiments.py --model mdl_prior --sizes 100 --seeds 42 \
    --results-file mdl_prior_smoke.csv                                    # a single run
```

### 6.2 MDL ablation: n-gram order and smoothing

Two controlled ablations of the conditional MDL classifier. Each one changes
only `n` or `k` of `MDLClassifier(n, k)`. Everything else is identical to `mdl`:
ladder subset ids, tokenization, the shared vocabulary (training words + `</s>` +
`<unk>`, never `<s>`), `n-1` `<s>` pads plus one `</s>`, and the
argmin-description-length rule.

    P(w | context, M_e) = (count_e(context, w) + k) / (count_e(context) + k * V)

| Study | Configurations (`model_name`) | Runs |
|---|---|---|
| A: n-gram order (k = 1) | `mdl_n2_k1`, `mdl` (n=3), `mdl_n4_k1` | 3 × 8 sizes × 5 seeds = 120 |
| B: smoothing (n = 3) | `mdl_n3_k0p1`, `mdl_n3_k0p5`, `mdl` (k=1), `mdl_n3_k2` | 4 × 8 × 5 = 160 |

The trigram `k=1` cell **is** the existing baseline. Its 40 results are reused,
not retrained, after `emotion_ait.ablation.verify_reference_results` checks
four things: the config summary, exactly one run per (size, seed), subset sizes
matching the ladder manifests, and the presence of validation F1 and
predictions files. `--rerun-reference` retrains it under the separate name
`mdl_n3_k1` for a reproducibility check.

**Model selection uses validation macro-F1 only**: the highest mean over all
sizes and seeds, plus a per-size selection table. Test numbers are reported
afterwards. Micro-F1 equals accuracy for single-label predictions. Test
confusion matrices are rebuilt from each run's saved predictions. Validation
predictions aren't saved, so there are no validation confusion matrices.

**Commands**

```bash
python scripts/run_mdl_ablation.py                      # both studies (200 new runs)
python scripts/run_mdl_ablation.py --study ngram        # Experiment A only
python scripts/run_mdl_ablation.py --study smoothing    # Experiment B only
python scripts/run_mdl_ablation.py --configs mdl_n4_k1 --sizes 100 --seeds 42 --results-tag debug   # one debug run
python scripts/report_mdl_ablation.py                   # tables + learning curves
```

The runner refuses to start if a target `results/metrics/<model>_ladder_results[_<tag>].csv`
already exists. The report refuses to overwrite its outputs without `--overwrite`.

**Outputs:** [`results/ablation/mdl_ablation_report.md`](results/ablation/mdl_ablation_report.md)
(all tables), `results/ablation/mdl_ablation_<study>_{by_size,summary}.csv`,
`results/ablation/mdl_ablation_confusion_matrices.csv`, and
`results/figures/mdl_ablation_{ngram,smoothing}.png`.

**Results** (macro-F1, mean over 5 seeds; n=3125 is the full training set,
identical for all seeds, so its sd is 0):

| Configuration | Validation (overall) | Test (overall) | Test n=100 (± sd) | Test n=3125 |
|---|---|---|---|---|
| **n=2, k=1** (selected, study A) | **0.574** | **0.526** | **0.362** ± 0.021 | **0.606** |
| n=3, k=1 (baseline) | 0.433 | 0.430 | 0.299 ± 0.024 | 0.495 |
| n=4, k=1 | 0.371 | 0.398 | 0.284 ± 0.032 | 0.455 |
| **n=3, k=0.1** (selected, study B) | **0.437** | **0.435** | 0.302 ± 0.022 | 0.511 |
| n=3, k=0.5 | 0.436 | 0.435 | 0.301 ± 0.023 | 0.508 |
| n=3, k=2 | 0.423 | 0.423 | 0.298 ± 0.024 | 0.491 |

Bigrams beat trigrams at every one of the 8 sizes on validation, and on test by
about 0.06–0.11 macro-F1. 4-grams are worse still: longer contexts are almost
never seen in ~3k short comments, so the models fall back towards uniform.
Smaller `k` helps slightly and consistently, because there is less probability
mass on unseen n-grams. But all four `k` values sit within about one seed
standard deviation of each other, so smoothing is a minor factor next to the
n-gram order.

**Known baseline quirk (reported, not changed):** the tokenizer `[a-z0-9']+`
keeps the ASCII apostrophe but splits on the curly `’`. As a result, `I'm` and
`I’m` become different tokens (`i'm` vs `i`, `m`). Curly apostrophes occur in
395 of the 3125 training texts.

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
│   ├── mdl_variants.py             <- mdl_prior / mdl_two_part / mdl_prequential
│   ├── ablation.py                 <- n-gram order / smoothing ablation configs
│   ├── multilabel/                 <- separate multi-label data track (config, preprocessing, sampling, data API)
│   ├── single_label_27/            <- separate 27-class single-label data track (config, preprocessing, sampling, data API)
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

## 11. Multi-label track (separate)

A self-contained multi-label data pipeline for two independent classifiers
(multi-label MDL and multi-label LSTM). It reads the same cached raw
GoEmotions files but changes **nothing** in the single-label track: no
existing config, function, dataset, manifest or result is touched.

**Dataset (Option A).** Targets, in this fixed column order:
`[joy, sadness, anger, fear]`. A comment is kept if its original labels
contain at least one target. Its labels are projected onto those four 0/1
columns, every other label is ignored, and comments with no target are
discarded. Official partitions are preserved, and ids follow the
single-label scheme, so every single-label example appears in the
multi-label data with the same id and text.

| Partition | Raw | Retained | Discarded | joy | sadness | anger | fear | Multi-label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 43,410 | 4,883 | 38,527 | 1,452 | 1,326 | 1,567 | 596 | 58 |
| validation | 5,426 | 595 | 4,831 | 172 | 143 | 195 | 90 | 5 |
| test | 5,427 | 590 | 4,837 | 161 | 156 | 198 | 78 | 3 |

`scripts/prepare_multilabel_data.py` checks these counts (stored in
`configs/multilabel.yaml` under `expected_counts`). On any mismatch it
writes nothing and prints the discrepancy.

**Training ladder.** Sizes `[100, 250, 500, 1000, 1500, 2000, 2500, 4883]`
× seeds `[42..46]`, stratified by label combination:

1. Each observed combination (e.g. `sadness+fear`) is one stratum, and its
   ids are put in a seeded order.
2. One global order is built by sequential Webster / Sainte-Laguë
   allocation: each next slot goes to the stratum with the largest
   `N_s / (allocated_s + 0.5)`, with ties going to the canonical stratum
   order.
3. Subset `n` is the first `n` ids of that order.

So every subset has exactly `n` examples, smaller subsets are prefixes of
larger ones, `n=4883` is the whole training partition, and every
combination stays within ±1 of its proportional share. Rare combinations
(4–17 training examples) only appear once their share reaches about one
half. For example, there are no two-emotion examples at `n=100`. Details
are in [`multilabel/sampling.py`](src/emotion_ait/multilabel/sampling.py).

Each manifest `data/subsets/multilabel_ladder/seed_<s>/n<size>.json` holds:
- the dataset id, policy, target order and stratification scheme;
- the seed, requested size and actual size;
- the ordered `example_ids` and their parallel `targets`;
- per-emotion `positive_counts` and `combination_counts`;
- `example_ids_sha256` and `train_partition_sha256`, so both developers can
  confirm they are using identical data.

**Commands** (`data/` is gitignored, so each developer runs these once.
Both are deterministic and refuse to overwrite existing output unless you
pass `--force`.)

```bash
python scripts/prepare_multilabel_data.py     # -> data/processed/multilabel/{train,validation,test}.csv + dataset_stats.json
python scripts/generate_multilabel_ladder.py  # -> data/subsets/multilabel_ladder/ (40 manifests)
python -m pytest tests/test_multilabel_pipeline.py
```

**Shared API** for both developers. Import only from
`emotion_ait.multilabel.data`. It returns plain pandas/numpy objects with
no model-specific processing:

```python
from emotion_ait.multilabel.data import (
    TARGET_LABELS,                            # ["joy", "sadness", "anger", "fear"]
    load_multilabel_dataframe,                # (split) -> full processed partition
    load_multilabel_ladder_subset_dataframe,  # (seed, size) -> training subset, manifest order
    load_multilabel_ladder_ids,               # (seed, size) -> ordered ids
    get_multilabel_targets,                   # (df) -> (N, 4) int64 0/1, TARGET_LABELS order
    get_multilabel_example_ids,               # (df) -> list[str]
)

train = load_multilabel_ladder_subset_dataframe(seed=42, size=500)
val, test = load_multilabel_dataframe("validation"), load_multilabel_dataframe("test")
X_train, Y_train = train["text"].tolist(), get_multilabel_targets(train)
```

Notes for both developers:
- **Row order:** processed partitions keep the raw order, and ladder
  subsets follow the manifest order. Shuffle inside your own training code
  if you need to.
- **Stale data:** the subset loader raises an error if the processed data
  no longer matches the manifest.
- **Data roles:** train on ladder subsets only, tune thresholds on
  validation, and use test only for final reporting.
- **Single-label comparison:** the single-label test set (391 examples) is
  a subset of this test set, which allows a direct comparison with the
  single-label results.

## 12. 27-class single-label track (separate)

A third, self-contained track. Each example has exactly one of the 27
GoEmotions emotions (`neutral` excluded). It reuses the cached raw data and
changes nothing in the 4-class or multi-label tracks.

**Policy.** A comment is kept if and only if its complete original label
set is exactly one label and that label isn't `neutral`. Multi-emotion,
neutral-only and neutral+emotion comments are discarded, never projected
onto a single class. Classes are numbered in the original GoEmotions order
with `neutral` removed: `admiration` = 0 … `surprise` = 26.

| Partition | Raw | Retained | Neutral only | Neutral + emotion | Multi-emotion | Min class | Max class | Imbalance |
|---|---:|---:|---:|---:|---:|---|---|---:|
| train | 43,410 | 23,485 | 12,823 | 1,396 | 5,706 | grief 39 | admiration 2,710 | 69.5× |
| validation | 5,426 | 2,956 | 1,592 | 174 | 704 | grief 6 | admiration 326 | 54.3× |
| test | 5,427 | 2,984 | 1,606 | 181 | 656 | grief 2 | admiration 348 | 174× |

All per-class counts are in `configs/single_label_27.yaml` (`expected_counts`)
and are checked on every run. The anger/fear/joy/sadness rows are exactly
the 4-class track's examples, with the same ids and texts.

**Ladder.** Sizes `[500, 1000, 2000, 4000, 8000, 12000, 16000, 23485]` ×
seeds `[42..46]`. Sampling is stratified by class using the same sequential
Webster scheme as the multi-label track: exact sizes, nested within each
seed, and every class within ±1 of its proportional share. With strictly
proportional sampling, all 27 classes are present only from n = 303 upwards,
since grief makes up 0.17% of the training set. The 4-class ladder sizes
(100, 250) would therefore miss rare classes: n=100 has only 23 classes. The
ladder starts at 500, where every class is present. The rare classes then
grow as follows (grief / pride / nervousness / relief):

| Size | Rare-class counts |
|---|---|
| 500 | 1 / 1 / 2 / 2 |
| 2000 | 3 / 4 / 7 / 8 |
| 8000 | 13 / 17 / 29 / 30 |

The generator refuses to write a ladder in which any class is missing.

**Commands** (`data/` is gitignored; both steps are deterministic and refuse
to overwrite existing output unless you pass `--force`):

```bash
python scripts/prepare_single_label_27_data.py     # -> data/processed/single_label_27/{train,validation,test}.csv
python scripts/generate_single_label_27_ladder.py  # -> data/subsets/single_label_27_ladder/ (40 manifests)
python -m pytest tests/test_single_label_27_pipeline.py
```

**Shared API** (`emotion_ait.single_label_27.data`):

```python
from emotion_ait.single_label_27.data import (
    CLASS_NAMES, LABEL_TO_INDEX, INDEX_TO_LABEL,     # fixed 27-class order
    load_single_label_27_dataframe,                  # (split) -> full partition
    load_single_label_27_ladder_subset_dataframe,    # (seed, size) -> subset, manifest order
    load_single_label_27_ladder_ids,                 # (seed, size) -> ordered ids
    get_single_label_27_labels,                      # (df) -> (N,) int64 class indices
    get_single_label_27_example_ids,                 # (df) -> list[str]
)
train = load_single_label_27_ladder_subset_dataframe(seed=42, size=2000)
X, y = train["text"].tolist(), get_single_label_27_labels(train)
```

**Shared experiment runner.** Both models go through one runner
([`single_label_27/experiments.py`](src/emotion_ait/single_label_27/experiments.py)),
so they train on the same subsets and are scored by the same code as every
other track: `run_experiment` with the 27-class mapping. Each run records:
- validation and test macro-F1, and test accuracy;
- per-class precision, recall and F1 for all 27 classes;
- train and inference time;
- a test predictions file;
- a `config_summary` with the dataset id and the hash of the subset's ids.

Files are prefixed `sl27_` and never collide with other tracks. A run
refuses to start if its results CSV already exists.

```bash
python scripts/run_single_label_27_experiments.py --model mdl --n 2 --k 1     # -> results/metrics/sl27_mdl_n2_k1_ladder_results.csv
python scripts/run_single_label_27_experiments.py --model lstm                # needs LSTMClassifier(num_classes=...), see below
python scripts/run_single_label_27_experiments.py --factory my_pkg.models:build --name lstm27   # any EmotionClassifier
python scripts/run_single_label_27_experiments.py --model mdl --sizes 500 --seeds 42 --results-tag debug   # quick debug run
```

**27-class MDL** ([`single_label_27/mdl.py`](src/emotion_ait/single_label_27/mdl.py),
`MDL27Classifier`) extends the existing `MDLClassifier` without changing it.
- **27 models, one vocabulary:** one bigram language model per class, in
  `CLASS_NAMES` order, all sharing one vocabulary built from the training
  subset. Defaults are `n=2`, `k=1`.
- **Decision rule:** `argmin_e L(x|M_e)`, with ties going to the lowest
  class index.
- **Optional prior:** `--prior` adds `-log2 P(e)` (training class
  frequency). That variant is named `sl27_mdl_prior_*`.
- **Kneser-Ney smoothing:** `--smoothing kneser_ney --kn-backoff {class,shared}`
  selects interpolated Kneser-Ney (any order `--n`) instead of add-k. Discounts are
  estimated from training counts as `D = n1/(n1+2 n2)` unless `--kn-discount`
  fixes them. The back-off distribution is either per class or pooled over the
  training subset. Every distribution is proper (sums to 1 and is positive)
  over the shared vocabulary, `</s>` and `<unk>`. Runs are named
  `sl27_mdl_n<n>_kn{class,shared}`.
- **Classes missing from training:** such a class still gets an empty model
  with finite description lengths, but it is reported in `missing_classes`
  and is never predicted.
- **Diagnostics:** `--diagnostics` writes
  `results/diagnostics/sl27_<name>_ladder_n<size>_<seed>.json`, with per-class
  training examples, unique bigrams, vocabulary coverage, mean training
  bits, validation/test OOV counts and predictions per class.
- **Other methods:** `predict_scores(texts)` returns an `(N, 27)` score
  matrix, and `score_breakdown(texts)` returns the conditional, prior and
  total bits.

```bash
python scripts/run_single_label_27_experiments.py --model mdl --n 2 --k 1                    # no prior (40 runs)
python scripts/run_single_label_27_experiments.py --model mdl --n 2 --k 1 --prior            # prior   (40 runs)
python scripts/run_single_label_27_experiments.py --model mdl --sizes 23485 --seeds 42 --results-tag seed42 --diagnostics
```

**27-class bigram Naive Bayes** ([`single_label_27/naive_bayes.py`](src/emotion_ait/single_label_27/naive_bayes.py),
`BigramNaiveBayes27`) is a separate generative classifier, written for comparison with MDL.
- **Model:** one add-k bigram language model per class, estimated as
  `P(w_i | w_{i-1}, e) = (count_e(w_{i-1}, w_i) + k) / (count_e(w_{i-1}) + kV)`.
  It reuses only the tokenizer, padding, n-gram and add-k helpers of
  `mdl_classifier.py`, so vocabulary, `<s>`/`</s>`/`<unk>` handling and
  smoothing are identical to MDL.
- **Two decision rules:**
  - `--variant ml`: `argmax_e log P(x|e)`.
  - `--variant map`: `argmax_e [log P(x|e) + log P(e)]`, with `P(e) = N_e/N`
    from training labels only.
- **Equivalence with MDL:** log-probabilities are base 2, so
  `log2 P(x|e) = -L(x|M_e)` exactly. ML Naive Bayes *is* conditional MDL, and
  MAP Naive Bayes is MDL with the class-prior code `-log2 P(e)`. This is
  verified in tests (bit-identical scores) and on real data (E27, E28): there
  are 0 differing predictions against the MDL runs.
- **Missing classes:** a class with no training data gets an empty model and
  is never predicted, as in `MDL27Classifier`.

```bash
python scripts/run_single_label_27_experiments.py --model naive_bayes --variant ml    # -> sl27_nb_ml_n2_k1
python scripts/run_single_label_27_experiments.py --model naive_bayes --variant map   # -> sl27_nb_map_n2_k1
```

**27-class neural bigram MDL** ([`single_label_27/neural_bigram.py`](src/emotion_ait/single_label_27/neural_bigram.py),
`NeuralBigramMDL27`) uses one PyTorch bigram language model per emotion.
- **Architecture:** `softmax(W · Embedding(w_{i-1}) + b)` over the same `V`
  prediction tokens as the count-based MDL (training words, `</s>` and
  `<unk>`), with the same tokenizer and padding and a vocabulary built from
  the training subset only.
- **Training:**
  - Each class model is trained from scratch (Adam, cross-entropy) on that
    class's training texts only.
  - Early stopping uses that class's validation texts: per-token validation
    loss, a patience limit, and the best epoch's weights are kept. Test data
    is never seen during fitting.
  - Defaults: `embedding_dim=128`, `lr=1e-3`, `batch_size=128`,
    `epochs=50`, `patience=2`, `weight_decay=0`.
- **Prediction:** `argmin_e L(x|M_e)`, where
  `L(x|M_e) = -(1/ln 2) Σ_i ln P_θ(w_i|w_{i-1},e)` is the summed token
  log-loss in bits. This is the *conditional* description length: the
  network parameters are not encoded, so it is not two-part MDL. `--prior`
  is a separate variant that adds `-log2 P(e)`.
- **Speed:** the softmax normaliser is computed once per distinct previous
  token, then reused for every text.
- **Training metadata:** each run writes it automatically to
  `results/diagnostics/sl27_neural_bigram_mdl_ladder_n<size>_<seed>.json`. It
  records epochs trained, the best epoch and validation bits per token per
  class, training tokens and examples per class, per-class and total
  training time, parameter counts, vocabulary size, device, and test/val OOV
  counts and predictions.
- **Reproducibility:** seeds for Python, NumPy, PyTorch and CUDA come from
  the ladder seed, and class `c` uses `seed*1000 + c`. `--device cpu` is
  bit-reproducible; CUDA (`--device auto`, the default when a GPU is present)
  can differ in the last digits.

```bash
python scripts/run_single_label_27_experiments.py --model neural_bigram_mdl --sizes 500 --seeds 42 --results-tag smoke   # smoke test
python scripts/run_single_label_27_experiments.py --model neural_bigram_mdl                                               # full ladder -> sl27_neural_bigram_mdl
```

`--factory` takes `build(seed: int) -> (model, config_summary_dict)`, which
returns a fresh, unfitted `EmotionClassifier`. This lets you plug in a model
without editing the shared script. `MDLClassifier` already handles 27
classes. `LSTMClassifier` is still 4-class only: it rejects labels above 3
and has 4 outputs. `--model lstm` stays disabled until the LSTM accepts a
`num_classes` argument.

**Evaluation caveat.** The test set has 2 grief, 7 pride, 7 relief and 12
nervousness examples. Per-class F1 for those classes is extremely noisy, and
macro-F1 is sensitive to them, so report micro-F1 (accuracy) and macro-F1
together.

## Experiment log

Every experiment run so far is listed in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md): its date, track,
configuration, the question it answered, its validation and test results, and its conclusion.
The descriptive fields come from [`experiments/registry.yaml`](experiments/registry.yaml), and every
number is recomputed from `results/metrics/*.csv`. To log a new experiment, run it, add an entry to
the registry and regenerate the log:

```bash
python scripts/update_experiment_log.py
```

`tests/test_experiment_log.py` fails if any results file is missing from the registry.

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
