# Emotion Classification by Minimum Description Length

Course project, Algorithmic Information & Artificial Intelligence (IA101, Télécom Paris).
Camil Daou, Tia Hakmeh.

We classify the emotion of Reddit comments ([GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions),
Demszky et al., 2020) with conditional minimum description length (MDL): one code
(language model) $M_e$ per emotion, and a comment $x$ is assigned to the emotion
whose code describes it in the fewest bits,

$$L(x \mid M_e) = -\sum_i \log_2 P(w_i \mid w_{i-n+1..i-1}, M_e), \qquad \hat e = \arg\min_e L(x \mid M_e).$$

The classifier is compared with an LSTM trained on identical data, across training
sizes, to ask when a simple code outperforms a more complex model and how much
complexity the data pay for.

## Main results

Test macro-F1, mean over 5 seeds, full training set.

| Track | MDL bigram, add-1 | MDL bigram, Kneser-Ney | Shared neural MDL | LSTM |
|---|---:|---:|---:|---:|
| 4 classes (3,125 training comments) | 0.606 | **0.724** | – | 0.645 |
| 27 classes (23,485 training comments) | 0.279 | 0.304 | 0.325 | **0.354** |

* 4 classes: the Kneser-Ney code is better than the LSTM at every training size (40/40 paired runs).
* 27 classes: Kneser-Ney is best up to ~2k comments, the shared neural code from 4k to 12k,
  the LSTM at full size; the LSTM overtakes Kneser-Ney between 12k and 16k comments.
* Adding the model cost $L(M_e)$ to each comment's score (per-text two-part MDL) predicts the
  smallest class for every comment (macro-F1 0.071); the model cost belongs to the dataset-level
  code, where it selects bigram Kneser-Ney among the n-gram configurations.
* A block-wise prequential label code gives the LSTM an effective learning cost of ~9·10³ bits
  (Kneser-Ney ~3.8·10³), far below the ~10⁸ bits of its weights at 32 bits each.

All experiments (E01–E37) are listed with their configuration and outcome in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), generated from [`experiments/registry.yaml`](experiments/registry.yaml).

## Data and protocol

* **Splits.** Official GoEmotions `simplified` splits (train 43,410 / validation 5,426 / test 5,427); never re-split.
* **4-class track.** Comments whose complete label set is exactly one of anger, fear, joy, sadness:
  3,125 / 357 / 391.
* **27-class track.** Comments with exactly one label, excluding `neutral`: 23,485 / 2,956 / 2,984
  (admiration 2,710 to grief 39 in train).
* **Multi-label track.** Data pipeline only (`src/emotion_ait/multilabel/`); not used in the
  experiments, since only 1.2% of comments with the four target emotions carry two of them.
* **Training-size ladders.** Nested, class-stratified subsets, 8 sizes × 5 seeds (42–46):
  100–3,125 (4 classes) and 500–23,485 (27 classes). All models use the same training ids.
* **Selection and evaluation.** Configurations are selected on validation macro-F1; test macro-F1,
  accuracy and Cohen's κ are reported. Models are compared with paired tests over seeds
  (t-test per size, Wilcoxon over the 40 runs of a ladder).

## Models

| Model | Module | Description |
|---|---|---|
| Conditional MDL, add-k | `emotion_ait.mdl_classifier` | per-emotion word n-gram code, $P = (c(h,w)+k)/(c(h)+kV)$ |
| Conditional MDL, Kneser-Ney | `emotion_ait.single_label_27.mdl`, `emotion_ait.mdl_kneser_ney` | interpolated Kneser-Ney, any order, per-class or shared back-off |
| MDL variants | `emotion_ait.mdl_variants` | class prior, per-text two-part cost, prequential coding |
| Dataset-level two-part code | `emotion_ait.two_part_mdl` | $\sum_e L(M_e) + \sum_i L(x_i \mid M_{y_i})$ for model selection |
| Naive Bayes | `emotion_ait.single_label_27.naive_bayes` | independent implementation; identical predictions to MDL |
| Neural bigram codes | `emotion_ait.single_label_27.neural_bigram`, `.shared_neural` | 27 separate networks, or one class-conditioned network |
| LSTM | `emotion_ait.lstm_classifier` | embedding 100, one LSTM layer of 128, softmax over classes |

All MDL models share one tokenizer (lowercase, `[a-z0-9']+`), one training-only vocabulary with
`<s>`, `</s>`, `<unk>`, and proper (normalised) conditional distributions.

## Repository layout

```
configs/                  YAML configurations (4-class, 27-class, multi-label)
src/emotion_ait/          package: data pipeline, models, evaluation
  single_label_27/        27-class track (data, MDL, Naive Bayes, neural codes)
  multilabel/             multi-label data pipeline
scripts/data/             download, filter, generate training ladders, verify
scripts/experiments/      training/evaluation runs (ladders, ablations, two-part, prequential)
scripts/analysis/         paired tests, figures, ablation report, experiment log
experiments/registry.yaml experiment registry (source of EXPERIMENT_LOG.md)
results/                  metrics, predictions, diagnostics, prequential traces
tests/                    unit and integration tests (pytest)
```

## Reproduction

```bash
pip install -r requirements.txt        # Python >= 3.9; PyTorch for the neural models

# data (deterministic; counts are checked against the configs)
python scripts/data/prepare_data.py                       # 4-class splits + subsets
python scripts/data/generate_ladder_subsets.py
python scripts/data/prepare_single_label_27_data.py
python scripts/data/generate_single_label_27_ladder.py

# 4-class experiments
python scripts/experiments/run_ladder_experiments.py --model mdl     # also: mdl_prior, mdl_two_part, mdl_prequential, lstm
python scripts/experiments/run_mdl_ablation.py --study all           # n-gram order and add-k
python scripts/experiments/run_mdl_kn_4class.py                      # Kneser-Ney
python scripts/experiments/run_two_part_mdl_4class.py                # dataset-level two-part code

# 27-class experiments
python scripts/experiments/run_single_label_27_experiments.py --model mdl --smoothing kneser_ney
python scripts/experiments/run_single_label_27_experiments.py --model shared_neural_mdl
python scripts/experiments/run_single_label_27_experiments.py --model lstm
python scripts/experiments/prequential_labels_27.py results/prequential_labels/preq_all.csv

# analysis
python scripts/analysis/paired_seed_tests.py
python scripts/analysis/update_experiment_log.py

pytest
```

Each run appends one row per (size, seed) to `results/metrics/<model>_ladder_results.csv`, with
the configuration, validation and test metrics; predictions are saved under `results/predictions/`.

## References

* D. Demszky et al. GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020.
* J. Rissanen. Modeling by shortest data description. Automatica, 1978.
* P. D. Grünwald. The Minimum Description Length Principle. MIT Press, 2007.
* E. Frank, C. Chui, I. H. Witten. Text categorization using compression models. DCC 2000.
* R. Kneser, H. Ney. Improved backing-off for m-gram language modeling. ICASSP 1995.
* L. Blier, Y. Ollivier. The description length of deep learning models. NeurIPS 2018.
