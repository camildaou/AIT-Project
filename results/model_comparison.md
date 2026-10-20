# Qualitative analysis of results

## Evaluation context

Both methods were evaluated on the same 391-example test set: 131 anger, 65 fear, 93 joy, and 102 sadness examples. Accuracy is the fraction of all examples classified correctly. Macro-F1 is the unweighted average of the four class F1 scores, so a good score on the most common class cannot hide weak performance on another class. For that reason, Macro-F1 is the stronger headline metric here, with accuracy as a useful secondary measure.

The first-training experiment has one run per size (seed 42), with actual sizes 100, 157, 249, 313, 500, 625, 1,250, 2,188, and 3,125. The ladder experiment has eight sizes and five seeds per size. The ladder sizes are 100, 249, 500, 1,000, 1,499, 2,000, 2,500, and 3,125. The sampling code makes the subsets nested and stratified within each seed, so larger sizes retain examples from smaller sizes and add more examples. Ladder summary values below are means across the five runs; standard deviations describe run-to-run spread, not confidence intervals.

The score-chart y-axes are narrowed to show differences more clearly, while staying within the valid 0–1 range. This makes slopes look steeper than they would on an axis starting at zero; use the tables for exact comparisons.

## First-training results

Values below are Macro-F1 / accuracy, shown as percentages.

| Training size | MDL Macro-F1 | MDL accuracy | LSTM Macro-F1 | LSTM accuracy | Macro-F1 lead |
|---:|---:|---:|---:|---:|---:|
| 100 | 28.7% | 29.2% | 21.6% | 37.1% | MDL +7.1 pp |
| 157 | 31.0% | 30.7% | 30.7% | 41.2% | MDL +0.3 pp |
| 249 | 35.2% | 34.8% | 34.4% | 38.1% | MDL +0.8 pp |
| 313 | 38.6% | 38.4% | 37.6% | 44.8% | MDL +1.0 pp |
| 500 | 43.3% | 43.7% | 33.3% | 38.6% | MDL +10.0 pp |
| 625 | 42.5% | 43.0% | 53.9% | 54.0% | LSTM +11.4 pp |
| 1,250 | 45.7% | 47.1% | 52.9% | 56.8% | LSTM +7.2 pp |
| 2,188 | 48.7% | 50.4% | 61.3% | 61.9% | LSTM +12.7 pp |
| 3,125 | 49.5% | 51.7% | 66.1% | 66.2% | LSTM +16.6 pp |

MDL has higher Macro-F1 through 500 examples, although the gaps at 157, 249, and 313 examples are small. LSTM takes the lead at 625 examples and has a larger margin at 2,188 and 3,125. From 100 to 3,125, Macro-F1 rises 20.7 percentage points for MDL and 44.4 points for LSTM.

The single-seed curves are not monotonic. LSTM Macro-F1 drops from 37.6% at 313 examples to 33.3% at 500, then climbs to 53.9% at 625. MDL has a smaller dip from 43.3% at 500 to 42.5% at 625. More training data does not guarantee a higher score in every individual run; these fluctuations make the multi-seed ladder more informative about the general pattern.

At 100 examples, LSTM accuracy is 37.1%, above MDL's 29.2%, but its Macro-F1 is lower (21.6% versus 28.7%). LSTM's anger F1 is 52.6% at that size, while its fear, joy, and sadness F1 scores are 10.4%, 18.0%, and 5.6%. This imbalance across classes explains why accuracy alone makes the result look stronger than Macro-F1 does. A classifier that always predicted anger would get 33.5% accuracy on this test set because anger is the largest class.

## Ladder learning curves

Values are mean Macro-F1 ± standard deviation / mean accuracy, in percentages.

| Training size | MDL Macro-F1 (mean ± SD) | MDL accuracy | LSTM Macro-F1 (mean ± SD) | LSTM accuracy | Mean Macro-F1 lead |
|---:|---:|---:|---:|---:|---:|
| 100 | 29.9% ± 2.4 | 29.9% | 24.1% ± 2.5 | 36.5% | MDL +5.8 pp |
| 249 | 35.6% ± 1.0 | 35.3% | 34.9% ± 5.1 | 41.4% | MDL +0.7 pp |
| 500 | 42.2% ± 1.6 | 42.7% | 44.4% ± 6.8 | 49.2% | LSTM +2.2 pp |
| 1,000 | 43.6% ± 2.1 | 44.7% | 51.5% ± 6.8 | 53.4% | LSTM +7.9 pp |
| 1,499 | 46.7% ± 1.8 | 48.1% | 58.6% ± 1.3 | 61.5% | LSTM +11.9 pp |
| 2,000 | 48.6% ± 2.1 | 50.1% | 61.2% ± 3.3 | 62.5% | LSTM +12.7 pp |
| 2,500 | 48.4% ± 1.1 | 50.2% | 65.8% ± 1.7 | 67.1% | LSTM +17.4 pp |
| 3,125 | 49.5% ± 0.0 | 51.7% | 64.5% ± 5.6 | 65.6% | LSTM +15.0 pp |

MDL improves steadily overall, from 29.9% Macro-F1 at 100 examples to 49.5% at 3,125. Its curve flattens after about 1,499 examples: the score changes from 46.7% to 49.5% over the remaining sizes, with a small dip at 2,500. LSTM gains more as data grows, rising from 24.1% to 65.8% at 2,500, then averaging 64.5% at 3,125. Its net increase from 100 to 3,125 is 40.4 percentage points.

The five seeds show how consistent the model comparison is at each size. MDL beats LSTM on Macro-F1 in all five runs at size 100. At 249, MDL wins three runs and LSTM wins two, consistent with their small 0.7-point mean difference. LSTM wins four of five runs at sizes 500 and 1,000, then wins all five runs at every size from 1,499 onward. The larger LSTM advantage at those sizes is therefore consistent across these runs, though five seeds are not enough to establish statistical significance.

LSTM's mean score falls from 65.8% at 2,500 to 64.5% at 3,125. Seed 43 accounts for much of this drop: its Macro-F1 goes from 64.3% to 54.7%; the other four seeds are flat or higher. The standard deviation also rises from 1.7 points at 2,500 to 5.6 at full size. This looks like run-to-run variation rather than evidence that the extra training data consistently hurts LSTM. MDL's standard deviation is zero at full size because the full dataset is the same across seeds and MDL produces the same score each time.

## Per-class F1

The ladder's five-seed mean class F1 at the smallest and largest sizes shows where the gains come from:

| Emotion | MDL: 100 → 3,125 | LSTM: 100 → 3,125 | LSTM lead at 3,125 |
|---|---:|---:|---:|
| Anger | 32.7% → 59.6% | 48.0% → 69.8% | +10.2 pp |
| Fear | 28.7% → 33.3% | 6.0% → 55.9% | +22.6 pp |
| Joy | 33.2% → 53.5% | 23.5% → 68.1% | +14.6 pp |
| Sadness | 24.9% → 51.6% | 18.8% → 64.3% | +12.7 pp |

Fear is MDL's weakest class across the ladder: its mean F1 is 31.3% at 500, then stays near 30–34% through full size. LSTM also starts very weak on fear (6.0% at size 100), but rises to 37.0% at 1,000 and 55.9% at full size. At 500, LSTM's fear F1 (22.4%) is still below MDL's (31.3%), even though LSTM is already ahead on anger, joy, and sadness. By 1,000 examples LSTM's mean F1 is higher for all four classes.

At full size, LSTM's strongest mean class is anger (69.8%), followed by joy (68.1%), sadness (64.3%), and fear (55.9%). The largest final gain over MDL is fear, at 22.6 percentage points. LSTM still varies by seed for individual classes: at full size, its fear F1 ranges from 46.0% to 62.9%, and joy ranges from 51.5% to 77.1%. The per-seed class plots make this variation visible.

## Overall conclusion and limits

The results support a data-size-dependent comparison: MDL is competitive with very small training sets, while LSTM benefits more as training data grows and is consistently ahead in the larger ladder runs. Macro-F1 and per-class F1 are essential to that conclusion; accuracy by itself conceals the weak small-data LSTM scores on fear, joy, and sadness.

The first-training experiment has one seed per size; the ladder has five. Standard deviations are descriptive run-to-run spreads, not confidence intervals, and the results do not establish statistical significance. The test set is also modest in size, especially for fear (65 examples), so small differences should be interpreted cautiously.

## Ladder per-run details

The following tables give the saved metrics for every ladder seed and training size. Accuracy, Macro-F1, and per-class F1 values are percentages rounded to one decimal place. Validation Macro-F1 is included because it was used during model selection. ?n? is the actual training subset size.

### MDL

| n | Seed | Test accuracy | Test Macro-F1 | Validation Macro-F1 | Anger F1 | Fear F1 | Joy F1 | Sadness F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 42 | 29.2% | 28.7% | 28.6% | 32.4% | 29.4% | 32.6% | 20.5% |
| 100 | 43 | 30.2% | 30.5% | 29.0% | 38.2% | 27.8% | 26.5% | 29.7% |
| 100 | 44 | 28.4% | 29.1% | 25.6% | 22.5% | 26.3% | 40.0% | 27.6% |
| 100 | 45 | 33.2% | 33.6% | 30.7% | 37.9% | 29.8% | 35.6% | 31.1% |
| 100 | 46 | 28.4% | 27.4% | 27.1% | 32.4% | 30.3% | 31.4% | 15.6% |
| 249 | 42 | 34.8% | 35.2% | 33.4% | 40.2% | 29.3% | 34.9% | 36.2% |
| 249 | 43 | 36.1% | 36.4% | 34.0% | 46.8% | 27.5% | 34.0% | 37.4% |
| 249 | 44 | 36.3% | 36.6% | 33.8% | 40.4% | 27.1% | 42.1% | 36.9% |
| 249 | 45 | 33.8% | 34.1% | 30.9% | 33.1% | 29.8% | 37.3% | 36.1% |
| 249 | 46 | 35.8% | 35.5% | 32.4% | 42.6% | 31.3% | 38.1% | 30.2% |
| 500 | 42 | 43.7% | 43.3% | 42.9% | 49.8% | 36.3% | 48.3% | 38.7% |
| 500 | 43 | 41.9% | 41.3% | 40.9% | 52.8% | 27.6% | 41.5% | 43.3% |
| 500 | 44 | 40.4% | 39.9% | 39.9% | 46.8% | 25.6% | 46.4% | 41.1% |
| 500 | 45 | 43.7% | 43.5% | 38.0% | 47.2% | 33.5% | 48.6% | 44.7% |
| 500 | 46 | 43.7% | 43.2% | 38.6% | 49.5% | 33.5% | 49.3% | 40.4% |
| 1000 | 42 | 45.5% | 44.2% | 43.9% | 55.5% | 30.0% | 50.9% | 40.4% |
| 1000 | 43 | 46.8% | 45.6% | 46.3% | 56.6% | 31.2% | 49.3% | 45.5% |
| 1000 | 44 | 41.7% | 40.5% | 44.8% | 49.1% | 25.6% | 46.8% | 40.2% |
| 1000 | 45 | 45.8% | 45.1% | 45.4% | 49.8% | 33.5% | 51.1% | 45.9% |
| 1000 | 46 | 43.7% | 42.5% | 44.5% | 52.3% | 27.4% | 48.0% | 42.2% |
| 1499 | 42 | 48.6% | 47.1% | 46.6% | 55.8% | 28.9% | 52.3% | 51.5% |
| 1499 | 43 | 50.4% | 48.5% | 47.1% | 61.7% | 32.6% | 51.0% | 48.7% |
| 1499 | 44 | 45.8% | 44.5% | 46.8% | 50.9% | 30.8% | 49.2% | 47.2% |
| 1499 | 45 | 49.1% | 48.1% | 50.2% | 53.9% | 36.4% | 53.9% | 48.1% |
| 1499 | 46 | 46.8% | 45.2% | 46.3% | 54.1% | 28.6% | 52.4% | 45.9% |
| 2000 | 42 | 50.6% | 49.0% | 46.7% | 57.3% | 33.3% | 55.6% | 49.8% |
| 2000 | 43 | 51.2% | 49.5% | 46.4% | 60.9% | 35.8% | 49.3% | 52.0% |
| 2000 | 44 | 51.4% | 49.8% | 50.9% | 58.5% | 36.4% | 54.5% | 50.0% |
| 2000 | 45 | 50.9% | 49.7% | 52.5% | 55.7% | 37.6% | 55.1% | 50.2% |
| 2000 | 46 | 46.5% | 44.8% | 48.1% | 54.8% | 28.8% | 51.7% | 44.1% |
| 2500 | 42 | 50.9% | 49.2% | 50.8% | 58.1% | 33.6% | 54.8% | 50.2% |
| 2500 | 43 | 49.1% | 47.5% | 47.0% | 57.7% | 35.7% | 48.6% | 48.0% |
| 2500 | 44 | 50.6% | 48.3% | 52.4% | 61.3% | 30.6% | 50.5% | 50.7% |
| 2500 | 45 | 51.4% | 49.8% | 52.6% | 56.9% | 36.2% | 53.7% | 52.2% |
| 2500 | 46 | 49.1% | 47.3% | 52.7% | 58.0% | 32.3% | 54.1% | 45.0% |
| 3125 | 42 | 51.7% | 49.5% | 52.7% | 59.6% | 33.3% | 53.5% | 51.6% |
| 3125 | 43 | 51.7% | 49.5% | 52.7% | 59.6% | 33.3% | 53.5% | 51.6% |
| 3125 | 44 | 51.7% | 49.5% | 52.7% | 59.6% | 33.3% | 53.5% | 51.6% |
| 3125 | 45 | 51.7% | 49.5% | 52.7% | 59.6% | 33.3% | 53.5% | 51.6% |
| 3125 | 46 | 51.7% | 49.5% | 52.7% | 59.6% | 33.3% | 53.5% | 51.6% |

### LSTM

| n | Seed | Test accuracy | Test Macro-F1 | Validation Macro-F1 | Anger F1 | Fear F1 | Joy F1 | Sadness F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 42 | 37.1% | 21.6% | 19.4% | 52.6% | 10.4% | 18.0% | 5.6% |
| 100 | 43 | 37.1% | 21.5% | 20.6% | 52.0% | 0.0% | 19.5% | 14.8% |
| 100 | 44 | 36.6% | 25.0% | 20.6% | 51.0% | 0.0% | 22.2% | 26.8% |
| 100 | 45 | 33.2% | 27.3% | 23.4% | 32.4% | 13.9% | 40.5% | 22.2% |
| 100 | 46 | 38.4% | 25.0% | 20.8% | 52.1% | 5.9% | 17.1% | 24.8% |
| 249 | 42 | 38.1% | 34.4% | 35.6% | 51.4% | 29.2% | 36.8% | 20.0% |
| 249 | 43 | 38.9% | 27.8% | 28.2% | 49.4% | 0.0% | 18.5% | 43.4% |
| 249 | 44 | 47.6% | 42.1% | 37.8% | 59.7% | 22.2% | 41.1% | 45.2% |
| 249 | 45 | 42.2% | 35.8% | 33.9% | 50.0% | 8.3% | 39.0% | 46.1% |
| 249 | 46 | 40.2% | 34.3% | 35.3% | 41.9% | 12.3% | 35.9% | 47.0% |
| 500 | 42 | 38.6% | 33.3% | 34.3% | 45.1% | 16.7% | 43.6% | 27.8% |
| 500 | 43 | 52.4% | 48.2% | 48.3% | 58.3% | 27.9% | 57.0% | 49.7% |
| 500 | 44 | 49.9% | 42.9% | 43.3% | 57.9% | 10.7% | 54.9% | 48.1% |
| 500 | 45 | 52.2% | 47.4% | 45.8% | 60.4% | 22.9% | 58.0% | 48.1% |
| 500 | 46 | 52.9% | 50.2% | 44.2% | 60.5% | 33.6% | 56.8% | 49.8% |
| 1000 | 42 | 46.8% | 42.5% | 44.5% | 49.0% | 21.1% | 49.5% | 50.3% |
| 1000 | 43 | 46.5% | 46.9% | 48.2% | 52.0% | 39.1% | 58.6% | 37.8% |
| 1000 | 44 | 54.5% | 54.2% | 50.3% | 61.7% | 41.8% | 65.2% | 48.0% |
| 1000 | 45 | 62.4% | 59.9% | 57.3% | 70.4% | 45.4% | 64.5% | 59.2% |
| 1000 | 46 | 56.8% | 54.0% | 58.1% | 66.0% | 37.9% | 62.9% | 49.3% |
| 1499 | 42 | 60.6% | 58.1% | 54.5% | 64.5% | 47.3% | 65.1% | 55.3% |
| 1499 | 43 | 61.9% | 59.3% | 64.0% | 72.2% | 47.1% | 61.9% | 56.1% |
| 1499 | 44 | 61.4% | 58.3% | 52.0% | 67.9% | 42.6% | 61.6% | 61.2% |
| 1499 | 45 | 63.4% | 60.4% | 55.4% | 72.6% | 39.2% | 70.3% | 59.6% |
| 1499 | 46 | 60.1% | 56.9% | 55.1% | 63.5% | 41.7% | 70.7% | 51.7% |
| 2000 | 42 | 65.7% | 63.8% | 54.4% | 68.5% | 55.1% | 64.5% | 67.1% |
| 2000 | 43 | 63.9% | 63.2% | 60.0% | 72.1% | 54.7% | 68.4% | 57.7% |
| 2000 | 44 | 61.4% | 59.6% | 57.0% | 66.4% | 48.9% | 61.1% | 62.1% |
| 2000 | 45 | 62.9% | 63.4% | 63.4% | 65.7% | 60.2% | 69.9% | 57.9% |
| 2000 | 46 | 58.6% | 56.1% | 55.7% | 66.7% | 43.5% | 58.7% | 55.6% |
| 2500 | 42 | 65.5% | 63.8% | 63.3% | 70.6% | 47.3% | 76.3% | 61.0% |
| 2500 | 43 | 66.8% | 64.3% | 59.6% | 70.3% | 51.4% | 73.9% | 61.7% |
| 2500 | 44 | 67.0% | 67.2% | 68.0% | 70.4% | 66.7% | 71.6% | 60.3% |
| 2500 | 45 | 66.8% | 66.0% | 63.6% | 68.7% | 57.9% | 73.4% | 64.2% |
| 2500 | 46 | 69.3% | 67.6% | 67.4% | 73.5% | 56.9% | 72.0% | 68.2% |
| 3125 | 42 | 66.2% | 66.1% | 65.9% | 66.7% | 62.9% | 68.9% | 65.9% |
| 3125 | 43 | 55.2% | 54.7% | 57.7% | 61.6% | 46.0% | 51.5% | 59.5% |
| 3125 | 44 | 66.5% | 66.0% | 64.8% | 67.0% | 59.4% | 74.6% | 63.1% |
| 3125 | 45 | 69.6% | 68.2% | 67.3% | 77.1% | 61.1% | 68.3% | 66.4% |
| 3125 | 46 | 70.3% | 67.6% | 70.9% | 76.6% | 50.0% | 77.1% | 66.7% |
