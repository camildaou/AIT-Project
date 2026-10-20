# Qualitative analysis of results

## Evaluation context

Both methods were evaluated on the same 391-example test set: 131 anger, 65 fear, 93 joy, and 102 sadness examples. Accuracy is the fraction of all examples classified correctly. Macro-F1 is the unweighted average of the four class F1 scores, so a good score on the most common class cannot hide weak performance on another class. For that reason, Macro-F1 is the stronger headline metric here, with accuracy as a useful secondary measure.

The first-training experiment has one run per size (seed 42), with actual sizes 100, 157, 249, 313, 500, 625, 1,250, 2,188, and 3,125. The ladder experiment has eight sizes and five seeds per size. The ladder sizes are 100, 249, 500, 1,000, 1,499, 2,000, 2,500, and 3,125. The sampling code makes the subsets nested and stratified within each seed, so larger sizes retain examples from smaller sizes and add more examples. Ladder summary values below are means across the five runs; standard deviations describe run-to-run spread, not confidence intervals.

The score-chart y-axes are narrowed to show differences more clearly, while staying within the valid 0–1 range. This makes slopes look steeper than they would on an axis starting at zero; use the tables for exact comparisons.

## First-training results

Values below are Macro-F1 / accuracy, shown as percentages.

| Training size | MDL | LSTM | Macro-F1 lead |
|---:|---:|---:|---:|
| 100 | 28.7 / 29.2 | 21.6 / 37.1 | MDL +7.1 pp |
| 157 | 31.0 / 30.7 | 30.7 / 41.2 | MDL +0.3 pp |
| 249 | 35.2 / 34.8 | 34.4 / 38.1 | MDL +0.8 pp |
| 313 | 38.6 / 38.4 | 37.6 / 44.8 | MDL +1.0 pp |
| 500 | 43.3 / 43.7 | 33.3 / 38.6 | MDL +10.0 pp |
| 625 | 42.5 / 43.0 | 53.9 / 54.0 | LSTM +11.4 pp |
| 1,250 | 45.7 / 47.1 | 52.9 / 56.8 | LSTM +7.2 pp |
| 2,188 | 48.7 / 50.4 | 61.3 / 61.9 | LSTM +12.7 pp |
| 3,125 | 49.5 / 51.7 | 66.1 / 66.2 | LSTM +16.6 pp |

MDL has higher Macro-F1 through 500 examples, although the gaps at 157, 249, and 313 examples are small. LSTM takes the lead at 625 examples and has a larger margin at 2,188 and 3,125. From 100 to 3,125, Macro-F1 rises 20.7 percentage points for MDL and 44.4 points for LSTM.

The single-seed curves are not monotonic. LSTM Macro-F1 drops from 37.6% at 313 examples to 33.3% at 500, then climbs to 53.9% at 625. MDL has a smaller dip from 43.3% at 500 to 42.5% at 625. More training data does not guarantee a higher score in every individual run; these fluctuations make the multi-seed ladder more informative about the general pattern.

At 100 examples, LSTM accuracy is 37.1%, above MDL's 29.2%, but its Macro-F1 is lower (21.6% versus 28.7%). LSTM's anger F1 is 52.6% at that size, while its fear, joy, and sadness F1 scores are 10.4%, 18.0%, and 5.6%. This imbalance across classes explains why accuracy alone makes the result look stronger than Macro-F1 does. A classifier that always predicted anger would get 33.5% accuracy on this test set because anger is the largest class.

## Ladder learning curves

Values are mean Macro-F1 ± standard deviation / mean accuracy, in percentages.

| Training size | MDL Macro-F1 ± SD / accuracy | LSTM Macro-F1 ± SD / accuracy | Mean Macro-F1 lead |
|---:|---:|---:|---:|
| 100 | 29.9 ± 2.4 / 29.9 | 24.1 ± 2.5 / 36.5 | MDL +5.8 pp |
| 249 | 35.6 ± 1.0 / 35.3 | 34.9 ± 5.1 / 41.4 | MDL +0.7 pp |
| 500 | 42.2 ± 1.6 / 42.7 | 44.4 ± 6.8 / 49.2 | LSTM +2.2 pp |
| 1,000 | 43.6 ± 2.1 / 44.7 | 51.5 ± 6.8 / 53.4 | LSTM +7.9 pp |
| 1,499 | 46.7 ± 1.8 / 48.1 | 58.6 ± 1.3 / 61.5 | LSTM +11.9 pp |
| 2,000 | 48.6 ± 2.1 / 50.1 | 61.2 ± 3.3 / 62.5 | LSTM +12.7 pp |
| 2,500 | 48.4 ± 1.1 / 50.2 | 65.8 ± 1.7 / 67.1 | LSTM +17.4 pp |
| 3,125 | 49.5 ± 0.0 / 51.7 | 64.5 ± 5.6 / 65.6 | LSTM +15.0 pp |

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
