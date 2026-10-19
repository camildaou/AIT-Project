# Data directory

This directory is populated by `scripts/prepare_data.py` and is mostly
git-ignored (see the repository `.gitignore`) because its contents are
either downloaded from the internet or deterministically regenerable.

* `raw/` -- cached parquet copies of the raw GoEmotions splits (`train`,
  `validation`, `test`), as downloaded from the Hugging Face Hub
  (`google-research-datasets/go_emotions`, config `simplified`). Not
  committed; regenerate with `python scripts/prepare_data.py`.
* `processed/` -- the filtered 4-class single-label subset, one CSV per
  split (`train.csv`, `validation.csv`, `test.csv`) with columns
  `example_id, text, label, label_name, original_split`, plus
  `dataset_stats.json` documenting the real retained counts per split
  per class. Not committed (regenerable), but see the project README
  for the actual counts observed when this pipeline was run.
* `subsets/` -- one JSON file per deterministic training subset (e.g.
  `train_pct10.json`, `train_abs250.json`) containing the sorted list
  of selected `example_id`s and sampling metadata, plus a combined
  `manifest.json` (metadata only, no ids, small enough to inspect
  quickly). Not committed (regenerable from the same seed).

Regenerate everything with:

```
python scripts/prepare_data.py
python scripts/verify_data.py
```
