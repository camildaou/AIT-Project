"""Separate multi-label track (Option A, targets ``[joy, sadness, anger, fear]``).

Nothing here changes the single-label pipeline. Model code (MDL or LSTM)
should only ever use :mod:`emotion_ait.multilabel.data`:

* :func:`~emotion_ait.multilabel.data.load_multilabel_dataframe`
* :func:`~emotion_ait.multilabel.data.load_multilabel_ladder_subset_dataframe`
* :func:`~emotion_ait.multilabel.data.get_multilabel_targets`
* :func:`~emotion_ait.multilabel.data.get_multilabel_example_ids`

Modules:

* :mod:`.config` -- ``configs/multilabel.yaml``.
* :mod:`.preprocessing` -- raw GoEmotions -> ``data/processed/multilabel/*.csv``.
* :mod:`.sampling` -- nested, label-combination-stratified ladder manifests.
* :mod:`.data` -- the shared loading API for both developers.
"""
