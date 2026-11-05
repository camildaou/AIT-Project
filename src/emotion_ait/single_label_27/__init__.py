"""Separate 27-class single-label track (all GoEmotions emotions except neutral).

Nothing here changes the 4-class single-label or the multi-label tracks.
Model code (MDL or LSTM) should only use :mod:`emotion_ait.single_label_27.data`.

Modules:

* :mod:`.config` -- ``configs/single_label_27.yaml``.
* :mod:`.preprocessing` -- raw GoEmotions -> ``data/processed/single_label_27/*.csv``.
* :mod:`.sampling` -- nested, class-stratified ladder manifests.
* :mod:`.data` -- the shared loading API and label mappings.
"""
