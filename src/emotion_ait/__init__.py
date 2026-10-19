"""emotion_ait: shared data pipeline and evaluation infrastructure for
comparing an MDL-based and an LSTM-based GoEmotions emotion classifier.

See the repository README for the project overview. Submodules:

* :mod:`emotion_ait.config` -- typed configuration loading.
* :mod:`emotion_ait.data_loader` -- raw GoEmotions download/caching.
* :mod:`emotion_ait.preprocessing` -- single-label 4-class filtering.
* :mod:`emotion_ait.sampling` -- deterministic, nested, stratified
  training subsets.
* :mod:`emotion_ait.interfaces` -- shared model interface (ABC).
* :mod:`emotion_ait.baselines` -- trivial reference models.
* :mod:`emotion_ait.mdl_classifier` / :mod:`emotion_ait.lstm_classifier`
  -- model stubs for the two collaborators.
* :mod:`emotion_ait.evaluation` -- shared metrics and results schema.
* :mod:`emotion_ait.experiments` -- orchestration glue.
"""

__version__ = "0.1.0"
