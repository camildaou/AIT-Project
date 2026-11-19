"""emotion_ait: minimum description length (MDL) and LSTM emotion
classifiers on GoEmotions, with a shared data pipeline and evaluation.

Data and protocol:

* :mod:`emotion_ait.config` -- typed configuration loading.
* :mod:`emotion_ait.data_loader` -- raw GoEmotions download/caching, processed splits.
* :mod:`emotion_ait.preprocessing` -- single-label 4-class filtering.
* :mod:`emotion_ait.sampling` -- deterministic, nested, stratified training subsets.
* :mod:`emotion_ait.interfaces` -- shared model interface (ABC).
* :mod:`emotion_ait.evaluation` -- metrics and results schema.
* :mod:`emotion_ait.experiments` -- one train/evaluate run.

Models (4-class track):

* :mod:`emotion_ait.mdl_classifier` -- conditional MDL with add-k n-gram codes.
* :mod:`emotion_ait.mdl_kneser_ney` -- conditional MDL with Kneser-Ney codes.
* :mod:`emotion_ait.mdl_variants` -- class prior, per-text two-part and prequential variants.
* :mod:`emotion_ait.two_part_mdl` -- dataset-level two-part code length (model selection).
* :mod:`emotion_ait.lstm_classifier` -- LSTM baseline.
* :mod:`emotion_ait.baselines` -- trivial reference models.

Other tracks: :mod:`emotion_ait.single_label_27` (27 classes: MDL, Naive Bayes,
neural bigram codes) and :mod:`emotion_ait.multilabel` (data pipeline only).
"""

__version__ = "0.1.0"
