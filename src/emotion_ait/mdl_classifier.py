"""Stub for the MDL (Minimum Description Length) emotion classifier.

Owned by: Camil DAOU.

TODO (not implemented yet, by design -- see project spec section 5):
    * For each of the 4 target emotions, train a smoothed n-gram
      language model (order and smoothing scheme TBD by Person A) on
      the training texts belonging to that class only.
    * At prediction time, score an input text under each of the 4
      per-class language models (e.g. compression length / negative
      log-likelihood) and predict the class whose model assigns the
      shortest description length (MDL principle), i.e. the class
      under which the text is "cheapest" to encode.
    * ``save``/``load`` must persist the 4 fitted per-class language
      models (whatever internal representation is chosen) so that a
      fitted classifier can be restored without retraining.

This module intentionally does not fabricate predictions or pretend
that training has occurred: all public methods raise
``NotImplementedError`` until Person A implements them.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from .interfaces import EmotionClassifier


class MDLClassifier(EmotionClassifier):
    """Minimum Description Length emotion classifier (stub).

    Parameters
    ----------
    n:
        N-gram order to use for the per-class language models.
        Placeholder default; Person A should tune this.
    smoothing:
        Name of the smoothing scheme to apply (e.g. "laplace",
        "kneser-ney"). Placeholder; not yet implemented.
    """

    def __init__(self, n: int = 3, smoothing: str = "laplace") -> None:
        self.n = n
        self.smoothing = smoothing
        # TODO(person-a): one fitted per-class language model per target
        # emotion, keyed by integer label id.
        self._per_class_models: Optional[dict] = None

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "MDLClassifier":
        # TODO(person-a): fit one smoothed n-gram language model per
        # class on `train_data[train_data.label == class_id].text`.
        raise NotImplementedError(
            "MDLClassifier.fit is not implemented yet. See the module docstring TODOs."
        )

    def predict(self, texts: Sequence[str]) -> List[int]:
        # TODO(person-a): score each text under every per-class language
        # model and return the argmin-description-length class id.
        raise NotImplementedError(
            "MDLClassifier.predict is not implemented yet. See the module docstring TODOs."
        )

    def save(self, path: Path) -> None:
        # TODO(person-a): serialize `self._per_class_models` to `path`.
        raise NotImplementedError("MDLClassifier.save is not implemented yet.")

    @classmethod
    def load(cls, path: Path) -> "MDLClassifier":
        # TODO(person-a): deserialize per-class language models from `path`.
        raise NotImplementedError("MDLClassifier.load is not implemented yet.")
