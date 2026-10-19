"""Simple, fully-working reference baselines.

These are *not* the research models under comparison (MDL vs. LSTM --
see :mod:`emotion_ait.mdl_classifier` / :mod:`emotion_ait.lstm_classifier`,
which are intentionally left as stubs). They exist purely so the shared
data pipeline, model interface, and evaluation protocol can be exercised
end-to-end today, with honestly-trained (if trivial) models.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .interfaces import EmotionClassifier


class MajorityClassBaseline(EmotionClassifier):
    """Always predicts the most frequent class seen during ``fit``."""

    def __init__(self) -> None:
        self._majority_class: Optional[int] = None

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "MajorityClassBaseline":
        if train_data.empty:
            raise ValueError("Cannot fit MajorityClassBaseline on an empty training set.")
        counts = Counter(train_data["label"].tolist())
        self._majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, texts: Sequence[str]) -> List[int]:
        if self._majority_class is None:
            raise RuntimeError("MajorityClassBaseline.predict called before fit().")
        return [self._majority_class for _ in texts]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"majority_class": self._majority_class}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MajorityClassBaseline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls()
        model._majority_class = payload["majority_class"]
        return model


class RandomBaseline(EmotionClassifier):
    """Predicts classes drawn from the empirical training label
    distribution, using a fixed seed for reproducibility."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._classes: Optional[List[int]] = None
        self._probabilities: Optional[List[float]] = None

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "RandomBaseline":
        if train_data.empty:
            raise ValueError("Cannot fit RandomBaseline on an empty training set.")
        counts = Counter(train_data["label"].tolist())
        total = sum(counts.values())
        self._classes = sorted(counts.keys())
        self._probabilities = [counts[c] / total for c in self._classes]
        return self

    def predict(self, texts: Sequence[str]) -> List[int]:
        if self._classes is None or self._probabilities is None:
            raise RuntimeError("RandomBaseline.predict called before fit().")
        rng = np.random.RandomState(self._seed)
        draws = rng.choice(self._classes, size=len(texts), p=self._probabilities)
        return [int(d) for d in draws]

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"seed": self._seed, "classes": self._classes, "probabilities": self._probabilities}, fh)

    @classmethod
    def load(cls, path: Path) -> "RandomBaseline":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        model = cls(seed=payload["seed"])
        model._classes = payload["classes"]
        model._probabilities = payload["probabilities"]
        return model
