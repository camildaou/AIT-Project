"""Shared model interface for the MDL and LSTM emotion classifiers.

Both collaborators implement this interface and *only* this interface.
The evaluation and experiment-orchestration code in
:mod:`emotion_ait.evaluation` / :mod:`emotion_ait.experiments` depends
only on these methods, so the MDL and LSTM implementations are free to
use completely different internal representations (e.g. per-class
n-gram language models vs. a PyTorch ``nn.Module``) without leaking
those details across the project.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


class EmotionClassifier(abc.ABC):
    """Abstract base class for an emotion classifier in this project.

    Implementations must treat ``fit`` as the *only* place where
    learning happens: ``predict`` must be a pure function of already
    learned state (it must not look at labels, and must not re-fit).
    """

    @abc.abstractmethod
    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "EmotionClassifier":
        """Fit the model.

        Parameters
        ----------
        train_data:
            A dataframe with at least ``text`` and ``label`` columns,
            as produced by :mod:`emotion_ait.preprocessing` /
            :mod:`emotion_ait.sampling` (never validation/test rows).
        validation_data:
            Optional dataframe with the same schema, usable for early
            stopping / model selection. Must never be the test set.

        Returns
        -------
        EmotionClassifier
            ``self``, to allow method chaining.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, texts: Sequence[str]) -> List[int]:
        """Predict integer class ids for a sequence of raw texts.

        Must only be called after :meth:`fit`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Persist all learned state to ``path`` so that :meth:`load` can
        fully restore the model later (including for a fresh process)."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> "EmotionClassifier":
        """Restore a model previously saved with :meth:`save`."""
        raise NotImplementedError
