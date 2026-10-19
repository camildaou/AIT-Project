"""Stub for the LSTM-based emotion classifier.

Owned by: TIA Hekmeh.

TODO (not implemented yet, by design -- see project spec section 5):
    * Tokenize and numericalize ``text`` (vocabulary construction
      strategy TBD by Person B -- e.g. a project-specific vocabulary
      built from the training subset only, to avoid leakage, or a
      pretrained tokenizer).
    * Build a PyTorch ``nn.Module`` with an embedding layer, one or
      more LSTM layers, and a linear classification head over the 4
      target classes.
    * Implement a standard supervised training loop (loss, optimizer,
      optional early stopping against ``validation_data``). Must not
      assume GPU availability -- default to CPU-friendly settings.
    * ``save``/``load`` must persist both the model weights and the
      vocabulary/tokenizer state needed to run inference later.

This module intentionally does not fabricate predictions or pretend
that training has occurred: all public methods raise
``NotImplementedError`` until Person B implements them.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from .interfaces import EmotionClassifier

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch is an expected dependency
    torch = None
    nn = None


class LSTMClassifier(EmotionClassifier):
    """LSTM-based emotion classifier (stub).

    Parameters
    ----------
    embedding_dim, hidden_dim, num_layers:
        Placeholder architecture hyperparameters; Person B should tune
        these.
    max_epochs, learning_rate, batch_size:
        Placeholder training hyperparameters.
    device:
        Defaults to "cpu"; this project assumes no GPU access.
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        max_epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        # TODO(person-b): the fitted nn.Module and the fitted
        # vocabulary/tokenizer used to numericalize raw text.
        self._model = None
        self._vocab = None

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "LSTMClassifier":
        # TODO(person-b): build the vocabulary from `train_data.text`,
        # construct the nn.Module, and run the training loop, optionally
        # using `validation_data` for early stopping / model selection.
        raise NotImplementedError(
            "LSTMClassifier.fit is not implemented yet. See the module docstring TODOs."
        )

    def predict(self, texts: Sequence[str]) -> List[int]:
        # TODO(person-b): numericalize `texts` with the fitted vocabulary
        # and run a forward pass through the fitted model.
        raise NotImplementedError(
            "LSTMClassifier.predict is not implemented yet. See the module docstring TODOs."
        )

    def save(self, path: Path) -> None:
        # TODO(person-b): persist model weights (`state_dict`) and the
        # vocabulary/tokenizer to `path`.
        raise NotImplementedError("LSTMClassifier.save is not implemented yet.")

    @classmethod
    def load(cls, path: Path) -> "LSTMClassifier":
        # TODO(person-b): restore model weights and vocabulary from `path`.
        raise NotImplementedError("LSTMClassifier.load is not implemented yet.")
