"""A small, CPU-friendly supervised LSTM baseline for GoEmotions."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from .interfaces import EmotionClassifier

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence
except ImportError:  # pragma: no cover - torch is an expected dependency
    torch = None
    nn = None
    pack_padded_sequence = None


_PAD, _UNK = "<PAD>", "<UNK>"


if nn is not None:
    class _LSTMNetwork(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                     num_layers: int, num_classes: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.output = nn.Linear(hidden_dim, num_classes)

        def forward(self, tokens, lengths):
            embedded = self.embedding(tokens)
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
            return self.output(hidden[-1])
else:  # pragma: no cover
    _LSTMNetwork = None


class LSTMClassifier(EmotionClassifier):
    """Word-token LSTM with a training-only vocabulary and validation stopping.

    Labels follow the shared project mapping: anger=0, fear=1, joy=2,
    sadness=3. The output layer always has four logits, including when a
    very small training subset happens to omit a class.
    """

    def __init__(
        self, embedding_dim: int = 100, hidden_dim: int = 128,
        num_layers: int = 1, max_epochs: int = 10, learning_rate: float = 1e-3,
        batch_size: int = 32, device: str = "cpu", seed: int = 42,
        min_frequency: int = 1, patience: int = 3, max_vocabulary_size: Optional[int] = 30000,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.min_frequency = min_frequency
        self.patience = patience
        self.max_vocabulary_size = max_vocabulary_size
        self._model = None
        self._vocab = None

    @staticmethod
    def _tokens(text: str) -> List[str]:
        # Lightweight, dependency-free tokenization; lowercase normalization
        # is applied consistently at training and inference time.
        return str(text).lower().split()

    def _build_vocab(self, texts: Sequence[str]) -> dict:
        counts = Counter(token for text in texts for token in self._tokens(text))
        words = [word for word, count in counts.most_common() if count >= self.min_frequency]
        if self.max_vocabulary_size is not None:
            words = words[: max(0, self.max_vocabulary_size - 2)]
        return {_PAD: 0, _UNK: 1, **{word: i + 2 for i, word in enumerate(words)}}

    def _encode(self, text: str) -> List[int]:
        assert self._vocab is not None
        ids = [self._vocab.get(token, 1) for token in self._tokens(text)]
        return ids or [1]  # LSTM packing requires sequence length >= 1.

    @staticmethod
    def _batch(items, device):
        lengths = torch.tensor([len(tokens) for tokens, _ in items], dtype=torch.long)
        width = int(lengths.max().item())
        x = torch.zeros((len(items), width), dtype=torch.long)
        y = torch.tensor([label for _, label in items], dtype=torch.long)
        for i, (tokens, _) in enumerate(items):
            x[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return x.to(device), lengths, y.to(device)

    def _dataset(self, frame: pd.DataFrame):
        return [(self._encode(text), int(label)) for text, label in zip(frame.text, frame.label)]

    def _predict_frame(self, frame: pd.DataFrame) -> List[int]:
        data = self._dataset(frame)
        predictions = []
        self._model.eval()
        with torch.no_grad():
            for start in range(0, len(data), self.batch_size):
                x, lengths, _ = self._batch(data[start:start + self.batch_size], self.device)
                predictions.extend(self._model(x, lengths).argmax(dim=1).cpu().tolist())
        return predictions

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "LSTMClassifier":
        if torch is None:
            raise ImportError("PyTorch is required for LSTMClassifier; install the project requirements.")
        required = {"text", "label"}
        if not required.issubset(train_data.columns):
            raise ValueError("train_data must contain 'text' and 'label' columns.")
        if train_data.empty:
            raise ValueError("Cannot fit LSTMClassifier on an empty dataframe.")
        if self.batch_size < 1 or self.max_epochs < 1 or self.min_frequency < 1:
            raise ValueError("batch_size, max_epochs, and min_frequency must be positive.")
        if self.num_layers < 1 or self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("LSTM dimensions and num_layers must be positive.")
        labels = train_data.label.astype(int)
        if not labels.between(0, 3).all():
            raise ValueError("Labels must use the project class ids 0 through 3.")
        if validation_data is not None and not required.issubset(validation_data.columns):
            raise ValueError("validation_data must contain 'text' and 'label' columns.")

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
        self._vocab = self._build_vocab(train_data.text.astype(str).tolist())
        self._model = _LSTMNetwork(
            len(self._vocab), self.embedding_dim, self.hidden_dim, self.num_layers, 4
        ).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        training = self._dataset(train_data)
        validation = self._dataset(validation_data) if validation_data is not None and not validation_data.empty else None
        best_state, best_score, stale_epochs = None, float("-inf"), 0

        for _epoch in range(self.max_epochs):
            self._model.train()
            order = list(range(len(training)))
            random.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                batch_items = [training[i] for i in order[start:start + self.batch_size]]
                x, lengths, y = self._batch(batch_items, self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self._model(x, lengths), y)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
                optimizer.step()

            if validation is not None:
                from sklearn.metrics import f1_score
                val_pred = self._predict_frame(validation_data)
                score = f1_score(validation_data.label.astype(int), val_pred,
                                 labels=[0, 1, 2, 3], average="macro", zero_division=0)
            else:
                # Without validation, retain the final epoch; no test data is
                # consulted for model selection.
                continue
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone()
                              for key, value in self._model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if self.patience >= 0 and stale_epochs >= self.patience + 1:
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, texts: Sequence[str]) -> List[int]:
        if torch is None:
            raise ImportError("PyTorch is required for LSTMClassifier; install the project requirements.")
        if self._model is None or self._vocab is None:
            raise RuntimeError("Call fit() or load() before predict().")
        if not texts:
            return []
        frame = pd.DataFrame({"text": list(texts), "label": [0] * len(texts)})
        return self._predict_frame(frame)

    def save(self, path: Path) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for LSTMClassifier; install the project requirements.")
        if self._model is None or self._vocab is None:
            raise RuntimeError("Cannot save an unfitted LSTMClassifier.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._model.state_dict(), "vocab": self._vocab,
            "params": {key: getattr(self, key) for key in (
                "embedding_dim", "hidden_dim", "num_layers", "max_epochs",
                "learning_rate", "batch_size", "device", "seed", "min_frequency",
                "patience", "max_vocabulary_size")},
        }, path)

    @classmethod
    def load(cls, path: Path) -> "LSTMClassifier":
        if torch is None:
            raise ImportError("PyTorch is required for LSTMClassifier; install the project requirements.")
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only was introduced
            payload = torch.load(Path(path), map_location="cpu")
        instance = cls(**payload["params"])
        instance._vocab = payload["vocab"]
        instance._model = _LSTMNetwork(
            len(instance._vocab), instance.embedding_dim, instance.hidden_dim,
            instance.num_layers, 4,
        )
        instance._model.load_state_dict(payload["state_dict"])
        instance._model.to(instance.device)
        instance._model.eval()
        return instance
