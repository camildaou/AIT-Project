"""Neural bigram language-model MDL classifier for the 27-class single-label track.

One small neural bigram language model per emotion ``e``::

    h_i = Embedding_e(w_{i-1})            (embedding_dim)
    z_i = W_e h_i + b_e                   (V logits)
    P_theta(w_i | w_{i-1}, e) = softmax(z_i)

trained from scratch, independently for each class, on that class's training
texts only (cross-entropy over (previous token, next token) pairs). Prediction
is the **conditional** description-length rule

    L(x | M_e) = -(1 / ln 2) * sum_i ln P_theta(w_i | w_{i-1}, e)     (bits, summed over tokens)
    e_hat      = argmin_e L(x | M_e)

This is not two-part MDL: the parameters of the neural models are not encoded.
An optional ``use_prior`` variant adds ``-log2 P(e)`` and is kept separate.

Conventions shared with the count-based MDL (:mod:`emotion_ait.mdl_classifier`):
the same tokenizer, one ``<s>`` before and one ``</s>`` after each text, one
shared vocabulary built from the current training subset only, and words
outside it mapped to ``<unk>``. Output tokens are the ``V`` prediction tokens
(training words + ``</s>`` + ``<unk>``; ``<s>`` is context only), so every
model's softmax is a proper distribution over the same ``V`` outcomes.

Validation texts of class ``e`` are used only for early stopping of model
``e`` (validation per-token NLL); test data is never seen during fitting.
A class with no training examples keeps an untrained model, is reported and
is never predicted (as in :class:`~emotion_ait.single_label_27.mdl.MDL27Classifier`);
a class without validation examples is trained for the full ``epochs``.
"""

from __future__ import annotations

import math
import pickle
import random
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..interfaces import EmotionClassifier
from ..mdl_classifier import _pad, _tokenize
from .data import CLASS_NAMES

_BOS, _EOS, _UNK = "<s>", "</s>", "<unk>"
_LN2 = math.log(2.0)


class NeuralBigramLM(nn.Module):
    """``P(w_i | w_{i-1}) = softmax(W Embedding(w_{i-1}) + b)``."""

    def __init__(self, input_size: int, output_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, output_size)

    def forward(self, previous: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(previous))

    def log_probabilities(self, previous: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.forward(previous), dim=-1)


class NeuralBigramMDL27(EmotionClassifier):
    """27-class conditional MDL classifier with neural bigram class models.

    Parameters
    ----------
    embedding_dim, learning_rate, batch_size, epochs, weight_decay, patience:
        Model/optimiser settings (Adam). Early stopping per class on the
        validation per-token NLL of that class; ``patience`` epochs without
        improvement stop training and the best epoch's weights are kept.
    seed:
        Seeds Python, NumPy and PyTorch (CUDA too); class ``c`` uses ``seed * 1000 + c``.
    device:
        ``"auto"`` (CUDA if available), ``"cpu"`` or ``"cuda"``. CPU runs are
        bit-reproducible; CUDA runs can differ in the last digits.
    use_prior:
        Separate variant: add ``-log2 P(e)`` (training frequency) to each score.
    """

    def __init__(
        self, embedding_dim: int = 128, learning_rate: float = 1e-3, batch_size: int = 128, epochs: int = 50,
        weight_decay: float = 0.0, patience: int = 2, seed: int = 42, device: str = "auto", use_prior: bool = False,
    ) -> None:
        for name, value in (("embedding_dim", embedding_dim), ("batch_size", batch_size), ("epochs", epochs)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}.")
        if learning_rate <= 0 or weight_decay < 0 or patience < 0:
            raise ValueError("learning_rate must be > 0, weight_decay >= 0 and patience >= 0.")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', got {device!r}.")
        self.embedding_dim = embedding_dim
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = float(weight_decay)
        self.patience = patience
        self.seed = int(seed)
        self.device = device
        self.use_prior = bool(use_prior)
        self.class_names: List[str] = list(CLASS_NAMES)
        self._token_to_id: Optional[Dict[str, int]] = None
        self._models: Optional[List[NeuralBigramLM]] = None
        self.class_priors: Optional[Dict[int, float]] = None
        self.missing_classes: List[int] = []
        self.training_metadata: Optional[Dict[str, object]] = None

    # --- vocabulary ----------------------------------------------------------

    def _resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def prediction_vocab_size(self) -> int:
        """``V``: training words + ``</s>`` + ``<unk>`` (the softmax outcomes)."""
        return len(self._token_to_id) - 1  # <s> is input-only

    @property
    def vocabulary(self) -> List[str]:
        """The shared fitted word vocabulary (sorted; excludes special tokens)."""
        if self._token_to_id is None:
            raise RuntimeError("NeuralBigramMDL27 used before fit().")
        return [t for t in self._token_to_id if t not in (_BOS, _EOS, _UNK)]

    def _transitions(self, text: object) -> Tuple[List[int], List[int]]:
        """(previous ids, next ids) for one text: ``<s> w_1 .. w_T </s>`` -> T + 1 pairs."""
        unk = self._token_to_id[_UNK]
        ids = [self._token_to_id.get(t, unk) for t in _pad(_tokenize(text), 2)]
        return ids[:-1], ids[1:]

    def _pairs(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous, following, owner = [], [], []
        for i, text in enumerate(texts):
            p, f = self._transitions(text)
            previous += p
            following += f
            owner += [i] * len(p)
        as_tensor = lambda x: torch.tensor(x, dtype=torch.long)  # noqa: E731
        return as_tensor(previous), as_tensor(following), as_tensor(owner)

    # --- fitting ---------------------------------------------------------------

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "NeuralBigramMDL27":
        if train_data.empty:
            raise ValueError("Cannot fit NeuralBigramMDL27 on an empty training set.")
        if "text" not in train_data.columns or "label" not in train_data.columns:
            raise KeyError("train_data must have 'text' and 'label' columns.")
        labels = [int(y) for y in train_data["label"]]
        invalid = sorted({y for y in labels if not 0 <= y < len(self.class_names)})
        if invalid:
            raise ValueError(f"Labels {invalid} are outside the 27-class range 0..{len(self.class_names) - 1}.")
        start = time.perf_counter()
        device = self._resolved_device()

        words = sorted({t for text in train_data["text"] for t in _tokenize(text)})
        # Ids 0..V-1 are the prediction tokens (softmax outcomes); <s> is the last, input-only id.
        self._token_to_id = {t: i for i, t in enumerate(words + [_EOS, _UNK, _BOS])}
        v = self.prediction_vocab_size

        counts = Counter(labels)
        self.class_priors = {c: counts.get(c, 0) / len(labels) for c in range(len(self.class_names))}
        self.missing_classes = [c for c in range(len(self.class_names)) if counts.get(c, 0) == 0]
        if self.missing_classes:
            warnings.warn(f"NeuralBigramMDL27: no training examples for {[self.class_names[c] for c in self.missing_classes]}; "
                          "their models stay untrained and they are excluded from prediction.")

        texts_by_class = {c: [] for c in range(len(self.class_names))}
        for text, y in zip(train_data["text"], labels):
            texts_by_class[y].append(text)
        val_by_class = {c: [] for c in range(len(self.class_names))}
        if validation_data is not None and not validation_data.empty:
            for text, y in zip(validation_data["text"], validation_data["label"]):
                if 0 <= int(y) < len(self.class_names):
                    val_by_class[int(y)].append(text)

        self._models, per_class = [], {}
        for c, name in enumerate(self.class_names):
            class_seed = self.seed * 1000 + c
            self._seed_everything(class_seed)
            model = NeuralBigramLM(len(self._token_to_id), v, self.embedding_dim).to(device)
            info = self._train_class(model, texts_by_class[c], val_by_class[c], device, class_seed)
            info.update(train_examples=len(texts_by_class[c]), validation_examples=len(val_by_class[c]))
            per_class[name] = info
            self._models.append(model.eval())

        self.training_metadata = {
            "model": "NeuralBigramMDL27",
            "embedding_dim": self.embedding_dim, "learning_rate": self.learning_rate, "batch_size": self.batch_size,
            "max_epochs": self.epochs, "weight_decay": self.weight_decay, "patience": self.patience,
            "seed": self.seed, "device": str(device), "use_prior": self.use_prior,
            "vocabulary_size": len(words), "prediction_vocabulary_size": v,
            "parameters_per_model": sum(p.numel() for p in self._models[0].parameters()),
            "parameters_total": sum(p.numel() for m in self._models for p in m.parameters()),
            "training_time_seconds": time.perf_counter() - start,
            "missing_classes": [self.class_names[c] for c in self.missing_classes],
            "per_class": per_class,
        }
        return self

    def _train_class(self, model: NeuralBigramLM, texts: List[str], val_texts: List[str], device: torch.device,
                     seed: int) -> Dict[str, object]:
        start = time.perf_counter()
        info: Dict[str, object] = {"train_tokens": 0, "validation_tokens": 0, "epochs_trained": 0, "best_epoch": None,
                                   "best_validation_bits_per_token": None, "train_loss_bits_per_token": [],
                                   "validation_bits_per_token": [], "stopped_early": False}
        if not texts:
            info["training_time_seconds"] = 0.0
            return info
        previous, following, _ = self._pairs(texts)
        info["train_tokens"] = int(previous.numel())
        val_previous, val_following, _ = self._pairs(val_texts) if val_texts else (None, None, None)
        info["validation_tokens"] = int(val_previous.numel()) if val_previous is not None else 0
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        generator = torch.Generator().manual_seed(seed)
        best_state, best_val, stale = None, math.inf, 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            order = torch.randperm(previous.numel(), generator=generator)
            total = 0.0
            for i in range(0, order.numel(), self.batch_size):
                batch = order[i:i + self.batch_size]
                x, y = previous[batch].to(device), following[batch].to(device)
                optimiser.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                (loss / batch.numel()).backward()
                optimiser.step()
                total += float(loss.detach())
            info["train_loss_bits_per_token"].append(total / previous.numel() / _LN2)
            info["epochs_trained"] = epoch
            if val_previous is None:
                continue
            val_bits = float(self._nll_nats(model, val_previous, val_following, device).sum()) / val_previous.numel() / _LN2
            info["validation_bits_per_token"].append(val_bits)
            if val_bits < best_val:
                best_val, stale, info["best_epoch"] = val_bits, 0, epoch
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                stale += 1
                if stale > self.patience:
                    info["stopped_early"] = True
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
            info["best_validation_bits_per_token"] = best_val
        else:
            info["best_epoch"] = info["epochs_trained"]
        info["training_time_seconds"] = time.perf_counter() - start
        return info

    # --- scoring -----------------------------------------------------------------

    @torch.no_grad()
    def _nll_nats(self, model: NeuralBigramLM, previous: torch.Tensor, following: torch.Tensor,
                  device: torch.device, chunk: int = 2048) -> torch.Tensor:
        """Per-transition ``-ln P(following | previous)``. The softmax normaliser is
        computed once per distinct previous token, then gathered."""
        model.eval()
        contexts, inverse = torch.unique(previous, return_inverse=True)
        log_z = torch.empty(contexts.numel(), device=device)
        for i in range(0, contexts.numel(), chunk):
            log_z[i:i + chunk] = torch.logsumexp(model(contexts[i:i + chunk].to(device)), dim=-1)
        h = model.embedding(previous.to(device))
        w = model.output.weight[following.to(device)]
        target_logits = (h * w).sum(-1) + model.output.bias[following.to(device)]
        return (log_z[inverse.to(device)] - target_logits).float()

    def class_description_lengths(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), 27)`` matrix of ``L(x | M_e)`` in bits (always finite)."""
        if self._models is None:
            raise RuntimeError("NeuralBigramMDL27 used before fit().")
        device = self._resolved_device()
        previous, following, owner = self._pairs(texts)
        bits = np.zeros((len(texts), len(self.class_names)))
        for c, model in enumerate(self._models):
            nll = self._nll_nats(model, previous, following, device).cpu().double()
            per_text = torch.zeros(len(texts), dtype=torch.double).index_add_(0, owner, nll)
            bits[:, c] = (per_text / _LN2).numpy()
        return bits

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        """Decision scores (bits, lower is better): ``L(x|M_e)`` (+ ``-log2 P(e)`` with
        ``use_prior``); ``+inf`` for classes absent from training."""
        scores = self.class_description_lengths(texts)
        if self.use_prior:
            scores = scores + np.array([-math.log2(p) if p > 0 else math.inf for p in self.class_priors.values()])
        if self.missing_classes:
            scores[:, self.missing_classes] = math.inf
        return scores

    def predict(self, texts: Sequence[str]) -> List[int]:
        """``argmin_e`` score; ties -> lowest class index."""
        if not len(texts):
            return []
        return [int(i) for i in np.argmin(self.predict_scores(texts), axis=1)]

    # --- diagnostics / persistence -----------------------------------------------------

    def diagnostics(self, train_data: pd.DataFrame, held_out: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, object]:
        """Training metadata plus, per held-out split, total/mean NLL of the predicted
        class, OOV counts and predictions per class."""
        report = dict(self.training_metadata or {})
        unk = self._token_to_id[_UNK]
        for split, texts in (held_out or {}).items():
            scores = self.predict_scores(texts)
            predictions = np.argmin(scores, axis=1)
            chosen = scores[np.arange(len(texts)), predictions]
            _, following, _ = self._pairs(texts)
            counts = Counter(int(p) for p in predictions)
            report[split] = {
                "texts": len(texts), "tokens": int(following.numel()),
                "oov_tokens": int((following == unk).sum()),
                "total_bits_predicted_class": float(chosen.sum()),
                "mean_bits_per_token_predicted_class": float(chosen.sum() / following.numel()),
                "predictions_per_class": {name: counts.get(c, 0) for c, name in enumerate(self.class_names)},
            }
        return report

    def save(self, path: Path) -> None:
        if self._models is None:
            raise RuntimeError("NeuralBigramMDL27.save called before fit().")
        state = {k: v for k, v in self.__dict__.items() if k != "_models"}
        state["_model_states"] = [{k: t.cpu() for k, t in m.state_dict().items()} for m in self._models]
        with open(path, "wb") as fh:
            pickle.dump({"class": type(self).__name__, "state": state}, fh)

    @classmethod
    def load(cls, path: Path) -> "NeuralBigramMDL27":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("class") != cls.__name__:
            raise ValueError(f"{path} holds a {payload.get('class')!r}, not a {cls.__name__!r}.")
        state = dict(payload["state"])
        model_states = state.pop("_model_states")
        model = cls.__new__(cls)
        model.__dict__.update(state)
        device = model._resolved_device()
        model._models = []
        for weights in model_states:
            lm = NeuralBigramLM(len(model._token_to_id), model.prediction_vocab_size, model.embedding_dim)
            lm.load_state_dict(weights)
            model._models.append(lm.to(device).eval())
        return model
