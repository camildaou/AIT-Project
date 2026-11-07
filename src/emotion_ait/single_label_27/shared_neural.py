"""Shared class-conditioned neural bigram MDL classifier (27-class track).

One neural bigram language model for all emotions, conditioned on the class::

    h   = tanh(A [Emb_word(w_{i-1}) ; Emb_class(e)] + a)
    P_theta(w_i | w_{i-1}, e) = softmax(W h + b + b_e)

Word embeddings, the hidden layer and the output layer are shared by all 27
classes, so every class benefits from the whole training subset; each class
only owns a small class embedding (which changes how the previous word is
read, through the nonlinearity) and an output bias ``b_e`` (its word-usage
offsets). Training maximises the joint likelihood of every training text
under its own class (cross-entropy over all (previous, next, class)
transitions); early stopping uses the validation per-token NLL of every
validation text under its own class. Test data is never used in fitting.

Prediction is the conditional description-length rule of the other MDL
models, ``argmin_e L(x | e)`` with ``L(x | e) = -(1/ln 2) sum_i ln P_theta(w_i | w_{i-1}, e)``,
scored by the same network for all 27 values of ``e`` -- so the 27 code
lengths are directly comparable (unlike 27 independently trained networks).
It is conditional MDL only: network parameters are not encoded.

Tokenization, ``<s>``/``</s>``/``<unk>``, the training-only shared vocabulary
and the ``V`` softmax outcomes are identical to
:class:`~emotion_ait.single_label_27.neural_bigram.NeuralBigramMDL27`, whose
vocabulary, seeding, prior variant, prediction and diagnostics code is reused.
"""

from __future__ import annotations

import math
import pickle
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..mdl_classifier import _tokenize
from .neural_bigram import _BOS, _EOS, _LN2, _UNK, NeuralBigramMDL27


class ClassConditionedBigramLM(nn.Module):
    """``softmax(W tanh(A [Emb_word(prev); Emb_class(e)] + a) + b + b_e)``."""

    def __init__(self, input_size: int, output_size: int, num_classes: int,
                 embedding_dim: int, class_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(input_size, embedding_dim)
        self.class_embedding = nn.Embedding(num_classes, class_dim)
        self.hidden = nn.Linear(embedding_dim + class_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_size)
        self.class_bias = nn.Embedding(num_classes, output_size)
        nn.init.zeros_(self.class_bias.weight)

    def hidden_state(self, previous: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hidden(torch.cat([self.word_embedding(previous), self.class_embedding(classes)], dim=-1)))

    def forward(self, previous: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden_state(previous, classes)) + self.class_bias(classes)


class SharedNeuralBigramMDL27(NeuralBigramMDL27):
    """27-class conditional MDL with one shared class-conditioned neural bigram LM.

    Parameters as :class:`NeuralBigramMDL27`, plus ``class_dim`` (class
    embedding size, default 32) and ``hidden_dim`` (default 128).
    """

    def __init__(self, embedding_dim: int = 128, class_dim: int = 32, hidden_dim: int = 128,
                 learning_rate: float = 1e-3, batch_size: int = 128, epochs: int = 50, weight_decay: float = 0.0,
                 patience: int = 2, seed: int = 42, device: str = "auto", use_prior: bool = False) -> None:
        super().__init__(embedding_dim=embedding_dim, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs,
                         weight_decay=weight_decay, patience=patience, seed=seed, device=device, use_prior=use_prior)
        for name, value in (("class_dim", class_dim), ("hidden_dim", hidden_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}.")
        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self._model: Optional[ClassConditionedBigramLM] = None

    # --- data --------------------------------------------------------------------

    def _labelled_pairs(self, texts: Sequence[str], labels: Sequence[int]):
        previous, following, owner = self._pairs(texts)
        classes = torch.tensor(labels, dtype=torch.long)[owner] if len(labels) else torch.zeros(0, dtype=torch.long)
        return previous, following, classes

    # --- fitting ---------------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "SharedNeuralBigramMDL27":
        if train_data.empty:
            raise ValueError("Cannot fit SharedNeuralBigramMDL27 on an empty training set.")
        if "text" not in train_data.columns or "label" not in train_data.columns:
            raise KeyError("train_data must have 'text' and 'label' columns.")
        labels = [int(y) for y in train_data["label"]]
        invalid = sorted({y for y in labels if not 0 <= y < len(self.class_names)})
        if invalid:
            raise ValueError(f"Labels {invalid} are outside the 27-class range 0..{len(self.class_names) - 1}.")
        start = time.perf_counter()
        device = self._resolved_device()
        self._seed_everything(self.seed)

        words = sorted({t for text in train_data["text"] for t in _tokenize(text)})
        self._token_to_id = {t: i for i, t in enumerate(words + [_EOS, _UNK, _BOS])}
        counts = Counter(labels)
        self.class_priors = {c: counts.get(c, 0) / len(labels) for c in range(len(self.class_names))}
        self.missing_classes = [c for c in range(len(self.class_names)) if counts.get(c, 0) == 0]
        if self.missing_classes:
            warnings.warn(f"SharedNeuralBigramMDL27: no training examples for {[self.class_names[c] for c in self.missing_classes]}; "
                          "they are excluded from prediction.")

        previous, following, classes = self._labelled_pairs(list(train_data["text"]), labels)
        val = None
        if validation_data is not None and not validation_data.empty:
            keep = [0 <= int(y) < len(self.class_names) for y in validation_data["label"]]
            vt = [t for t, k in zip(validation_data["text"], keep) if k]
            vy = [int(y) for y, k in zip(validation_data["label"], keep) if k]
            val = self._labelled_pairs(vt, vy) if vt else None

        model = ClassConditionedBigramLM(len(self._token_to_id), self.prediction_vocab_size, len(self.class_names),
                                         self.embedding_dim, self.class_dim, self.hidden_dim).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        generator = torch.Generator().manual_seed(self.seed)
        history = {"train_loss_bits_per_token": [], "validation_bits_per_token": []}
        best_state, best_val, best_epoch, stale, stopped_early, epochs_trained = None, math.inf, None, 0, False, 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            order = torch.randperm(previous.numel(), generator=generator)
            total = 0.0
            for i in range(0, order.numel(), self.batch_size):
                batch = order[i:i + self.batch_size]
                logits = model(previous[batch].to(device), classes[batch].to(device))
                loss = loss_fn(logits, following[batch].to(device))
                optimiser.zero_grad(set_to_none=True)
                (loss / batch.numel()).backward()
                optimiser.step()
                total += float(loss.detach())
            epochs_trained = epoch
            history["train_loss_bits_per_token"].append(total / previous.numel() / _LN2)
            if val is None:
                continue
            val_bits = float(self._nll_nats_shared(model, *val, device).sum()) / val[0].numel() / _LN2
            history["validation_bits_per_token"].append(val_bits)
            if val_bits < best_val:
                best_val, best_epoch, stale = val_bits, epoch, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                stale += 1
                if stale > self.patience:
                    stopped_early = True
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model.eval()

        per_class = {}
        val_nll = self._nll_nats_shared(model, *val, device).cpu() if val is not None else None
        for c, name in enumerate(self.class_names):
            in_class = classes == c
            entry = {"train_examples": counts.get(c, 0), "train_tokens": int(in_class.sum())}
            if val is not None:
                mask = val[2] == c
                entry["validation_tokens"] = int(mask.sum())
                entry["validation_bits_per_token"] = float(val_nll[mask].sum() / mask.sum() / _LN2) if mask.any() else None
            per_class[name] = entry
        self.training_metadata = {
            "model": "SharedNeuralBigramMDL27",
            "embedding_dim": self.embedding_dim, "class_dim": self.class_dim, "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate, "batch_size": self.batch_size, "max_epochs": self.epochs,
            "weight_decay": self.weight_decay, "patience": self.patience, "seed": self.seed, "device": str(device),
            "use_prior": self.use_prior, "vocabulary_size": len(words), "prediction_vocabulary_size": self.prediction_vocab_size,
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_class_specific": int(model.class_embedding.weight.numel() + model.class_bias.weight.numel()),
            "train_tokens": int(previous.numel()), "epochs_trained": epochs_trained, "best_epoch": best_epoch or epochs_trained,
            "best_validation_bits_per_token": None if best_state is None else best_val, "stopped_early": stopped_early,
            "training_time_seconds": time.perf_counter() - start,
            "missing_classes": [self.class_names[c] for c in self.missing_classes],
            "per_class": per_class, **history,
        }
        return self

    # --- scoring -----------------------------------------------------------------------

    @torch.no_grad()
    def _nll_nats_shared(self, model: ClassConditionedBigramLM, previous: torch.Tensor, following: torch.Tensor,
                         classes: torch.Tensor, device: torch.device, chunk: int = 4096) -> torch.Tensor:
        """Per-transition ``-ln P(following | previous, class)``; the softmax normaliser
        is computed once per distinct (previous, class) pair."""
        model.eval()
        keys = previous * len(self.class_names) + classes
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        u_prev, u_cls = unique_keys // len(self.class_names), unique_keys % len(self.class_names)
        log_z = torch.empty(unique_keys.numel(), device=device)
        for i in range(0, unique_keys.numel(), chunk):
            log_z[i:i + chunk] = torch.logsumexp(model(u_prev[i:i + chunk].to(device), u_cls[i:i + chunk].to(device)), dim=-1)
        nll = torch.empty(previous.numel(), device=device)
        for i in range(0, previous.numel(), chunk):
            p, f, c = previous[i:i + chunk].to(device), following[i:i + chunk].to(device), classes[i:i + chunk].to(device)
            h = model.hidden_state(p, c)
            target = (h * model.output.weight[f]).sum(-1) + model.output.bias[f] + model.class_bias.weight[c, f]
            nll[i:i + chunk] = log_z[inverse[i:i + chunk].to(device)] - target
        return nll

    def class_description_lengths(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), 27)`` matrix of ``L(x | e)`` in bits, all from the one shared model."""
        if self._model is None:
            raise RuntimeError("SharedNeuralBigramMDL27 used before fit().")
        device = self._resolved_device()
        previous, following, owner = self._pairs(texts)
        bits = np.zeros((len(texts), len(self.class_names)))
        for c in range(len(self.class_names)):
            classes = torch.full_like(previous, c)
            nll = self._nll_nats_shared(self._model, previous, following, classes, device).cpu().double()
            bits[:, c] = (torch.zeros(len(texts), dtype=torch.double).index_add_(0, owner, nll) / _LN2).numpy()
        return bits

    # --- persistence ---------------------------------------------------------------------

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("SharedNeuralBigramMDL27.save called before fit().")
        state = {k: v for k, v in self.__dict__.items() if k not in ("_model", "_models")}
        state["_model_state"] = {k: t.cpu() for k, t in self._model.state_dict().items()}
        with open(path, "wb") as fh:
            pickle.dump({"class": type(self).__name__, "state": state}, fh)

    @classmethod
    def load(cls, path: Path) -> "SharedNeuralBigramMDL27":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("class") != cls.__name__:
            raise ValueError(f"{path} holds a {payload.get('class')!r}, not a {cls.__name__!r}.")
        state = dict(payload["state"])
        weights = state.pop("_model_state")
        model = cls.__new__(cls)
        model.__dict__.update(state)
        lm = ClassConditionedBigramLM(len(model._token_to_id), model.prediction_vocab_size, len(model.class_names),
                                      model.embedding_dim, model.class_dim, model.hidden_dim)
        lm.load_state_dict(weights)
        model._model = lm.to(model._resolved_device()).eval()
        model._models = None
        return model
