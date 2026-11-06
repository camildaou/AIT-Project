"""Generative bigram Naive Bayes classifier for the 27-class single-label track.

An independent model (it does not subclass or modify any MDL classifier); it
only reuses the low-level helpers of :mod:`emotion_ait.mdl_classifier`
(tokenizer, BOS/EOS padding, n-gram extraction and the add-``k`` formula), so
that tokenization, vocabulary, ``<s>``/``</s>``/``<unk>`` handling, context
construction and smoothing are identical to the conditional MDL classifier.

Model
-----
For each of the 27 emotions ``e`` (class order of
:data:`emotion_ait.single_label_27.data.CLASS_NAMES`) a class-conditional
bigram language model is estimated from that class's training texts::

    log P(x | e) = sum_i log P(w_i | w_{i-1}, e)
    P(w_i | w_{i-1}, e) = (count_e(w_{i-1}, w_i) + k) / (count_e(w_{i-1}) + k V)

with ``V`` = distinct words of the training subset + ``</s>`` + ``<unk>``
(``<s>`` is context only), one shared vocabulary for all classes, and
out-of-vocabulary words mapped to ``<unk>``. ``k = 1`` by default.

Decision rules (``variant``):

* ``"ml"`` (maximum likelihood): ``argmax_e log P(x | e)``.
* ``"map"`` (maximum a posteriori, conventional Naive Bayes):
  ``argmax_e [log P(x | e) + log P(e)]`` with ``P(e) = N_e / N`` from the
  training labels only.

Relation to conditional MDL
---------------------------
The MDL description length is ``L(x | M_e) = -log2 P(x | e)``, so
``argmax_e log P(x | e) = argmin_e L(x | M_e)``: the ML variant is the
conditional MDL classifier, and the MAP variant is MDL with the class-prior
code ``-log2 P(e)``. Logarithms here are **base 2** (the base does not change
any argmax), and the per-token terms are summed in the same order as the MDL
implementation, so ``log2 P(x | e)`` is exactly ``-L(x | M_e)`` in floating
point and predictions -- including ties -- coincide.

Edge cases
----------
All 27 classes always have a model. A class with no training examples gets
an empty model (every token ``log2(1/V)``, finite) and is excluded from the
argmax (its decision score is ``-inf``), with a warning -- the same policy as
:class:`emotion_ait.single_label_27.mdl.MDL27Classifier`. Ties go to the lowest
class index. Empty texts are scored by their single ``</s>`` token.
"""

from __future__ import annotations

import math
import pickle
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..interfaces import EmotionClassifier
from ..mdl_classifier import _laplace_probability, _ngrams, _pad, _tokenize
from .data import CLASS_NAMES

_UNK = "<unk>"
VARIANTS = ("ml", "map")


class BigramNaiveBayes27(EmotionClassifier):
    """27-class generative n-gram Naive Bayes (see module docstring).

    Parameters
    ----------
    n:
        N-gram order (default 2, bigram).
    k:
        Additive smoothing constant (default 1.0, add-one).
    variant:
        ``"ml"`` (likelihood only) or ``"map"`` (likelihood + class prior).
    """

    def __init__(self, n: int = 2, k: float = 1.0, variant: str = "ml") -> None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be an int >= 1, got {n!r}.")
        if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k <= 0:
            raise ValueError(f"k must be a finite number > 0, got {k!r}.")
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}.")
        self.n = n
        self.k = float(k)
        self.variant = variant
        self.class_names: List[str] = list(CLASS_NAMES)
        self._word_vocab: Optional[set] = None
        self._prediction_vocab_size: Optional[int] = None
        self._ngram_counts: Optional[List[Counter]] = None
        self._context_counts: Optional[List[Counter]] = None
        self.class_counts: Optional[Dict[int, int]] = None
        self.class_priors: Optional[Dict[int, float]] = None
        self.missing_classes: List[int] = []

    # --- fitting -------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "BigramNaiveBayes27":
        # validation_data is accepted for interface compatibility and never used.
        if train_data.empty:
            raise ValueError("Cannot fit BigramNaiveBayes27 on an empty training set.")
        if "text" not in train_data.columns or "label" not in train_data.columns:
            raise KeyError("train_data must have 'text' and 'label' columns.")
        labels = [int(label) for label in train_data["label"]]
        invalid = sorted({y for y in labels if not 0 <= y < len(self.class_names)})
        if invalid:
            raise ValueError(f"Labels {invalid} are outside the 27-class range 0..{len(self.class_names) - 1}.")

        tokenized = [_tokenize(text) for text in train_data["text"]]
        self._word_vocab = {token for tokens in tokenized for token in tokens}
        self._prediction_vocab_size = len(self._word_vocab) + 2  # + </s> + <unk>; <s> is context only

        self._ngram_counts = [Counter() for _ in self.class_names]
        self._context_counts = [Counter() for _ in self.class_names]
        for tokens, label in zip(tokenized, labels):
            for gram in _ngrams(_pad(tokens, self.n), self.n):
                self._ngram_counts[label][gram] += 1
                self._context_counts[label][gram[:-1]] += 1

        counts = Counter(labels)
        self.class_counts = {c: counts.get(c, 0) for c in range(len(self.class_names))}
        self.class_priors = {c: count / len(labels) for c, count in self.class_counts.items()}
        self.missing_classes = [c for c, count in self.class_counts.items() if count == 0]
        if self.missing_classes:
            warnings.warn(
                f"BigramNaiveBayes27: no training examples for {[self.class_names[c] for c in self.missing_classes]}; "
                "they get empty (uniform) models and are excluded from prediction."
            )
        return self

    def _check_fitted(self) -> None:
        if self._ngram_counts is None:
            raise RuntimeError("BigramNaiveBayes27 used before fit().")

    @property
    def vocabulary(self) -> List[str]:
        """The shared fitted word vocabulary (sorted; excludes ``</s>``/``<unk>``)."""
        self._check_fitted()
        return sorted(self._word_vocab)

    @property
    def predictable_classes(self) -> List[int]:
        return [c for c in range(len(self.class_names)) if c not in self.missing_classes]

    # --- probabilities ---------------------------------------------------------

    def probability(self, context: tuple, word: str, class_id: int) -> float:
        """Smoothed ``P(word | context, e)`` (``context`` = previous ``n - 1`` tokens)."""
        self._check_fitted()
        return _laplace_probability(
            ngram_count=self._ngram_counts[class_id].get(tuple(context) + (word,), 0),
            context_count=self._context_counts[class_id].get(tuple(context), 0),
            k=self.k,
            prediction_vocab_size=self._prediction_vocab_size,
        )

    def _log_likelihood(self, tokens: List[str], class_id: int) -> float:
        normalized = [token if token in self._word_vocab else _UNK for token in tokens]
        total = 0.0
        for gram in _ngrams(_pad(normalized, self.n), self.n):
            total += math.log2(self.probability(gram[:-1], gram[-1], class_id))
        return total

    def class_log_likelihoods(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), 27)`` matrix of ``log2 P(x | e)`` (always finite)."""
        self._check_fitted()
        rows = []
        for text in texts:
            tokens = _tokenize(text)
            rows.append([self._log_likelihood(tokens, c) for c in range(len(self.class_names))])
        return np.array(rows, dtype=float).reshape(len(texts), len(self.class_names))

    def log_priors(self) -> np.ndarray:
        """``(27,)`` vector of ``log2 P(e)`` (``-inf`` for classes absent from training)."""
        self._check_fitted()
        return np.array([math.log2(p) if p > 0 else -math.inf for p in (self.class_priors[c] for c in range(len(self.class_names)))])

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), 27)`` decision scores: ``log2 P(x|e)`` (ML) or
        ``log2 P(x|e) + log2 P(e)`` (MAP); ``-inf`` for classes absent from training."""
        scores = self.class_log_likelihoods(texts)
        if self.variant == "map":
            scores = scores + self.log_priors()
        if self.missing_classes:
            scores[:, self.missing_classes] = -math.inf
        return scores

    def predict(self, texts: Sequence[str]) -> List[int]:
        """``argmax_e`` of :meth:`predict_scores`; ties -> lowest class index."""
        scores = self.predict_scores(texts)
        eligible = self.predictable_classes
        predictions = []
        for row in scores:
            best = eligible[0]
            for c in eligible[1:]:
                if row[c] > row[best]:
                    best = c
            predictions.append(best)
        return predictions

    # --- persistence -----------------------------------------------------------

    def save(self, path: Path) -> None:
        self._check_fitted()
        with open(path, "wb") as fh:
            pickle.dump({"class": type(self).__name__, "state": self.__dict__}, fh)

    @classmethod
    def load(cls, path: Path) -> "BigramNaiveBayes27":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("class") != cls.__name__:
            raise ValueError(f"{path} holds a {payload.get('class')!r}, not a {cls.__name__!r}.")
        model = cls.__new__(cls)
        model.__dict__.update(payload["state"])
        return model
