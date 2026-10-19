"""The MDL (Minimum Description Length) emotion classifier.

Owned by: Camil DAOU.

Approach
--------
For each of the 4 target emotions we fit one smoothed n-gram (default:
trigram, ``n=3``) word-level language model on the training texts
belonging to that class only. At prediction time, an input text is
scored under every per-class language model as a description length
(the total number of bits needed to encode it, i.e. the summed
negative log2-probability of its n-grams), and the predicted class is
the one whose model assigns the *shortest* description length -- the
class under which the text is "cheapest" to encode (the MDL
principle).

Smoothing
---------
Only additive ("laplace" / add-``k``) smoothing is implemented so far::

    P(w_i | context) = (count(context, w_i) + k) / (count(context) + k * V)

where ``V`` is the size of the shared **prediction vocabulary**: every
distinct word observed in the *current training subset* (across all
classes, so every per-class model is smoothed against the same
denominator), plus the end-of-sequence marker ``</s>`` and the
out-of-vocabulary marker ``<unk>``. The beginning-of-sequence marker
``<s>`` is deliberately **excluded** from ``V`` -- it is only ever used
as left-padding context, never as a token a model predicts, so counting
it in ``V`` would make the per-context probabilities sum to less than
one. With ``<s>`` excluded, for any fixed context the probabilities
``P(w | context)`` over all ``w`` in the prediction vocabulary are each
strictly positive and sum to exactly one (see
``tests/test_mdl_classifier.py`` for a direct numerical check), for any
n-gram order (unigram, bigram, trigram, ...). ``k`` defaults to ``1.0``
(classic add-one / Laplace smoothing) and must be finite and strictly
positive. Any other ``smoothing`` value raises ``NotImplementedError``
for now.

Tokenization
------------
Simple, deterministic word tokenization: lowercase, then extract
maximal runs of letters/digits/apostrophes. Tokens are padded with
``n - 1`` beginning-of-sequence markers and a single end-of-sequence
marker before counting/scoring, as is standard for n-gram language
models, so the first and last words of a comment are modeled too.
Out-of-vocabulary tokens (seen at prediction time only, i.e. words not
present anywhere in the training subset) are mapped to a shared
``<unk>`` token. Empty strings and very short texts are handled without
error: they simply produce fewer n-grams (down to the single
``(..., </s>)`` n-gram for an empty string), each still scored with a
strictly positive probability thanks to smoothing.

Model-complexity / two-part MDL
--------------------------------
This classifier implements only the *conditional* description length
term ``L(x | M_e)`` of the two-part MDL objective ``L(M) +
sum_x L(x | M)`` -- it does **not** add a model-description-length
term ``L(M_e)`` for the n-gram tables themselves. See
``docs/mdl_model_complexity.md`` for the investigation into how (and
whether) such a term should be defined and used; that investigation is
intentionally kept out of this module so it cannot change baseline
predictions.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .interfaces import EmotionClassifier

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_BOS = "<s>"
_EOS = "</s>"
_UNK = "<unk>"

_PerClassModel = Dict[str, "Counter[Tuple[str, ...]]"]


def _tokenize(text: object) -> List[str]:
    """Lowercase word tokenization: maximal runs of letters/digits/apostrophes."""
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


def _pad(tokens: List[str], n: int) -> List[str]:
    """Pad a token sequence with ``n - 1`` BOS markers and one EOS marker."""
    return [_BOS] * (n - 1) + tokens + [_EOS]


def _ngrams(padded_tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(padded_tokens[i : i + n]) for i in range(len(padded_tokens) - n + 1)]


def _laplace_probability(ngram_count: int, context_count: int, k: float, prediction_vocab_size: int) -> float:
    """Additive (Laplace / add-``k``) smoothed conditional probability.

    ``prediction_vocab_size`` must be the size of the vocabulary of
    *predictable* tokens only (i.e. excluding ``<s>``, see the module
    docstring) for the resulting probabilities to sum to one over that
    vocabulary for any fixed context.
    """
    return (ngram_count + k) / (context_count + k * prediction_vocab_size)


class MDLClassifier(EmotionClassifier):
    """Minimum Description Length emotion classifier.

    Parameters
    ----------
    n:
        N-gram order to use for the per-class language models (default
        ``3``, i.e. trigrams). Must be an ``int >= 1``.
    smoothing:
        Name of the smoothing scheme to apply. Only ``"laplace"``
        (additive / add-``k`` smoothing, see module docstring) is
        implemented so far.
    k:
        Additive smoothing constant used by the ``"laplace"`` scheme
        (``k=1.0`` is classic Laplace / add-one smoothing). Must be
        finite and strictly positive.
    """

    def __init__(self, n: int = 3, smoothing: str = "laplace", k: float = 1.0) -> None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n-gram order 'n' must be an int, got {type(n).__name__} ({n!r}).")
        if n < 1:
            raise ValueError(f"n-gram order 'n' must be >= 1, got {n}.")
        if smoothing != "laplace":
            raise NotImplementedError(
                f"Smoothing scheme '{smoothing}' is not implemented yet; only 'laplace' is supported."
            )
        if not isinstance(k, (int, float)) or isinstance(k, bool):
            raise TypeError(f"Smoothing constant 'k' must be a real number, got {type(k).__name__} ({k!r}).")
        if not math.isfinite(k) or k <= 0:
            raise ValueError(f"Smoothing constant 'k' must be finite and strictly positive, got {k}.")
        self.n = n
        self.smoothing = smoothing
        self.k = float(k)
        # One fitted per-class language model per target emotion, keyed by
        # integer label id: {label_id: {"ngram_counts": Counter, "context_counts": Counter}}.
        self._per_class_models: Optional[Dict[int, _PerClassModel]] = None
        # Distinct real words observed anywhere in the training subset
        # (never includes <s>/</s>/<unk>; see `_prediction_vocab_size`).
        self._word_vocab: Optional[Set[str]] = None
        # len(self._word_vocab) + 2 (for </s> and <unk>); <s> is a
        # context-only marker and is deliberately excluded, see the
        # module docstring's "Smoothing" section.
        self._prediction_vocab_size: Optional[int] = None
        self._classes: Optional[List[int]] = None

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "MDLClassifier":
        # validation_data is accepted for interface compatibility but is
        # currently unused -- count-based n-gram models do not need early
        # stopping (unlike the LSTM); `k` could be tuned against it later.
        if train_data.empty:
            raise ValueError("Cannot fit MDLClassifier on an empty training set.")
        if "text" not in train_data.columns or "label" not in train_data.columns:
            raise KeyError("train_data must have 'text' and 'label' columns.")

        tokenized = [_tokenize(t) for t in train_data["text"].tolist()]
        labels = train_data["label"].tolist()

        word_vocab: Set[str] = set()
        for tokens in tokenized:
            word_vocab.update(tokens)
        self._word_vocab = word_vocab
        # Prediction vocabulary = real training words + </s> + <unk>.
        # <s> is excluded: it can only ever appear as left-padding
        # context, never as a predicted token.
        self._prediction_vocab_size = len(word_vocab) + 2

        self._classes = sorted(set(labels))
        self._per_class_models = {}
        for class_id in self._classes:
            ngram_counts: "Counter[Tuple[str, ...]]" = Counter()
            context_counts: "Counter[Tuple[str, ...]]" = Counter()
            for tokens, label in zip(tokenized, labels):
                if label != class_id:
                    continue
                padded = _pad(tokens, self.n)
                for ngram in _ngrams(padded, self.n):
                    ngram_counts[ngram] += 1
                    context_counts[ngram[:-1]] += 1
            self._per_class_models[class_id] = {
                "ngram_counts": ngram_counts,
                "context_counts": context_counts,
            }
        return self

    def _description_length(self, tokens: List[str], class_id: int) -> float:
        """Return the conditional description length (in bits) of
        ``tokens`` under the fitted language model for ``class_id``::

            L(x | M_e) = -sum_i log2 P(w_i | preceding_context, M_e)

        Lower is better (a better fit under that class's model)."""
        model = self._per_class_models[class_id]
        ngram_counts = model["ngram_counts"]
        context_counts = model["context_counts"]
        word_vocab = self._word_vocab
        v = self._prediction_vocab_size
        k = self.k

        normalized = [token if token in word_vocab else _UNK for token in tokens]
        padded = _pad(normalized, self.n)

        total_bits = 0.0
        for ngram in _ngrams(padded, self.n):
            context = ngram[:-1]
            probability = _laplace_probability(
                ngram_count=ngram_counts.get(ngram, 0),
                context_count=context_counts.get(context, 0),
                k=k,
                prediction_vocab_size=v,
            )
            total_bits += -math.log2(probability)
        return total_bits

    def description_lengths(self, texts: Sequence[str]) -> List[Dict[int, float]]:
        """Return, for every input text, the conditional description
        length ``L(x | M_e)`` (in bits) under *every* candidate
        emotion's language model, as a ``{class_id: bits}`` dict.

        This is a diagnostic companion to :meth:`predict` (e.g. to
        inspect the margin between the best and second-best class, or
        to plot score distributions) -- it does not predict anything
        itself. :meth:`predict` is equivalent to taking, per text, the
        key with the minimum value in the corresponding dict here.
        """
        if self._per_class_models is None or self._classes is None:
            raise RuntimeError("MDLClassifier.description_lengths called before fit().")

        results: List[Dict[int, float]] = []
        for text in texts:
            tokens = _tokenize(text)
            results.append({class_id: self._description_length(tokens, class_id) for class_id in self._classes})
        return results

    def predict(self, texts: Sequence[str]) -> List[int]:
        """Predict the class with the shortest description length for
        each text: ``argmin_e L(x | M_e)``.

        Ties (two or more classes with an exactly equal description
        length for a given text) are broken deterministically in favor
        of the **smallest class id** among the tied classes: classes
        are iterated in ascending order when building each text's score
        dict, and Python's ``min`` keeps the first entry it encounters
        that achieves the minimum.
        """
        predictions: List[int] = []
        for scores in self.description_lengths(texts):
            predictions.append(min(scores, key=scores.get))
        return predictions

    def save(self, path: Path) -> None:
        if self._per_class_models is None:
            raise RuntimeError("MDLClassifier.save called before fit().")
        payload = {
            "schema_version": 2,
            "n": self.n,
            "smoothing": self.smoothing,
            "k": self.k,
            "word_vocab": self._word_vocab,
            "prediction_vocab_size": self._prediction_vocab_size,
            "classes": self._classes,
            "per_class_models": {
                class_id: {
                    "ngram_counts": dict(model["ngram_counts"]),
                    "context_counts": dict(model["context_counts"]),
                }
                for class_id, model in self._per_class_models.items()
            },
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Path) -> "MDLClassifier":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        model = cls(n=payload["n"], smoothing=payload["smoothing"], k=payload["k"])
        model._word_vocab = payload["word_vocab"]
        model._prediction_vocab_size = payload["prediction_vocab_size"]
        model._classes = payload["classes"]
        model._per_class_models = {
            class_id: {
                "ngram_counts": Counter(data["ngram_counts"]),
                "context_counts": Counter(data["context_counts"]),
            }
            for class_id, data in payload["per_class_models"].items()
        }
        return model
