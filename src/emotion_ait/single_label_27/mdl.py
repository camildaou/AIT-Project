"""MDL classifier for the 27-class single-label track.

A thin extension of :class:`emotion_ait.mdl_classifier.MDLClassifier` (whose
code and behaviour are untouched): same tokenization, ``<s>``/``</s>``/``<unk>``
conventions, add-``k`` smoothing and conditional description length

    L(x | M_e) = -sum_i log2 P(w_i | w_{i-1}, M_e)
    P(w_i | w_{i-1}, M_e) = (count_e(w_{i-1}, w_i) + k) / (count_e(w_{i-1}) + k * V)

with **one shared vocabulary** built from the current training subset only
(``V`` = its distinct words + ``</s>`` + ``<unk>``; ``<s>`` is context only).
Defaults are the best 4-class setting: bigram, ``k = 1``.

Differences from the baseline:

* **All 27 classes always have a model**, indexed exactly as
  :data:`emotion_ait.single_label_27.data.CLASS_NAMES`. A class with no
  training examples gets an empty model (uniform: ``log2 V`` bits per
  token, so its description lengths stay finite). It is listed in
  :attr:`missing_classes`, a warning is emitted, and it is **excluded from
  the argmin**: a class never seen in training cannot be predicted (an
  empty model would otherwise undercut every trained model on unseen
  bigrams). Its score in :meth:`predict_scores` is ``+inf``.
* **Optional class prior** (``use_prior=False`` by default):
  ``score_e(x) = L(x | M_e) - log2 P(e)``, ``P(e) = N_e / N`` from the
  training subset's labels only. With ``use_prior=False`` the score is
  ``L(x | M_e)``, identical to the baseline's.
* Ties go to the lowest class index.
* :meth:`diagnostics` reports per-class statistics for rare-class analysis.
* **Optional interpolated Kneser-Ney smoothing** (``smoothing="kneser_ney"``,
  any order ``n``) instead of add-``k``. For order ``m`` with context ``h``
  (the last ``m - 1`` tokens) and ``h'`` = ``h`` without its first token::

      P_m(w | h) = (max(c_m(h w) - D_m, 0) + D_m * N1+(h .) * P_{m-1}(w | h')) / c_m(h .)   if c_m(h .) > 0
                 = P_{m-1}(w | h')                                                         otherwise
      P_0(w)     = 1 / V

  At the top order ``m = n``, ``c_n`` are the class's raw n-gram counts. At
  every lower order, ``c_m(g)`` is the continuation count of the m-gram
  ``g``: the number of distinct tokens seen before it in the (m+1)-gram
  types. ``c_m(h .)`` is the sum over ``w`` and ``N1+(h .)`` the number of
  distinct ``w`` after ``h``. For ``n = 2`` this is
  ``P(w|v) = (max(c(v,w) - D, 0) + D N1+(v .) P_1(w)) / c(v)`` with the
  continuation unigram ``P_1``. The recursion ends in the uniform ``1/V``
  over the same prediction vocabulary, so every distribution is proper (sums
  to one) and strictly positive, including for ``</s>`` and ``<unk>``.
  ``kn_backoff="class"`` builds the lower orders from the class's own n-gram
  types; ``"shared"`` pools the n-gram types of the whole training subset
  (still training data only; class models then share their lower-order
  distributions). Each level's discount is estimated from its own counts as
  ``D = n1 / (n1 + 2 n2)`` (``n1``/``n2``: events seen once/twice), clipped
  to ``[0.1, 1]`` (0.75 if undefined), unless ``kn_discount`` fixes it.
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

from ..mdl_classifier import MDLClassifier, _ngrams, _pad, _tokenize
from .data import CLASS_NAMES


class MDL27Classifier(MDLClassifier):
    """27-class conditional MDL classifier (see module docstring).

    Parameters
    ----------
    n, smoothing, k:
        As :class:`MDLClassifier`; defaults ``n=2``, ``"laplace"``, ``k=1.0``.
    use_prior:
        Add ``-log2 P(e)`` (training-subset class frequency) to each score.
    smoothing:
        ``"laplace"`` (add-``k``, default) or ``"kneser_ney"`` (interpolated,
        any order; ``k`` is then unused).
    kn_backoff:
        ``"class"`` or ``"shared"`` lower-order distribution for Kneser-Ney.
    kn_discount:
        Fixed Kneser-Ney discount for every level; ``None`` estimates them.
    """

    def __init__(
        self, n: int = 2, smoothing: str = "laplace", k: float = 1.0, use_prior: bool = False,
        kn_backoff: str = "shared", kn_discount: Optional[float] = None,
    ) -> None:
        if smoothing not in ("laplace", "kneser_ney"):
            raise NotImplementedError(f"Smoothing {smoothing!r} is not implemented; use 'laplace' or 'kneser_ney'.")
        if kn_backoff not in ("class", "shared"):
            raise ValueError(f"kn_backoff must be 'class' or 'shared', got {kn_backoff!r}.")
        if kn_discount is not None and not 0 < kn_discount <= 1:
            raise ValueError(f"kn_discount must be in (0, 1], got {kn_discount}.")
        super().__init__(n=n, smoothing="laplace", k=k)  # base class validates n and k
        self.smoothing = smoothing
        self.kn_backoff = kn_backoff
        self.kn_discount = kn_discount
        self._kn: Optional[Dict[str, object]] = None
        self.use_prior = bool(use_prior)
        self.class_names: List[str] = list(CLASS_NAMES)
        self.missing_classes: List[int] = []
        self.class_counts: Optional[Dict[int, int]] = None
        self.class_priors: Optional[Dict[int, float]] = None

    # --- fitting ---------------------------------------------------------

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "MDL27Classifier":
        # validation_data is accepted for interface compatibility and unused:
        # vocabulary, counts and priors come from train_data only.
        labels = [int(label) for label in train_data["label"]] if "label" in train_data.columns else []
        invalid = sorted({label for label in labels if not 0 <= label < len(self.class_names)})
        if invalid:
            raise ValueError(f"Labels {invalid} are outside the 27-class range 0..{len(self.class_names) - 1}.")
        super().fit(train_data, validation_data)

        counts = Counter(labels)
        self.class_counts = {c: counts.get(c, 0) for c in range(len(self.class_names))}
        self.missing_classes = [c for c, count in self.class_counts.items() if count == 0]
        for c in self.missing_classes:
            self._per_class_models[c] = {"ngram_counts": Counter(), "context_counts": Counter()}
        self._classes = list(range(len(self.class_names)))
        if self.missing_classes:
            warnings.warn(
                f"MDL27Classifier: no training examples for {[self.class_names[c] for c in self.missing_classes]}; "
                "they get empty (uniform) models and are excluded from prediction."
            )
        total = len(labels)
        self.class_priors = {c: count / total for c, count in self.class_counts.items()}
        self._kn = self._fit_kneser_ney() if self.smoothing == "kneser_ney" else None
        return self

    # --- Kneser-Ney --------------------------------------------------------

    def _estimate_discount(self, counts) -> float:
        if self.kn_discount is not None:
            return float(self.kn_discount)
        values = list(counts)
        n1, n2 = values.count(1), values.count(2)
        if n1 + 2 * n2 == 0:
            return 0.75
        return min(1.0, max(0.1, n1 / (n1 + 2 * n2)))

    def _level(self, counts: Dict[tuple, int]) -> Dict[str, object]:
        """One Kneser-Ney order: counts plus per-context totals, types and discount."""
        totals: Counter = Counter()
        types: Counter = Counter()
        for gram, count in counts.items():
            totals[gram[:-1]] += count
            types[gram[:-1]] += 1
        return {"counts": counts, "context_totals": dict(totals), "context_types": dict(types),
                "discount": self._estimate_discount(counts.values()) if counts else None}

    def _lower_levels(self, top_types) -> Dict[int, Dict[str, object]]:
        """Orders n-1 .. 1, each with continuation counts from the order above."""
        levels = {}
        types = set(top_types)
        for m in range(self.n - 1, 0, -1):
            continuation = Counter(gram[1:] for gram in types)
            levels[m] = self._level(dict(continuation))
            types = set(continuation)
        return levels

    def _fit_kneser_ney(self) -> Dict[str, object]:
        shared = None
        if self.kn_backoff == "shared":
            shared = self._lower_levels({g for m in self._per_class_models.values() for g in m["ngram_counts"]})
        per_class = {}
        for c, model in self._per_class_models.items():
            top = self._level(dict(model["ngram_counts"]))
            per_class[c] = {
                "discount": top["discount"],
                "top": top,
                "lower": shared if shared is not None else self._lower_levels(model["ngram_counts"]),
            }
        return {"per_class": per_class, "shared_lower": shared}

    def _kn_probability(self, context: tuple, word: str, class_id: int) -> float:
        """P_KN(word | context) for one class; ``context`` holds the last n-1 tokens."""
        kn = self._kn["per_class"][class_id]
        p = 1.0 / self._prediction_vocab_size
        for m in range(1, self.n + 1):
            level = kn["top"] if m == self.n else kn["lower"][m]
            h = tuple(context[len(context) - (m - 1):]) if m > 1 else ()
            total = level["context_totals"].get(h, 0)
            if total:
                discount = level["discount"]
                seen = max(level["counts"].get(h + (word,), 0) - discount, 0.0)
                p = (seen + discount * level["context_types"][h] * p) / total
        return p

    def _description_length(self, tokens: List[str], class_id: int) -> float:
        if self.smoothing != "kneser_ney":
            return super()._description_length(tokens, class_id)
        normalized = [token if token in self._word_vocab else "<unk>" for token in tokens]
        return -sum(math.log2(self._kn_probability(g[:-1], g[-1], class_id)) for g in _ngrams(_pad(normalized, self.n), self.n))

    @property
    def vocabulary(self) -> List[str]:
        """The shared fitted word vocabulary (sorted; excludes ``</s>``/``<unk>``)."""
        if self._word_vocab is None:
            raise RuntimeError("MDL27Classifier.vocabulary accessed before fit().")
        return sorted(self._word_vocab)

    @property
    def predictable_classes(self) -> List[int]:
        return [c for c in range(len(self.class_names)) if c not in self.missing_classes]

    # --- scoring ---------------------------------------------------------

    def _prior_bits(self, class_id: int) -> float:
        if not self.use_prior:
            return 0.0
        p = self.class_priors[class_id]
        return -math.log2(p) if p > 0 else math.inf

    def score_breakdown(self, texts: Sequence[str]) -> List[Dict[int, Dict[str, float]]]:
        """Per text, per class: ``conditional_bits`` = L(x|M_e) (always finite),
        ``prior_bits`` (0 without prior) and ``total_bits`` (``+inf`` for a
        class missing from training)."""
        rows = []
        for scores in super().description_lengths(texts):
            row = {}
            for c, bits in scores.items():
                prior = self._prior_bits(c)
                total = math.inf if c in self.missing_classes else bits + prior
                row[c] = {"conditional_bits": bits, "prior_bits": prior, "total_bits": total}
            rows.append(row)
        return rows

    def description_lengths(self, texts: Sequence[str]) -> List[Dict[int, float]]:
        """Decision score per class (``total_bits``), keyed by class index."""
        return [{c: parts["total_bits"] for c, parts in row.items()} for row in self.score_breakdown(texts)]

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), 27)`` decision scores in bits, columns in class-index order."""
        return np.array([[row[c] for c in range(len(self.class_names))] for row in self.description_lengths(texts)],
                        dtype=float).reshape(len(texts), len(self.class_names))

    def predict(self, texts: Sequence[str]) -> List[int]:
        """``argmin_e score_e(x)`` over predictable classes; ties -> lowest index."""
        if self._per_class_models is None:
            raise RuntimeError("MDL27Classifier.predict called before fit().")
        eligible = self.predictable_classes
        return [min(eligible, key=lambda c: scores[c]) for scores in self.description_lengths(texts)]

    # --- diagnostics -----------------------------------------------------

    def _oov_counts(self, texts: Sequence[str]) -> Dict[str, int]:
        tokens = [t for text in texts for t in _tokenize(text)]
        oov = sum(1 for t in tokens if t not in self._word_vocab)
        return {"tokens": len(tokens), "oov_tokens": oov}

    def diagnostics(
        self, train_data: pd.DataFrame, held_out: Optional[Dict[str, Sequence[str]]] = None
    ) -> Dict[str, object]:
        """Per-class training statistics and, for each held-out split given as
        ``{name: texts}``, OOV token counts and predictions per class."""
        texts_by_class: Dict[int, List[str]] = {c: [] for c in range(len(self.class_names))}
        for text, label in zip(train_data["text"], train_data["label"]):
            texts_by_class[int(label)].append(text)
        per_class = {}
        for c, name in enumerate(self.class_names):
            texts = texts_by_class[c]
            words = {t for text in texts for t in _tokenize(text)}
            bits = [self._description_length(_tokenize(text), c) for text in texts]
            tokens = sum(len(_tokenize(text)) + 1 for text in texts)  # + </s>
            per_class[name] = {
                "train_examples": len(texts),
                "unique_ngrams": len(self._per_class_models[c]["ngram_counts"]),
                "vocabulary_coverage": len(words) / len(self._word_vocab) if self._word_vocab else 0.0,
                "mean_train_bits_per_example": float(np.mean(bits)) if bits else None,
                "mean_train_bits_per_token": (sum(bits) / tokens) if tokens else None,
                "prior": self.class_priors[c],
            }
        report: Dict[str, object] = {
            "n": self.n, "k": self.k, "use_prior": self.use_prior, "smoothing": self.smoothing,
            "kn_backoff": self.kn_backoff if self._kn else None,
            "kn_discounts": {self.class_names[c]: v["discount"] for c, v in self._kn["per_class"].items()} if self._kn else None,
            "shared_vocabulary_size": len(self._word_vocab),
            "prediction_vocabulary_size": self._prediction_vocab_size,
            "missing_classes": [self.class_names[c] for c in self.missing_classes],
            "per_class": per_class,
        }
        for split, texts in (held_out or {}).items():
            predictions = Counter(self.predict(texts))
            report[split] = {**self._oov_counts(texts),
                             "predictions_per_class": {name: predictions.get(c, 0) for c, name in enumerate(self.class_names)}}
            report[split]["oov_rate"] = report[split]["oov_tokens"] / report[split]["tokens"] if report[split]["tokens"] else 0.0
        return report

    # --- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        if self._per_class_models is None:
            raise RuntimeError("MDL27Classifier.save called before fit().")
        with open(path, "wb") as fh:
            pickle.dump({"class": type(self).__name__, "state": self.__dict__}, fh)

    @classmethod
    def load(cls, path: Path) -> "MDL27Classifier":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("class") != cls.__name__:
            raise ValueError(f"{path} holds a {payload.get('class')!r}, not a {cls.__name__!r}.")
        model = cls.__new__(cls)
        model.__dict__.update(payload["state"])
        return model
