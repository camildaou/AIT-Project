"""Experimental variants of the MDL emotion classifier.

Owned by: Camil DAOU.

Every variant reuses :class:`~emotion_ait.mdl_classifier.MDLClassifier`
unchanged -- same tokenization, shared prediction vocabulary ``V``,
n-gram order, add-``k`` smoothing and conditional description length::

    L(x | M_e) = -sum_i log2 P(w_i | w_{i-n+1}, ..., w_{i-1}, M_e)

and differs only in what is added to it, or in how the per-class
models are fitted. The baseline module itself is never modified, so
``mdl`` predictions and scores stay exactly as they were.

All scores are in bits. Ties are broken in favour of the smallest class
id, exactly as in the baseline.

``mdl_prior`` -- :class:`MDLPriorClassifier`
    ``argmin_e [L(x | M_e) - log2 P(e)]`` with ``P(e) = N_e / N``, the
    class frequencies of the *training subset* (validation/test labels
    are never looked at). This is the MAP / two-part code for the pair
    ``(e, x)``: ``-log2 P(e)`` bits to send the label, then ``L(x | M_e)``
    bits to send the text.

``mdl_two_part`` -- :class:`MDLTwoPartClassifier`
    ``argmin_e [L(M_e) + L(x | M_e)]``, where ``L(M_e)`` is the cost of
    transmitting class ``e``'s n-gram count table::

        L(M_e) = gamma(T_e + 1) + log2 C(U, T_e) + sum_g gamma(count_e(g))

    with ``T_e`` distinct n-grams, ``U = (|W| + 1) ** n`` possible
    n-grams over the training words ``W`` (plus ``<s>`` as context /
    ``</s>`` as prediction) and Elias-gamma lengths
    ``gamma(x) = 2 * floor(log2 x) + 1``. Kept for reference: charging a
    whole-dataset model cost on every single text swamps ``L(x | M_e)``
    and collapses predictions onto the class with the smallest table.

``mdl_prequential`` -- :class:`MDLPrequentialClassifier`
    Sequential (prequential) coding of the labelled training subset.
    Protocol, for a fixed processing order ``(x_1, y_1), ..., (x_N, y_N)``:

    1. Start from *empty* per-class count tables (initial portion of size
       0). With add-``k`` smoothing an empty model is already a proper
       code (every token costs ``log2 V`` bits), so all ``N`` training
       observations are coded and ``L_preq`` is a complete code for the
       subset at every size and seed.
    2. For ``t = 1..N``: score ``x_t`` under every class model *as it is
       before seeing* ``(x_t, y_t)``; record those per-class bits, the
       text code length ``L(x_t | M_{y_t}^{(t-1)})`` and the label code
       length ``-log2 P_{t-1}(y_t)``, where
       ``P_{t-1}(e) = (N_e^{(t-1)} + 1) / (t - 1 + K)`` is a sequential
       Laplace estimate over the ``K`` classes of the subset.
    3. Then update the n-gram counts of the true class ``y_t`` only.

    Reported totals::

        L_preq_text  = sum_t L(x_t | M_{y_t}^{(t-1)})
        L_preq_label = sum_t -log2 P_{t-1}(y_t)
        L_preq_joint = L_preq_text + L_preq_label

    The vocabulary ``V`` and the class set are those of the full training
    subset, fixed before coding starts (the alphabet is treated as known
    in advance, exactly as in the baseline). Consequently, the final
    count tables after step 3 are identical to those of a batch
    :class:`MDLClassifier` fit on the same subset, so held-out
    validation/test predictions equal those of ``mdl`` (or of
    ``mdl_prior`` with ``use_prior=True``, whose prior is then the final
    ``N_e / N``). The new quantity is ``L_preq`` itself. Validation and
    test data are only ever scored with these final models; they never
    update any model.

    The processing order is supplied by the caller (the ladder runner
    uses :func:`emotion_ait.sampling.load_ladder_prequential_order`) or,
    if omitted, is a permutation of the rows sorted by ``example_id``
    drawn with ``seed``.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .mdl_classifier import MDLClassifier, _ngrams, _pad, _tokenize


def _class_priors(labels: Sequence[int]) -> Dict[int, float]:
    """``P(e) = N_e / N`` for every class present in ``labels``."""
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    return {class_id: counts[class_id] / total for class_id in sorted(counts)}


def _elias_gamma_bits(x: int) -> int:
    """Length in bits of the Elias-gamma code for a positive integer ``x``."""
    if x < 1:
        raise ValueError(f"Elias-gamma codes are only defined for x >= 1, got {x}.")
    return 2 * (x.bit_length() - 1) + 1


def _log2_binomial(n: float, k: int) -> float:
    """``log2 C(n, k)`` via log-gamma, so it stays finite for huge ``n``."""
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def _model_description_length(ngram_counts: "Counter[Tuple[str, ...]]", word_vocab_size: int, n: int) -> float:
    """Two-part model cost ``L(M_e)`` (in bits) of one class's n-gram table."""
    distinct = len(ngram_counts)
    universe = float(word_vocab_size + 1) ** n
    return (
        _elias_gamma_bits(distinct + 1)
        + _log2_binomial(universe, distinct)
        + sum(_elias_gamma_bits(count) for count in ngram_counts.values())
    )


class _ClassCostMDLClassifier(MDLClassifier):
    """Baseline MDL score plus a per-class constant cost ``C_e`` (bits):
    ``score_e(x) = L(x | M_e) + C_e``. Subclasses define ``C_e``."""

    #: Name of the per-class cost in :meth:`score_breakdown`.
    cost_name = "cost_bits"

    def __init__(self, n: int = 3, smoothing: str = "laplace", k: float = 1.0) -> None:
        super().__init__(n=n, smoothing=smoothing, k=k)
        self._class_cost_bits: Optional[Dict[int, float]] = None

    def _compute_class_costs(self, train_data: pd.DataFrame) -> Dict[int, float]:
        raise NotImplementedError

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "_ClassCostMDLClassifier":
        # validation_data is passed through but, as in the baseline, unused.
        super().fit(train_data, validation_data)
        self._class_cost_bits = self._compute_class_costs(train_data)
        return self

    def score_breakdown(self, texts: Sequence[str]) -> List[Dict[int, Dict[str, float]]]:
        """Per text, per class: ``conditional_bits`` (``L(x | M_e)``), the
        class cost (named :attr:`cost_name`) and ``total_bits``."""
        conditional = super().description_lengths(texts)
        return [
            {
                class_id: {
                    "conditional_bits": bits,
                    self.cost_name: self._class_cost_bits[class_id],
                    "total_bits": bits + self._class_cost_bits[class_id],
                }
                for class_id, bits in scores.items()
            }
            for scores in conditional
        ]

    def description_lengths(self, texts: Sequence[str]) -> List[Dict[int, float]]:
        """Total score ``L(x | M_e) + C_e`` per class; :meth:`predict`
        (inherited) takes the argmin of this."""
        return [
            {class_id: parts["total_bits"] for class_id, parts in breakdown.items()}
            for breakdown in self.score_breakdown(texts)
        ]

    def save(self, path: Path) -> None:
        if self._per_class_models is None:
            raise RuntimeError(f"{type(self).__name__}.save called before fit().")
        with open(path, "wb") as fh:
            pickle.dump({"class": type(self).__name__, "state": self.__dict__}, fh)

    @classmethod
    def load(cls, path: Path) -> "_ClassCostMDLClassifier":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("class") != cls.__name__:
            raise ValueError(f"{path} holds a {payload.get('class')!r}, not a {cls.__name__!r}.")
        model = cls.__new__(cls)
        model.__dict__.update(payload["state"])
        return model


class MDLPriorClassifier(_ClassCostMDLClassifier):
    """``mdl_prior``: ``argmin_e [L(x | M_e) - log2 P(e)]``, ``P(e) = N_e / N``
    estimated from the training subset's labels only."""

    cost_name = "prior_bits"

    def __init__(self, n: int = 3, smoothing: str = "laplace", k: float = 1.0) -> None:
        super().__init__(n=n, smoothing=smoothing, k=k)
        self.class_priors: Optional[Dict[int, float]] = None

    def _compute_class_costs(self, train_data: pd.DataFrame) -> Dict[int, float]:
        self.class_priors = _class_priors(train_data["label"].tolist())
        return {class_id: -math.log2(p) for class_id, p in self.class_priors.items()}


class MDLTwoPartClassifier(_ClassCostMDLClassifier):
    """``mdl_two_part``: ``argmin_e [L(M_e) + L(x | M_e)]`` (see module docstring)."""

    cost_name = "model_bits"

    def _compute_class_costs(self, train_data: pd.DataFrame) -> Dict[int, float]:
        return {
            class_id: _model_description_length(model["ngram_counts"], len(self._word_vocab), self.n)
            for class_id, model in self._per_class_models.items()
        }


class MDLPrequentialClassifier(_ClassCostMDLClassifier):
    """``mdl_prequential``: prequential coding of the training subset,
    then held-out classification with the final models (see module
    docstring for the exact protocol).

    Parameters
    ----------
    use_prior:
        If True, held-out scores add ``-log2 P(e)`` with the final
        ``P(e) = N_e / N`` (as ``mdl_prior``); otherwise they are
        ``L(x | M_e)`` only (as ``mdl``).
    order:
        Processing order as a sequence of ``example_id`` values; must be
        a permutation of ``train_data["example_id"]``.
    seed:
        Used only when ``order`` is None, to permute the rows sorted by
        ``example_id`` (or by position if there is no such column).

    After :meth:`fit`, :attr:`prequential_trace` holds one dict per coded
    observation and :attr:`prequential_summary` the totals.
    """

    cost_name = "prior_bits"

    def __init__(
        self, n: int = 3, smoothing: str = "laplace", k: float = 1.0,
        use_prior: bool = False, order: Optional[Sequence[str]] = None, seed: int = 0,
    ) -> None:
        super().__init__(n=n, smoothing=smoothing, k=k)
        self.use_prior = bool(use_prior)
        self.order = None if order is None else list(order)
        self.seed = int(seed)
        self.class_priors: Optional[Dict[int, float]] = None
        self.prequential_trace: Optional[List[Dict[str, object]]] = None
        self.prequential_summary: Optional[Dict[str, object]] = None

    def _processing_positions(self, train_data: pd.DataFrame) -> List[int]:
        if self.order is not None:
            if "example_id" not in train_data.columns:
                raise KeyError("train_data must have an 'example_id' column when an explicit order is given.")
            ids = train_data["example_id"].tolist()
            position_of = {example_id: i for i, example_id in enumerate(ids)}
            if len(position_of) != len(ids) or len(self.order) != len(ids) or set(self.order) != set(ids):
                raise ValueError("order must be a permutation of train_data's (unique) example_id values.")
            return [position_of[example_id] for example_id in self.order]
        if "example_id" in train_data.columns:
            base = sorted(range(len(train_data)), key=lambda i: str(train_data["example_id"].iloc[i]))
        else:
            base = list(range(len(train_data)))
        return [base[i] for i in np.random.default_rng(self.seed).permutation(len(base))]

    def _update(self, tokens: List[str], class_id: int) -> None:
        """Add one training text's n-grams to ``class_id``'s model only."""
        model = self._per_class_models[class_id]
        for ngram in _ngrams(_pad(tokens, self.n), self.n):
            model["ngram_counts"][ngram] += 1
            model["context_counts"][ngram[:-1]] += 1

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> "MDLPrequentialClassifier":
        # validation_data is accepted for interface compatibility and never
        # used: held-out data must not update the models.
        if train_data.empty:
            raise ValueError("Cannot fit MDLPrequentialClassifier on an empty training set.")
        if "text" not in train_data.columns or "label" not in train_data.columns:
            raise KeyError("train_data must have 'text' and 'label' columns.")

        tokenized = [_tokenize(t) for t in train_data["text"].tolist()]
        labels = [int(label) for label in train_data["label"].tolist()]
        positions = self._processing_positions(train_data)

        # Fixed alphabet: vocabulary and class set of the full training
        # subset, known before coding starts (identical to the baseline).
        word_vocab = set()
        for tokens in tokenized:
            word_vocab.update(tokens)
        self._word_vocab = word_vocab
        self._prediction_vocab_size = len(word_vocab) + 2
        self._classes = sorted(set(labels))
        self._per_class_models = {
            class_id: {"ngram_counts": Counter(), "context_counts": Counter()} for class_id in self._classes
        }

        num_classes = len(self._classes)
        label_counts = {class_id: 0 for class_id in self._classes}
        has_ids = "example_id" in train_data.columns
        trace: List[Dict[str, object]] = []
        cumulative_text = cumulative_label = 0.0
        for t, row in enumerate(positions, start=1):
            tokens, true_class = tokenized[row], labels[row]
            # Predict BEFORE updating with (x_t, y_t).
            bits = {class_id: self._description_length(tokens, class_id) for class_id in self._classes}
            text_bits = bits[true_class]
            label_bits = -math.log2((label_counts[true_class] + 1) / (t - 1 + num_classes))
            cumulative_text += text_bits
            cumulative_label += label_bits
            trace.append({
                "position": t,
                "example_id": train_data["example_id"].iloc[row] if has_ids else row,
                "true_label": true_class,
                "online_predicted_label": min(bits, key=bits.get),
                "bits_by_class": bits,
                "text_bits": text_bits,
                "label_bits": label_bits,
                "joint_bits": text_bits + label_bits,
                "cumulative_text_bits": cumulative_text,
                "cumulative_joint_bits": cumulative_text + cumulative_label,
            })
            # Only now reveal the observation to its true class's model.
            self._update(tokens, true_class)
            label_counts[true_class] += 1

        self.class_priors = {class_id: count / len(labels) for class_id, count in label_counts.items()}
        self._class_cost_bits = {
            class_id: (-math.log2(p) if self.use_prior else 0.0) for class_id, p in self.class_priors.items()
        }
        self.prequential_trace = trace
        processed_ids = [str(row["example_id"]) for row in trace]
        self.prequential_summary = {
            "n_observations": len(trace),
            "n_updates": sum(label_counts.values()),
            "update_counts": dict(label_counts),
            "total_text_bits": cumulative_text,
            "total_label_bits": cumulative_label,
            "total_joint_bits": cumulative_text + cumulative_label,
            "text_bits_by_class": {
                class_id: sum(r["text_bits"] for r in trace if r["true_label"] == class_id)
                for class_id in self._classes
            },
            "mean_text_bits_per_observation": cumulative_text / len(trace),
            "online_accuracy": sum(r["online_predicted_label"] == r["true_label"] for r in trace) / len(trace),
            "order_sha1": hashlib.sha1("\n".join(processed_ids).encode("utf-8")).hexdigest(),
        }
        return self
